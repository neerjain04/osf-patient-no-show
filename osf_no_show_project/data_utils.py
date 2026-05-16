# =============================================================================
# data_utils.py
# -----------------------------------------------------------------------------
# Central data-loading and preparation module.
# Every other file in this project imports from here, so all path and column
# constants live in one place. If a CSV is renamed or moved, fix it here only.
# =============================================================================

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# os.path.abspath(__file__) gives the absolute path of THIS file.
# ".." steps one level up from osf_no_show_project/ to the workspace root
# where train.csv, test.csv, etc. actually live.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------
# Using constants avoids typos when referencing these columns throughout the
# project. Change them here and every file picks up the update automatically.
TARGET = "NO_SHOW_FLG"   # binary label: 1 = patient did not show up, 0 = showed / cancelled
ID_COL = "ID"             # unique appointment identifier, not a predictive feature


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------
def load_data():
    # Read all three CSVs from the parent directory.
    # train.csv  — labelled rows used for model training (has NO_SHOW_FLG).
    # test.csv   — unlabelled rows we need to predict for Kaggle submission.
    # metaData.csv — data dictionary describing each feature (reference only).
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    # metaData.csv contains special characters (e.g. degree symbol) that are
    # encoded in latin-1, not UTF-8, so we specify the encoding explicitly.
    meta_df = pd.read_csv(os.path.join(DATA_DIR, "metaData.csv"), encoding="latin-1")

    # Pandas 2.x may infer StringDtype for string columns instead of object.
    # CatBoost requires object dtype for categorical columns, so normalise here
    # once so every downstream file is unaffected.
    for df in (train_df, test_df):
        for col in df.select_dtypes(include="string").columns:
            df[col] = df[col].astype(object)

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df, meta_df


# ---------------------------------------------------------------------------
# get_feature_target_split
# ---------------------------------------------------------------------------
def get_feature_target_split(df):
    # Separate the dataframe into three parts:
    #   y    — the target column we want the model to predict
    #   ids  — the ID column used only for submission; never feed it to a model
    #          because it is an arbitrary key with no predictive meaning
    #   X    — all remaining columns, which are the actual model features
    y = df[TARGET]
    ids = df[ID_COL]
    X = df.drop(columns=[TARGET, ID_COL])  # drop both non-feature columns
    return X, y, ids


# ---------------------------------------------------------------------------
# get_cat_features
# ---------------------------------------------------------------------------
def get_cat_features(X):
    # CatBoost needs to know which columns are categorical so it can apply its
    # internal ordered target encoding instead of treating them as numbers.
    # Pandas stores string columns with dtype=object, so we detect them that way.
    # In this dataset virtually all features are string categories.
    return [col for col in X.columns if X[col].dtype == object]


# ---------------------------------------------------------------------------
# prepare_test_data
# ---------------------------------------------------------------------------
def prepare_test_data(test_df):
    # The test CSV has no NO_SHOW_FLG column (that's what we're predicting),
    # so we only need to strip the ID column before passing data to the model.
    ids = test_df[ID_COL]               # save IDs — needed for submission file
    X_test = test_df.drop(columns=[ID_COL])  # features only, no label or ID
    return X_test, ids


# ---------------------------------------------------------------------------
# add_interaction_features
# ---------------------------------------------------------------------------
def add_interaction_features(X):
    # Why interaction features?
    #   CatBoost's ordered target encoding learns each feature independently.
    #   It cannot automatically discover that the COMBINATION of two features
    #   is more predictive than either one alone. By concatenating two columns
    #   into a single string, we create a new categorical feature that encodes
    #   the joint signal explicitly.
    #
    # Example: PATIENT_NOSHOWRATE_CATEGORY="High" + DAYS_BETWEEN_CATEGORY="Long"
    #   → "High_x_Long"  ← CatBoost learns no-show rate for THIS exact pairing,
    #     which is far more predictive than knowing each column's rate separately.
    #
    # All new features are strings (dtype=object), so get_cat_features() will
    # automatically include them in the CatBoost cat_features list.
    #
    # IMPORTANT: call this function on BOTH train and test with the same logic,
    # and call it BEFORE get_cat_features() so the new columns are included.

    X = X.copy()  # avoid mutating the caller's dataframe

    # Newer pandas (2.x) can infer StringDtype for string columns during .copy()
    # and .astype(str) calls. CatBoost requires object dtype for categorical
    # columns, so we normalise everything to object up front.
    for col in X.select_dtypes(include="string").columns:
        X[col] = X[col].astype(object)

    def concat(col_a, col_b):
        # fillna("Missing") before concatenation so NaN becomes "Missing_x_Value"
        # rather than "nan_x_Value", which is consistent with how we handle
        # missing values elsewhere. Cast to str for any numeric columns, then
        # explicitly to object so the result is never StringDtype.
        a = X[col_a].fillna("Missing").astype(str).astype(object)
        b = X[col_b].fillna("Missing").astype(str).astype(object)
        return (a + "_x_" + b).astype(object)

    # 1. Patient's own no-show history × days between booking and appointment.
    #    Rationale: a habitual no-shower who booked far in advance is the
    #    highest-risk appointment in the dataset.
    X["PATIENT_NOSHOWRATE_x_DAYS_BETWEEN"] = concat(
        "PATIENT_NOSHOWRATE_CATEGORY", "DAYS_BETWEEN_CATEGORY"
    )

    # 2. Patient's no-show history × MyChart activation.
    #    Rationale: patients with reminders/portal access (MyChart=1) AND a
    #    high no-show rate are a distinct, high-risk subgroup — they have been
    #    reminded but still tend not to show up.
    X["PATIENT_NOSHOWRATE_x_MYCHART"] = concat(
        "PATIENT_NOSHOWRATE_CATEGORY", "MYCHART_ACTIVATED"
    )

    # 3. Appointment hour × day of week.
    #    Rationale: Monday 8am and Friday 4pm have very different no-show rates.
    #    This captures scheduling time-slot risk that neither column encodes alone.
    X["HOUR_x_DAY_OF_WEEK"] = concat("HOUR_CODE", "DAY_OF_WEEK_CODE")

    # 4. Department no-show rate × patient no-show rate.
    #    Rationale: a high-risk patient in a high-risk department is a doubly
    #    strong signal. The interaction makes this double-risk explicit.
    X["DEPT_NOSHOWRATE_x_PATIENT_NOSHOWRATE"] = concat(
        "DEPT_NOSHOW_RATE_CATEGORY", "PATIENT_NOSHOWRATE_CATEGORY"
    )

    # 5. Visit type × days between.
    #    Rationale: long-wait follow-up appointments have very different no-show
    #    dynamics than same-day urgent visits. Visit type and wait time together
    #    capture appointment urgency, which strongly predicts attendance.
    X["VISIT_TYPE_x_DAYS_BETWEEN"] = concat("VISIT_TYPE", "DAYS_BETWEEN_CATEGORY")

    # 6. Age × patient no-show rate.
    #    Rationale: AGE_CATEGORY and PATIENT_NOSHOWRATE_CATEGORY are the #2 and
    #    #3 most important features individually. Their interaction captures that
    #    a young patient with a high no-show history is far riskier than either
    #    signal alone suggests.
    X["AGE_x_PATIENT_NOSHOWRATE"] = concat(
        "AGE_CATEGORY", "PATIENT_NOSHOWRATE_CATEGORY"
    )

    # 7. Patient avg appt-to-doc wait × days between booking and appointment.
    #    Rationale: PATIENT_AVG_APPT2DOC_CATEGORY is the single most important
    #    feature. Pairing it with how far in advance the appointment was booked
    #    captures whether a patient who already experiences long in-clinic waits
    #    is also booking far out — a compounding frustration signal.
    X["PATIENT_APPT2DOC_x_DAYS_BETWEEN"] = concat(
        "PATIENT_AVG_APPT2DOC_CATEGORY", "DAYS_BETWEEN_CATEGORY"
    )

    return X


# ---------------------------------------------------------------------------
# add_frequency_features
# ---------------------------------------------------------------------------
def add_frequency_features(X_train, X_test):
    """Add frequency encoding for every categorical column.

    For each categorical column, computes the proportion of training rows that
    have each category value, then maps that proportion onto both train and test.
    This gives CatBoost a numeric signal about how common/rare each category is
    — a genuinely different signal type from its internal target encoding.

    Frequencies are computed on X_train only (then mapped to X_test) to avoid
    leaking test distribution into train.

    New columns are named <COL>_FREQ and are float64 (numeric, not categorical),
    so CatBoost treats them as regular split-based features.
    """
    X_train = X_train.copy()
    X_test  = X_test.copy()

    cat_cols = [col for col in X_train.columns if X_train[col].dtype == object]

    for col in cat_cols:
        freq_map = X_train[col].value_counts(normalize=True)
        new_col  = f"{col}_FREQ"
        X_train[new_col] = X_train[col].map(freq_map).fillna(0.0).astype(float)
        X_test[new_col]  = X_test[col].map(freq_map).fillna(0.0).astype(float)

    return X_train, X_test


# ---------------------------------------------------------------------------
# Quick sanity check — run this file directly to verify everything loads
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_df, test_df, meta_df = load_data()
    X, y, ids = get_feature_target_split(train_df)
    cat_features = get_cat_features(X)
    # Print class balance — the dataset is heavily imbalanced (~95% showed up)
    # so ROC-AUC is a much better metric than accuracy.
    print(f"Features: {X.shape[1]}, Target distribution:\n{y.value_counts()}")
    print(f"Categorical features ({len(cat_features)}): {cat_features}")
    X_test, test_ids = prepare_test_data(test_df)
    print(f"Test features: {X_test.shape}")
