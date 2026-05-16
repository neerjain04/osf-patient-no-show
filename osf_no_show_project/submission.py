# =============================================================================
# submission.py
# -----------------------------------------------------------------------------
# Generates the final Kaggle submission file by training CatBoost on the full
# training dataset and predicting no-show probabilities for every test row.
#
# Two modes (controlled by the use_ensemble flag):
#
#   use_ensemble=True  (default)
#     Train N_FOLDS separate models, each on a different 4/5 of the training
#     data, then average their test predictions. This is called "fold averaging"
#     or a "bagged ensemble". It reduces prediction variance because each model
#     sees slightly different data. This is the recommended mode for submission.
#
#   use_ensemble=False
#     Train a single model on 100% of the training data with verbose output.
#     Faster to run but slightly higher variance in predictions.
#
# Output: submissions/<output_name>  with columns [ID, NO_SHOW_FLG]
#   - NO_SHOW_FLG contains predicted probabilities (0.0–1.0), not hard labels.
#   - Kaggle evaluates submissions with ROC-AUC, which requires probabilities.
# =============================================================================

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier

from data_utils import load_data, get_feature_target_split, get_cat_features, prepare_test_data

# Folder where submission CSVs are written.
SUBMISSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submissions")
N_FOLDS = 5   # CV folds per seed
SEEDS = [0, 7, 42, 123, 456, 999, 1337, 2024, 31337, 77777]  # 10 seeds — averaged to reduce prediction variance

# Best params from v2 (the highest LB submission so far: 0.78166).
_MODEL_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.03,
    "depth": 8,
}


def generate_submission(output_name="submission.csv", use_ensemble=True):
    # Step 1: Load all data.
    train_df, test_df, _ = load_data()

    # Full training features and labels — we use ALL of them here (no held-out
    # validation set) because every row helps the final model generalise better.
    X_train_full, y_train_full, _ = get_feature_target_split(train_df)

    # Test features and their IDs (IDs go into the submission file).
    X_test, test_ids = prepare_test_data(test_df)

    cat_features = get_cat_features(X_train_full)

    # Step 2: Fill NaN values in categorical columns with the string "Missing".
    # CatBoost cannot process actual NaN values in columns declared as categorical.
    for col in cat_features:
        X_train_full[col] = X_train_full[col].fillna("Missing")
        X_test[col] = X_test[col].fillna("Missing")

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Mode A: Seed-averaged fold ensemble (recommended)
    # Trains N_FOLDS × len(SEEDS) models. Each seed varies both the CV split
    # and the model initialisation, so averaging across seeds reduces variance
    # more than folds alone.
    # -------------------------------------------------------------------------
    if use_ensemble:
        total_models = N_FOLDS * len(SEEDS)
        print(f"\nTraining {N_FOLDS}-fold × {len(SEEDS)}-seed CatBoost ensemble ({total_models} models)...")

        test_preds_all = np.zeros(len(X_test))

        for s_idx, seed in enumerate(SEEDS, 1):
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            for fold, (train_idx, _) in enumerate(skf.split(X_train_full, y_train_full), 1):
                X_tr = X_train_full.iloc[train_idx].copy()
                y_tr = y_train_full.iloc[train_idx]

                model = CatBoostClassifier(
                    **_MODEL_PARAMS,
                    eval_metric="Logloss",
                    cat_features=cat_features,
                    random_seed=seed,
                    task_type="GPU",
                    verbose=0,
                )
                model.fit(X_tr, y_tr)

                fold_preds = model.predict_proba(X_test)[:, 1]
                test_preds_all += fold_preds / total_models
                print(f"  Seed {s_idx}/{len(SEEDS)}, Fold {fold}/{N_FOLDS} done.")

        test_preds = test_preds_all

    # -------------------------------------------------------------------------
    # Mode B: Single model on full training data
    # -------------------------------------------------------------------------
    else:
        print("\nTraining CatBoost on full training set...")
        model = CatBoostClassifier(
            **_MODEL_PARAMS,
            eval_metric="Logloss",
            cat_features=cat_features,
            random_seed=42,
            task_type="GPU",
            verbose=100,
        )
        model.fit(X_train_full, y_train_full)
        test_preds = model.predict_proba(X_test)[:, 1]

    # Step 3: Build the submission dataframe and save it.
    # The format matches sample_submission.csv: columns ID and NO_SHOW_FLG.
    submission_df = pd.DataFrame({"ID": test_ids.values, "NO_SHOW_FLG": test_preds})
    out_path = os.path.join(SUBMISSIONS_DIR, output_name)
    submission_df.to_csv(out_path, index=False)
    print(f"\nSubmission saved to submissions/{output_name}  ({len(submission_df):,} rows)")
    # Sanity check: predicted no-show rate should be in the same ballpark as the
    # training set no-show rate (~5%). A very different number suggests a bug.
    print(f"Predicted no-show rate: {test_preds.mean():.4f}")


if __name__ == "__main__":
    generate_submission()
