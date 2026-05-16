# =============================================================================
# stack.py
# -----------------------------------------------------------------------------
# Two-level stacking ensemble for the OSF no-show prediction problem.
#
# What is stacking?
#   Instead of relying on one model, we train multiple "base" models and then
#   train a second "meta" model that learns the best way to combine their
#   predictions. Each base model has different strengths:
#
#     CatBoost  — native ordered target encoding for categoricals; excels at
#                 high-cardinality string features out of the box
#     LightGBM  — leaf-wise tree growth; finds different decision boundaries
#                 than CatBoost even on the same data; fast
#     XGBoost   — level-wise growth with strong regularisation; tends to be
#                 more conservative, capturing yet another viewpoint
#
#   The meta-model (Logistic Regression) then learns something like:
#     "When CatBoost says 0.8 and LightGBM says 0.6, the true answer is 0.75"
#   It corrects for each model's systematic biases.
#
# Why OOF (Out-of-Fold) predictions?
#   If we trained a base model on all training data and then used its training
#   predictions to train the meta-model, the meta-model would just learn to
#   trust whichever base model memorised the training data most. The base models
#   would look perfect on training rows but fail on test rows.
#
#   Instead, for each training row we use the base model's prediction for that
#   row when it was in the HELD-OUT fold — a row it had never seen during
#   training. This gives honest, unbiased predictions that reflect real
#   generalisation performance.
#
# Pipeline:
#   Level 0 (base models — trained with 5-fold CV):
#     CatBoost  → oof_catboost,  test_preds_catboost
#     LightGBM  → oof_lgbm,      test_preds_lgbm
#     XGBoost   → oof_xgb,       test_preds_xgb
#
#   Level 1 (meta model — trained on OOF predictions):
#     LogReg([oof_catboost, oof_lgbm, oof_xgb]) → final test predictions
#
# Categorical handling:
#   CatBoost   — raw string columns (native support)
#   LightGBM   — OrdinalEncoder (LightGBM handles integer-encoded categories
#                better than raw strings when using its categorical feature API;
#                simpler than manual target encoding and still effective)
#   XGBoost    — OrdinalEncoder (XGBoost requires numeric inputs only)
#
# Output:
#   submissions/<output_name>          — final stacked submission
#   results/stack_oof_predictions.csv  — OOF preds from all 3 base models
# =============================================================================

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

from data_utils import load_data, get_feature_target_split, get_cat_features, prepare_test_data

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SUBMISSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submissions")
N_FOLDS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill_cat_nulls(X, cat_features):
    """Replace NaN in categorical columns with 'Missing' (required by CatBoost)."""
    X = X.copy()
    for col in cat_features:
        X[col] = X[col].fillna("Missing")
    return X


def _ordinal_encode(X_train, X_val, X_test_full, cat_features):
    """
    Fit OrdinalEncoder on X_train and apply to X_train, X_val, X_test_full.

    Why fit only on X_train?
      The encoder must not see validation or test data during fitting, otherwise
      it would encode unseen categories in a way that leaks future information.
      unknown_value=-1 handles categories in val/test that weren't in train.

    Returns encoded copies of all three DataFrames.
    """
    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        # dtype float so NaN is preserved for any remaining nulls in numeric cols
        dtype=float,
    )
    X_tr = X_train.copy()
    X_v = X_val.copy()
    X_te = X_test_full.copy()

    # Fill nulls in categoricals before encoding (OrdinalEncoder can't handle NaN)
    for col in cat_features:
        X_tr[col] = X_tr[col].fillna("Missing")
        X_v[col] = X_v[col].fillna("Missing")
        X_te[col] = X_te[col].fillna("Missing")

    X_tr[cat_features] = enc.fit_transform(X_tr[cat_features])
    X_v[cat_features] = enc.transform(X_v[cat_features])
    X_te[cat_features] = enc.transform(X_te[cat_features])

    return X_tr, X_v, X_te


# ---------------------------------------------------------------------------
# Level-0 trainers
# ---------------------------------------------------------------------------

def _train_catboost_fold(X_train, y_train, X_val, y_val, cat_features):
    """Train one CatBoost fold and return (val_preds, model)."""
    model = CatBoostClassifier(
        iterations=757,
        learning_rate=0.03416,
        depth=7,
        l2_leaf_reg=7.4877,
        bagging_temperature=0.1755,
        eval_metric="AUC",
        cat_features=cat_features,
        random_seed=42,
        task_type="GPU",
        verbose=0,
        early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    return model.predict_proba(X_val)[:, 1], model


def _train_lgbm_fold(X_train, y_train, X_val, y_val, cat_features):
    """
    Train one LightGBM fold and return (val_preds, model).

    LightGBM with categorical_feature uses its internal split algorithm for
    integer-encoded categoricals, which is faster and often better than
    one-hot encoding.
    """
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,        # 2^depth-1 for depth≈6; controls tree complexity
        min_child_samples=20, # minimum data per leaf; prevents overfitting on small groups
        subsample=0.8,        # row subsampling for variance reduction
        colsample_bytree=0.8, # feature subsampling for decorrelation from CatBoost
        random_state=42,
        verbose=-1,           # suppress output
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=-1),  # suppress per-iteration logs
        ],
        categorical_feature=cat_features,
    )
    return model.predict_proba(X_val)[:, 1], model


def _train_xgb_fold(X_train, y_train, X_val, y_val):
    """
    Train one XGBoost fold and return (val_preds, model).

    XGBoost requires fully numeric input (no strings), so we pass
    ordinal-encoded features. scale_pos_weight handles the ~5% class imbalance
    by up-weighting the minority (no-show) class during training.
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos  # ~19 for 5% no-show rate

    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,   # compensate for class imbalance
        eval_metric="auc",
        random_state=42,
        verbosity=0,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model.predict_proba(X_val)[:, 1], model


# ---------------------------------------------------------------------------
# Main stacking function
# ---------------------------------------------------------------------------

def run_stacking(output_name="submission_stack.csv"):
    """
    Full two-level stacking pipeline.

    Steps:
      1. Load data
      2. Run 5-fold CV for each base model, collecting OOF train predictions
         and fold-averaged test predictions
      3. Train meta-model (Logistic Regression) on the 3-column OOF matrix
      4. Generate final test predictions via the meta-model
      5. Save submission
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

    # Step 1: Load data
    train_df, test_df, _ = load_data()
    X, y, train_ids = get_feature_target_split(train_df)
    X_test_raw, test_ids = prepare_test_data(test_df)
    cat_features = get_cat_features(X)

    n_train = len(X)
    n_test = len(X_test_raw)

    # Accumulators for OOF predictions (one per row of training set)
    oof_cat = np.zeros(n_train)
    oof_lgb = np.zeros(n_train)
    oof_xgb = np.zeros(n_train)

    # Accumulators for test predictions (averaged across folds)
    test_cat = np.zeros(n_test)
    test_lgb = np.zeros(n_test)
    test_xgb = np.zeros(n_test)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    print(f"\n{'='*60}")
    print(f"Running {N_FOLDS}-fold stacking: CatBoost + LightGBM + XGBoost")
    print(f"{'='*60}\n")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"--- Fold {fold}/{N_FOLDS} ---")

        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        # ---------------------------------------------------------------------
        # CatBoost — raw string categoricals
        # ---------------------------------------------------------------------
        X_tr_cat = _fill_cat_nulls(X.iloc[train_idx], cat_features)
        X_val_cat = _fill_cat_nulls(X.iloc[val_idx], cat_features)
        X_test_cat = _fill_cat_nulls(X_test_raw, cat_features)

        val_preds_cat, model_cat = _train_catboost_fold(
            X_tr_cat, y_train_fold, X_val_cat, y_val_fold, cat_features
        )
        oof_cat[val_idx] = val_preds_cat
        test_cat += model_cat.predict_proba(X_test_cat)[:, 1] / N_FOLDS
        print(f"  CatBoost  AUC: {roc_auc_score(y_val_fold, val_preds_cat):.4f}")

        # ---------------------------------------------------------------------
        # LightGBM — ordinal-encoded categoricals
        # ---------------------------------------------------------------------
        X_tr_lgb, X_val_lgb, X_test_lgb = _ordinal_encode(
            X.iloc[train_idx], X.iloc[val_idx], X_test_raw, cat_features
        )
        # LightGBM expects integer-typed categorical columns
        cat_idx = [X_tr_lgb.columns.get_loc(c) for c in cat_features]

        val_preds_lgb, model_lgb = _train_lgbm_fold(
            X_tr_lgb, y_train_fold, X_val_lgb, y_val_fold, cat_idx
        )
        oof_lgb[val_idx] = val_preds_lgb
        test_lgb += model_lgb.predict_proba(X_test_lgb)[:, 1] / N_FOLDS
        print(f"  LightGBM  AUC: {roc_auc_score(y_val_fold, val_preds_lgb):.4f}")

        # ---------------------------------------------------------------------
        # XGBoost — ordinal-encoded categoricals + scale_pos_weight
        # ---------------------------------------------------------------------
        X_tr_xgb, X_val_xgb, X_test_xgb = _ordinal_encode(
            X.iloc[train_idx], X.iloc[val_idx], X_test_raw, cat_features
        )

        val_preds_xgb, model_xgb = _train_xgb_fold(
            X_tr_xgb, y_train_fold, X_val_xgb, y_val_fold
        )
        oof_xgb[val_idx] = val_preds_xgb
        test_xgb += model_xgb.predict_proba(X_test_xgb)[:, 1] / N_FOLDS
        print(f"  XGBoost   AUC: {roc_auc_score(y_val_fold, val_preds_xgb):.4f}")

    # Step 3: Report base model OOF scores
    print(f"\n--- Base Model OOF AUCs ---")
    print(f"  CatBoost : {roc_auc_score(y, oof_cat):.4f}")
    print(f"  LightGBM : {roc_auc_score(y, oof_lgb):.4f}")
    print(f"  XGBoost  : {roc_auc_score(y, oof_xgb):.4f}")

    # Step 4: Weighted average blend
    # We use a simple weighted average instead of a meta-model because
    # Logistic Regression can be pulled down by weaker models (XGBoost here).
    # Weights are proportional to each model's OOF AUC score, so the stronger
    # model (CatBoost) automatically gets more influence.
    auc_cat = roc_auc_score(y, oof_cat)
    auc_lgb = roc_auc_score(y, oof_lgb)
    auc_xgb = roc_auc_score(y, oof_xgb)
    total = auc_cat + auc_lgb + auc_xgb
    w_cat = auc_cat / total
    w_lgb = auc_lgb / total
    w_xgb = auc_xgb / total

    print(f"\n--- AUC-weighted blend ---")
    print(f"  Weight  CatBoost={w_cat:.3f}  LightGBM={w_lgb:.3f}  XGBoost={w_xgb:.3f}")

    oof_stack = w_cat * oof_cat + w_lgb * oof_lgb + w_xgb * oof_xgb
    stack_auc = roc_auc_score(y, oof_stack)
    print(f"  Stack OOF AUC: {stack_auc:.4f}")

    # Step 5: Generate final test predictions
    final_preds = w_cat * test_cat + w_lgb * test_lgb + w_xgb * test_xgb

    # Save OOF predictions for analysis
    oof_df = pd.DataFrame({
        "ID": train_ids.values,
        "oof_catboost": oof_cat,
        "oof_lgbm": oof_lgb,
        "oof_xgb": oof_xgb,
        "oof_stack": oof_stack,
        "y_true": y.values,
    })
    oof_path = os.path.join(RESULTS_DIR, "stack_oof_predictions.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"\nOOF predictions saved to: results/stack_oof_predictions.csv")

    # Save individual OOF + test predictions as .npy for future ensemble use
    np.save(os.path.join(RESULTS_DIR, "stack_cb_oof.npy"),   oof_cat)
    np.save(os.path.join(RESULTS_DIR, "stack_lgbm_oof.npy"), oof_lgb)
    np.save(os.path.join(RESULTS_DIR, "stack_xgb_oof.npy"),  oof_xgb)
    np.save(os.path.join(RESULTS_DIR, "stack_cb_test.npy"),   test_cat)
    np.save(os.path.join(RESULTS_DIR, "stack_lgbm_test.npy"), test_lgb)
    np.save(os.path.join(RESULTS_DIR, "stack_xgb_test.npy"),  test_xgb)
    print("  Saved stack_cb/lgbm/xgb _oof.npy + _test.npy")

    # Save submission
    sub_df = pd.DataFrame({"ID": test_ids.values, "NO_SHOW_FLG": final_preds})
    out_path = os.path.join(SUBMISSIONS_DIR, output_name)
    sub_df.to_csv(out_path, index=False)
    print(f"Submission saved to: submissions/{output_name}  ({len(sub_df):,} rows)")
    print(f"Predicted no-show rate: {final_preds.mean():.4f}")
