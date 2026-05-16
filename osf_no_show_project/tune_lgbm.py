# =============================================================================
# tune_lgbm.py
# -----------------------------------------------------------------------------
# Optuna hyperparameter search for LightGBM.
#
# Strategy:
#   - 30% stratified subsample for speed (same as CatBoost tuning)
#   - 3-fold CV per trial, 100 trials total
#   - GPU via Intel Xe iGPU (OpenCL)
#   - Saves best params to results/best_lgbm_params.json
#   - blend.py auto-loads these params on next run
#
# Run:
#   python main.py --tune-lgbm
#   python main.py --tune-lgbm --lgbm-trials 150
# =============================================================================

import os
import json
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb

from data_utils import load_data, get_feature_target_split, get_cat_features

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N_TRIALS = 100
N_FOLDS = 3
SUBSAMPLE_FRAC = 0.30   # use 30% of data per trial — same strategy as CatBoost tuning


def _encode(X_train, X_val, cat_features):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_train[cat_features] = enc.fit_transform(X_train[cat_features]).astype(int)
    X_val[cat_features] = enc.transform(X_val[cat_features]).astype(int)
    return X_train, X_val


def objective(trial, X, y, cat_features):
    # ---- Search space ----
    # num_leaves: key complexity param for LGBM. Higher = more complex model.
    # Default is 31; we go up to 255 to allow richer trees.
    num_leaves = trial.suggest_int("num_leaves", 31, 255)

    # learning_rate: log scale so small values (0.01) and large (0.2) explored equally
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)

    # n_estimators: capped at 1500 to keep trials fast; early stopping will
    # usually stop well before this anyway
    n_estimators = trial.suggest_int("n_estimators", 300, 1500)

    # min_child_samples: minimum samples per leaf — higher = more regularisation
    min_child_samples = trial.suggest_int("min_child_samples", 10, 200)

    # subsample: fraction of rows to use per tree (row subsampling)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)

    # colsample_bytree: fraction of columns to use per tree
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)

    # reg_alpha (L1) and reg_lambda (L2): regularisation on leaf weights
    reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
    reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)

    # max_depth: -1 means unlimited (let num_leaves control complexity).
    # Optionally cap to prevent very deep trees.
    max_depth = trial.suggest_int("max_depth", 3, 10)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr_raw = X.iloc[train_idx].copy()
        X_val_raw = X.iloc[val_idx].copy()
        y_tr = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # Fill missing categoricals before encoding
        for col in cat_features:
            X_tr_raw[col] = X_tr_raw[col].fillna("Missing")
            X_val_raw[col] = X_val_raw[col].fillna("Missing")

        X_tr_enc, X_val_enc = _encode(X_tr_raw, X_val_raw, cat_features)

        model = lgb.LGBMClassifier(
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            max_depth=max_depth,
            device="gpu",
            n_jobs=-1,
            verbose=-1,
            random_state=42,
        )

        model.fit(
            X_tr_enc, y_tr,
            categorical_feature=cat_features,
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            eval_set=[(X_val_enc, y_val)],
            eval_metric="auc",
        )

        val_preds = model.predict_proba(X_val_enc)[:, 1]
        fold_scores.append(roc_auc_score(y_val, val_preds))

        # Median pruning: stop trial early if it's clearly worse than median
        trial.report(np.mean(fold_scores), fold_idx)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(fold_scores)


def run_lgbm_tuning(n_trials=N_TRIALS):
    train_df, _, _ = load_data()
    X, y, _ = get_feature_target_split(train_df)
    cat_features = get_cat_features(X)

    # 30% stratified subsample — representative but 3x faster per trial
    _, X_sub, _, y_sub = _stratified_subsample(X, y, frac=SUBSAMPLE_FRAC)

    print(f"\nLightGBM Optuna tuning: {n_trials} trials × {N_FOLDS} folds")
    print(f"Subsample size: {len(X_sub):,} rows ({SUBSAMPLE_FRAC*100:.0f}% of training data)")
    print("GPU: Intel Xe iGPU (OpenCL)\n")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )

    study.optimize(
        lambda trial: objective(trial, X_sub, y_sub, cat_features),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\nBest trial: #{best.number}")
    print(f"  AUC: {best.value:.6f}")
    print(f"  Params: {best.params}")

    # Save params + metadata
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {"best_cv_auc": best.value, **best.params}
    params_path = os.path.join(RESULTS_DIR, "best_lgbm_params.json")
    with open(params_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {params_path}")

    # Save full trial results for inspection
    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(RESULTS_DIR, "optuna_lgbm_results.csv"), index=False)
    print("Full trial results saved to results/optuna_lgbm_results.csv")

    return best.params


def _stratified_subsample(X, y, frac):
    """Return (X_rest, X_sub, y_rest, y_sub) keeping class balance."""
    from sklearn.model_selection import train_test_split
    X_rest, X_sub, y_rest, y_sub = train_test_split(
        X, y, test_size=frac, stratify=y, random_state=42
    )
    return X_rest, X_sub, y_rest, y_sub


if __name__ == "__main__":
    run_lgbm_tuning()
