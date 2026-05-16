# =============================================================================
# blend_te_lgbm.py
# -----------------------------------------------------------------------------
# Target-Encoded LightGBM + CatBoost blend / stack.
#
# Pipeline:
#   1. Train CatBoost ensemble (v10 params: lr=0.01, iter=3000, depth=8).
#      Caches OOF + test preds to results/cb_v10_oof.npy so rerunning the
#      TE-LGBM portion doesn't require retraining CatBoost.
#
#   2. Train Target-Encoded LightGBM ensemble.
#      Uses sklearn TargetEncoder(cv=5) — leakage-safe OOF target encoding.
#      For each (seed, fold): TE is fit on that fold's train only, applied to
#      val. Test set uses a TE fit on the full training data.
#
#   3. Gate checks:
#       Gate 1: TE-LGBM OOF AUC must beat ordinal LGBM baseline (~0.762)
#       Gate 2: correlation(CB OOF, TE-LGBM OOF) must be < 0.990
#
#   4a. Both gates pass  → Logistic Regression stacker + weighted blend +
#                          rank average. Best by OOF AUC is submitted.
#   4b. Either gate fails → simple OOF-weighted blend (no stacking—not enough
#                           diversity to justify it).
#
# Usage:
#   python main.py --blend-te --output v11.csv
# =============================================================================

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder
from catboost import CatBoostClassifier
import lightgbm as lgb

from data_utils import load_data, get_feature_target_split, get_cat_features, prepare_test_data

SUBMISSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submissions")
RESULTS_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

N_FOLDS = 5
SEEDS   = [0, 7, 42, 123, 456, 999, 1337, 2024, 31337, 77777]

# v10 CatBoost params (best LB so far: 0.78218)
_CB_PARAMS = {
    "iterations":    3000,
    "learning_rate": 0.01,
    "depth":         8,
}

# LightGBM params — v8/v10 defaults (outperformed Optuna-tuned params on LB)
_LGBM_PARAMS = {
    "n_estimators":     1000,
    "learning_rate":    0.05,
    "num_leaves":       127,
    "min_child_samples": 20,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "device":           "gpu",
    "n_jobs":           -1,
    "verbose":          -1,
}

# Gate thresholds
_GATE1_AUC_THRESHOLD  = 0.763   # TE-LGBM must beat ordinal baseline (~0.762)
_GATE2_CORR_THRESHOLD = 0.990   # must be diverse enough from CatBoost


def _rank_avg(preds_list):
    """Return the mean of percentile ranks across a list of prediction arrays."""
    ranks = [pd.Series(p).rank(pct=True).values for p in preds_list]
    return np.mean(ranks, axis=0)


def _fmt(seconds):
    """Format seconds as Xh Ym Zs."""
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def run_blend_te(output_name="submission_blend_te.csv"):
    run_start = time.time()
    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    train_df, test_df, _ = load_data()
    X_train_full, y_train_full, _ = get_feature_target_split(train_df)
    X_test, test_ids = prepare_test_data(test_df)
    cat_features = get_cat_features(X_train_full)

    for col in cat_features:
        X_train_full[col] = X_train_full[col].fillna("Missing")
        X_test[col]       = X_test[col].fillna("Missing")

    total_models = N_FOLDS * len(SEEDS)
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR,     exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: CatBoost ensemble — load cache or train fresh
    # ------------------------------------------------------------------
    cb_oof_path  = os.path.join(RESULTS_DIR, "cb_v10_oof.npy")
    cb_test_path = os.path.join(RESULTS_DIR, "cb_v10_test.npy")

    if os.path.exists(cb_oof_path) and os.path.exists(cb_test_path):
        print("Loading cached CatBoost v10 OOF/test predictions from results/ ...")
        cb_oof  = np.load(cb_oof_path)
        cb_test = np.load(cb_test_path)
        cb_auc  = roc_auc_score(y_train_full, cb_oof)
        print(f"CatBoost OOF AUC (cached): {cb_auc:.4f}")
    else:
        print(f"\nTraining CatBoost {N_FOLDS}-fold x {len(SEEDS)}-seed ensemble "
              f"({total_models} models) — params: lr={_CB_PARAMS['learning_rate']}, "
              f"iter={_CB_PARAMS['iterations']}, depth={_CB_PARAMS['depth']}")
        cb_oof  = np.zeros(len(X_train_full))
        cb_test = np.zeros(len(X_test))

        cb_phase_start = time.time()
        cb_fold_times  = []

        for s_idx, seed in enumerate(SEEDS, 1):
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            for fold, (train_idx, val_idx) in enumerate(
                    skf.split(X_train_full, y_train_full), 1):
                fold_start = time.time()
                X_tr  = X_train_full.iloc[train_idx].copy()
                y_tr  = y_train_full.iloc[train_idx]
                X_val = X_train_full.iloc[val_idx].copy()

                model = CatBoostClassifier(
                    **_CB_PARAMS,
                    eval_metric="Logloss",
                    cat_features=cat_features,
                    random_seed=seed,
                    task_type="GPU",
                    verbose=0,
                )
                model.fit(X_tr, y_tr)
                cb_oof[val_idx] += model.predict_proba(X_val)[:, 1] / len(SEEDS)
                cb_test         += model.predict_proba(X_test)[:, 1] / total_models

                fold_elapsed = time.time() - fold_start
                cb_fold_times.append(fold_elapsed)
                models_done      = (s_idx - 1) * N_FOLDS + fold
                models_remaining = total_models - models_done
                avg_fold_time    = sum(cb_fold_times) / len(cb_fold_times)
                eta              = avg_fold_time * models_remaining
                elapsed_total    = time.time() - cb_phase_start
                print(f"  CB Seed {s_idx}/{len(SEEDS)}, Fold {fold}/{N_FOLDS} done "
                      f"| fold: {_fmt(fold_elapsed)} "
                      f"| elapsed: {_fmt(elapsed_total)} "
                      f"| ETA: {_fmt(eta)} "
                      f"| {models_done}/{total_models} models")

        cb_phase_time = time.time() - cb_phase_start
        np.save(cb_oof_path,  cb_oof)
        np.save(cb_test_path, cb_test)
        cb_auc = roc_auc_score(y_train_full, cb_oof)
        print(f"CatBoost OOF AUC: {cb_auc:.4f}  "
              f"(phase took {_fmt(cb_phase_time)}, saved to results/ for reuse)")

    # ------------------------------------------------------------------
    # Step 2: Target-Encoded LightGBM ensemble
    # ------------------------------------------------------------------
    print(f"\nTraining Target-Encoded LightGBM {N_FOLDS}-fold x {len(SEEDS)}-seed "
          f"({total_models} models) ...")
    print("  TargetEncoder(cv=5, smooth='auto') — leakage-safe OOF encoding")

    # Fit TargetEncoder on FULL training data for test-set transformation.
    # This is the best available encoding for unseen data.
    te_full = TargetEncoder(cv=5, smooth="auto", random_state=42)
    te_full.fit(X_train_full[cat_features], y_train_full)
    X_test_te = X_test.copy()
    X_test_te[cat_features] = te_full.transform(X_test[cat_features])

    lgbm_oof  = np.zeros(len(X_train_full))
    lgbm_test = np.zeros(len(X_test))

    lgbm_phase_start = time.time()
    lgbm_fold_times  = []

    for s_idx, seed in enumerate(SEEDS, 1):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(
                skf.split(X_train_full, y_train_full), 1):
            fold_start = time.time()
            X_tr_raw  = X_train_full.iloc[train_idx].copy()
            y_tr      = y_train_full.iloc[train_idx]
            X_val_raw = X_train_full.iloc[val_idx].copy()

            # OOF-safe encoding: fit TE on THIS fold's train only.
            # cv=5 inside fit_transform prevents leakage within the fold too.
            te = TargetEncoder(cv=5, smooth="auto", random_state=seed)

            X_tr_enc  = X_tr_raw.copy()
            X_val_enc = X_val_raw.copy()

            X_tr_enc[cat_features]  = te.fit_transform(
                X_tr_raw[cat_features], y_tr)
            X_val_enc[cat_features] = te.transform(X_val_raw[cat_features])

            model = lgb.LGBMClassifier(**_LGBM_PARAMS, random_state=seed)
            # No categorical_feature — features are now numeric target-encoded floats
            model.fit(X_tr_enc, y_tr)

            lgbm_oof[val_idx] += model.predict_proba(X_val_enc)[:, 1] / len(SEEDS)
            lgbm_test         += model.predict_proba(X_test_te)[:, 1] / total_models

            fold_elapsed = time.time() - fold_start
            lgbm_fold_times.append(fold_elapsed)
            models_done      = (s_idx - 1) * N_FOLDS + fold
            models_remaining = total_models - models_done
            avg_fold_time    = sum(lgbm_fold_times) / len(lgbm_fold_times)
            eta              = avg_fold_time * models_remaining
            elapsed_total    = time.time() - lgbm_phase_start
            print(f"  TE-LGBM Seed {s_idx}/{len(SEEDS)}, Fold {fold}/{N_FOLDS} done "
                  f"| fold: {_fmt(fold_elapsed)} "
                  f"| elapsed: {_fmt(elapsed_total)} "
                  f"| ETA: {_fmt(eta)} "
                  f"| {models_done}/{total_models} models")

    lgbm_phase_time = time.time() - lgbm_phase_start
    # Save TE-LGBM preds for reference
    np.save(os.path.join(RESULTS_DIR, "lgbm_te_oof.npy"),  lgbm_oof)
    np.save(os.path.join(RESULTS_DIR, "lgbm_te_test.npy"), lgbm_test)

    lgbm_auc = roc_auc_score(y_train_full, lgbm_oof)
    corr     = float(np.corrcoef(cb_oof, lgbm_oof)[0, 1])

    print(f"TE-LGBM phase took {_fmt(lgbm_phase_time)}")
    print(f"\n{'='*60}")
    print(f"TE-LightGBM OOF AUC  : {lgbm_auc:.4f}  (ordinal baseline ~0.762)")
    print(f"Correlation with CB  : {corr:.4f}")
    print(f"Gate 1  AUC > {_GATE1_AUC_THRESHOLD} : "
          f"{'PASS' if lgbm_auc > _GATE1_AUC_THRESHOLD else 'FAIL'}")
    print(f"Gate 2  corr < {_GATE2_CORR_THRESHOLD}: "
          f"{'PASS' if corr < _GATE2_CORR_THRESHOLD else 'FAIL'}")
    print(f"{'='*60}")

    both_gates_pass = (lgbm_auc > _GATE1_AUC_THRESHOLD and
                       corr      < _GATE2_CORR_THRESHOLD)

    # ------------------------------------------------------------------
    # Step 3: Build candidate submissions
    # ------------------------------------------------------------------
    # --- Weighted blend (OOF grid search) — always built ---
    candidates = {}
    best_w, best_blend_auc = 0.9, 0.0
    for w in np.arange(0.0, 1.01, 0.05):
        auc = roc_auc_score(y_train_full, w * cb_oof + (1 - w) * lgbm_oof)
        if auc > best_blend_auc:
            best_blend_auc, best_w = auc, w

    candidates["weighted_blend"] = {
        "oof_auc":    best_blend_auc,
        "test_preds": best_w * cb_test + (1 - best_w) * lgbm_test,
    }
    print(f"\nWeighted blend OOF AUC : {best_blend_auc:.4f}"
          f"  (CB {best_w:.2f} + LGBM {1-best_w:.2f})")

    # --- Rank averaging — always built ---
    rank_oof  = _rank_avg([cb_oof,  lgbm_oof])
    rank_test = _rank_avg([cb_test, lgbm_test])
    rank_auc  = roc_auc_score(y_train_full, rank_oof)
    candidates["rank_avg"] = {"oof_auc": rank_auc, "test_preds": rank_test}
    print(f"Rank-averaged OOF AUC  : {rank_auc:.4f}")

    # --- Logistic regression stacker — only if gates pass ---
    if both_gates_pass:
        print("\nBoth gates PASSED — building logistic regression stacker ...")
        meta_X_oof  = np.column_stack([cb_oof,  lgbm_oof])
        meta_X_test = np.column_stack([cb_test, lgbm_test])

        meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        meta.fit(meta_X_oof, y_train_full)

        stacked_oof   = meta.predict_proba(meta_X_oof)[:, 1]
        stacked_test  = meta.predict_proba(meta_X_test)[:, 1]
        stacked_auc   = roc_auc_score(y_train_full, stacked_oof)

        candidates["stacked"] = {"oof_auc": stacked_auc, "test_preds": stacked_test}
        print(f"Stacked OOF AUC        : {stacked_auc:.4f}")
        print(f"LR meta weights — CB: {meta.coef_[0][0]:.4f}, "
              f"LGBM: {meta.coef_[0][1]:.4f}")
    else:
        print("\nGate(s) FAILED — skipping stacker. "
              "Submitting best of weighted blend / rank avg to learn direction.")

    best_method = max(candidates, key=lambda k: candidates[k]["oof_auc"])
    best_auc    = candidates[best_method]["oof_auc"]
    test_preds  = candidates[best_method]["test_preds"]

    # ------------------------------------------------------------------
    # Step 4: Save submission (best_method/best_auc/test_preds set above)
    # ------------------------------------------------------------------
    print(f"\nBest method: {best_method}  (OOF AUC: {best_auc:.4f})")

    submission_df = pd.DataFrame({"ID": test_ids.values, "NO_SHOW_FLG": test_preds})
    out_path      = os.path.join(SUBMISSIONS_DIR, output_name)
    submission_df.to_csv(out_path, index=False)
    print(f"Submission saved → submissions/{output_name}  ({len(submission_df):,} rows)")
    print(f"Predicted no-show rate: {test_preds.mean():.4f}")
    print(f"Total runtime: {_fmt(time.time() - run_start)}")
