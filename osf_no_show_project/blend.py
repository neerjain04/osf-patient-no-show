# =============================================================================
# blend.py
# -----------------------------------------------------------------------------
# Blends a CatBoost ensemble with a LightGBM ensemble.
#
# Both models train with 10-seed × 5-fold CV (50 models each = 100 total).
# OOF predictions are used to find the optimal blend weight automatically.
# The blended test predictions are saved as the submission file.
#
# Usage:
#   python main.py --blend --output v8.csv
# =============================================================================

import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import lightgbm as lgb

from data_utils import load_data, get_feature_target_split, get_cat_features, prepare_test_data, add_interaction_features, add_frequency_features

SUBMISSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submissions")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

N_FOLDS = 5
SEEDS = [
    0, 7, 42, 123, 456, 999, 1337, 2024, 31337, 77777,   # original 10 (v10)
    100, 200, 314, 2025, 8888, 9999, 12345, 54321, 99999, 111111,  # 10 new seeds
]

# CatBoost params — lower LR + more iterations for better convergence (v10)
_CB_PARAMS = {
    "iterations": 3000,
    "learning_rate": 0.01,
    "depth": 8,
}

# LightGBM fallback defaults (used if tune_lgbm has not been run yet)
_LGBM_DEFAULTS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}


def _fmt(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _load_lgbm_params():
    """Use default LGBM params. Tuned params (v9) underperformed defaults (v8) on this dataset."""
    print("Using default LGBM params (tuned params hurt LB on this categorical-heavy dataset).")
    return _LGBM_DEFAULTS.copy()


def _encode_for_lgbm(X_train, X_test, cat_features):
    """OrdinalEncode categorical columns to non-negative integers for LightGBM."""
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[cat_features] = enc.fit_transform(X_train[cat_features]).astype(int)
    X_test[cat_features] = enc.transform(X_test[cat_features]).astype(int)
    return X_train, X_test


def run_blend_freq(output_name="submission_blend_freq.csv"):
    """CB+LGBM blend (10 seeds × 5 folds) with frequency-encoded features added.
    For every categorical column, adds a numeric <COL>_FREQ column = proportion
    of training rows with that value. Trains from scratch, saves freq_cb/lgbm npy."""
    run_start = time.time()
    train_df, test_df, _ = load_data()
    X_train_full, y_train_full, _ = get_feature_target_split(train_df)
    X_test, test_ids = prepare_test_data(test_df)

    X_train_full, X_test = add_frequency_features(X_train_full, X_test)
    cat_features = get_cat_features(X_train_full)  # freq cols are float, excluded automatically
    n_freq = sum(1 for c in X_train_full.columns if c.endswith("_FREQ"))
    print(f"Frequency encoding: added {n_freq} numeric FREQ columns → {X_train_full.shape[1]} total features")

    for col in cat_features:
        X_train_full[col] = X_train_full[col].fillna("Missing")
        X_test[col] = X_test[col].fillna("Missing")

    X_train_lgbm, X_test_lgbm = _encode_for_lgbm(X_train_full, X_test, cat_features)

    active_seeds = SEEDS[:10]
    n_seeds = len(active_seeds)
    total_models = N_FOLDS * n_seeds
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    lgbm_params = _load_lgbm_params()
    lgbm_params.update({"device": "gpu", "n_jobs": -1, "verbose": -1})

    # --- CatBoost ---
    print(f"\nTraining CatBoost [+FREQ] {N_FOLDS}-fold × {n_seeds}-seed ({total_models} models)...")
    cb_oof  = np.zeros(len(X_train_full))
    cb_test = np.zeros(len(X_test))
    cb_times = []
    cb_start = time.time()
    for s_idx, seed in enumerate(active_seeds, 1):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), 1):
            fold_start = time.time()
            X_tr  = X_train_full.iloc[train_idx].copy()
            y_tr  = y_train_full.iloc[train_idx]
            X_val = X_train_full.iloc[val_idx].copy()
            model = CatBoostClassifier(
                **_CB_PARAMS, eval_metric="Logloss",
                cat_features=cat_features, random_seed=seed,
                task_type="GPU", verbose=0,
            )
            model.fit(X_tr, y_tr)
            cb_oof[val_idx] += model.predict_proba(X_val)[:, 1] / n_seeds
            cb_test          += model.predict_proba(X_test)[:, 1] / total_models
            fold_elapsed = time.time() - fold_start
            cb_times.append(fold_elapsed)
            done = (s_idx - 1) * N_FOLDS + fold
            eta  = (sum(cb_times) / len(cb_times)) * (total_models - done)
            print(f"  CB Seed {s_idx}/{n_seeds}, Fold {fold}/{N_FOLDS} done"
                  f" | fold: {_fmt(fold_elapsed)} | ETA: {_fmt(eta)} | {done}/{total_models}")
    cb_auc = roc_auc_score(y_train_full, cb_oof)
    print(f"CB [+FREQ] OOF AUC: {cb_auc:.6f}  (took {_fmt(time.time() - cb_start)})")
    np.save(os.path.join(RESULTS_DIR, "freq_cb_oof.npy"),  cb_oof)
    np.save(os.path.join(RESULTS_DIR, "freq_cb_test.npy"), cb_test)

    # --- LightGBM ---
    print(f"\nTraining LightGBM [+FREQ] {N_FOLDS}-fold × {n_seeds}-seed ({total_models} models)...")
    lgbm_oof  = np.zeros(len(X_train_lgbm))
    lgbm_test = np.zeros(len(X_test_lgbm))
    lgbm_times = []
    lgbm_start = time.time()
    for s_idx, seed in enumerate(active_seeds, 1):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_lgbm, y_train_full), 1):
            fold_start = time.time()
            X_tr  = X_train_lgbm.iloc[train_idx]
            y_tr  = y_train_full.iloc[train_idx]
            X_val = X_train_lgbm.iloc[val_idx]
            model = lgb.LGBMClassifier(**lgbm_params, random_state=seed)
            model.fit(X_tr, y_tr, categorical_feature=cat_features)
            lgbm_oof[val_idx] += model.predict_proba(X_val)[:, 1] / n_seeds
            lgbm_test          += model.predict_proba(X_test_lgbm)[:, 1] / total_models
            fold_elapsed = time.time() - fold_start
            lgbm_times.append(fold_elapsed)
            done = (s_idx - 1) * N_FOLDS + fold
            eta  = (sum(lgbm_times) / len(lgbm_times)) * (total_models - done)
            print(f"  LGBM Seed {s_idx}/{n_seeds}, Fold {fold}/{N_FOLDS} done"
                  f" | fold: {_fmt(fold_elapsed)} | ETA: {_fmt(eta)} | {done}/{total_models}")
    lgbm_auc = roc_auc_score(y_train_full, lgbm_oof)
    print(f"LGBM [+FREQ] OOF AUC: {lgbm_auc:.6f}  (took {_fmt(time.time() - lgbm_start)})")
    np.save(os.path.join(RESULTS_DIR, "freq_lgbm_oof.npy"),  lgbm_oof)
    np.save(os.path.join(RESULTS_DIR, "freq_lgbm_test.npy"), lgbm_test)

    # --- Optimal blend ---
    best_w, best_auc = 0.5, 0.0
    for w in np.arange(0.0, 1.01, 0.05):
        blended = w * cb_oof + (1 - w) * lgbm_oof
        auc = roc_auc_score(y_train_full, blended)
        if auc > best_auc:
            best_auc, best_w = auc, w
    print(f"\nOptimal blend [+FREQ]: CB {best_w:.2f} + LGBM {1-best_w:.2f}")
    print(f"Blended OOF AUC: {best_auc:.6f}")
    test_preds = best_w * cb_test + (1 - best_w) * lgbm_test
    submission_df = pd.DataFrame({"ID": test_ids.values, "NO_SHOW_FLG": test_preds})
    submission_df.to_csv(os.path.join(SUBMISSIONS_DIR, output_name), index=False)
    print(f"Submission saved to submissions/{output_name}  ({len(submission_df):,} rows)")
    print(f"Total runtime: {_fmt(time.time() - run_start)}")


def run_blend_fe(output_name="submission_blend_fe.csv"):
    """Same as run_blend (10 seeds × 5 folds, CB+LGBM) but with interaction
    features added. Trains from scratch — no cache (different feature space).
    Saves cb_fe_oof/test.npy + lgbm_fe_oof/test.npy."""
    run_start = time.time()
    train_df, test_df, _ = load_data()
    X_train_full, y_train_full, _ = get_feature_target_split(train_df)
    X_test, test_ids = prepare_test_data(test_df)

    # Add 7 interaction features (concatenated categoricals)
    X_train_full = add_interaction_features(X_train_full)
    X_test = add_interaction_features(X_test)
    cat_features = get_cat_features(X_train_full)  # picks up new cols automatically

    n_orig = X_train_full.shape[1] - 7  # 20 original + 7 new
    print(f"Feature engineering: {n_orig} original + 7 interaction = {X_train_full.shape[1]} total features")

    for col in cat_features:
        X_train_full[col] = X_train_full[col].fillna("Missing")
        X_test[col] = X_test[col].fillna("Missing")

    X_train_lgbm, X_test_lgbm = _encode_for_lgbm(X_train_full, X_test, cat_features)

    active_seeds = SEEDS[:10]
    n_seeds = len(active_seeds)
    total_models = N_FOLDS * n_seeds
    tag_cb = "cb_fe"
    tag_lgbm = "lgbm_fe"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    lgbm_params = _load_lgbm_params()
    lgbm_params.update({"device": "gpu", "n_jobs": -1, "verbose": -1})

    # --- CatBoost ---
    print(f"\nTraining CatBoost [+FE] {N_FOLDS}-fold × {n_seeds}-seed ({total_models} models)...")
    cb_oof  = np.zeros(len(X_train_full))
    cb_test = np.zeros(len(X_test))
    cb_times = []
    cb_start = time.time()

    for s_idx, seed in enumerate(active_seeds, 1):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), 1):
            fold_start = time.time()
            X_tr  = X_train_full.iloc[train_idx].copy()
            y_tr  = y_train_full.iloc[train_idx]
            X_val = X_train_full.iloc[val_idx].copy()
            model = CatBoostClassifier(
                **_CB_PARAMS, eval_metric="Logloss",
                cat_features=cat_features, random_seed=seed,
                task_type="GPU", verbose=0,
            )
            model.fit(X_tr, y_tr)
            cb_oof[val_idx] += model.predict_proba(X_val)[:, 1] / n_seeds
            cb_test          += model.predict_proba(X_test)[:, 1] / total_models
            fold_elapsed = time.time() - fold_start
            cb_times.append(fold_elapsed)
            done = (s_idx - 1) * N_FOLDS + fold
            eta  = (sum(cb_times) / len(cb_times)) * (total_models - done)
            print(f"  CB Seed {s_idx}/{n_seeds}, Fold {fold}/{N_FOLDS} done"
                  f" | fold: {_fmt(fold_elapsed)} | ETA: {_fmt(eta)} | {done}/{total_models}")

    cb_auc = roc_auc_score(y_train_full, cb_oof)
    print(f"CB [+FE] OOF AUC: {cb_auc:.6f}  (took {_fmt(time.time() - cb_start)})")
    np.save(os.path.join(RESULTS_DIR, f"{tag_cb}_oof.npy"),  cb_oof)
    np.save(os.path.join(RESULTS_DIR, f"{tag_cb}_test.npy"), cb_test)
    print(f"  Saved {tag_cb}_oof.npy + {tag_cb}_test.npy")

    # --- LightGBM ---
    print(f"\nTraining LightGBM [+FE] {N_FOLDS}-fold × {n_seeds}-seed ({total_models} models)...")
    lgbm_oof  = np.zeros(len(X_train_lgbm))
    lgbm_test = np.zeros(len(X_test_lgbm))
    lgbm_times = []
    lgbm_start = time.time()

    for s_idx, seed in enumerate(active_seeds, 1):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_lgbm, y_train_full), 1):
            fold_start = time.time()
            X_tr  = X_train_lgbm.iloc[train_idx]
            y_tr  = y_train_full.iloc[train_idx]
            X_val = X_train_lgbm.iloc[val_idx]
            model = lgb.LGBMClassifier(**lgbm_params, random_state=seed)
            model.fit(X_tr, y_tr, categorical_feature=cat_features)
            lgbm_oof[val_idx] += model.predict_proba(X_val)[:, 1] / n_seeds
            lgbm_test          += model.predict_proba(X_test_lgbm)[:, 1] / total_models
            fold_elapsed = time.time() - fold_start
            lgbm_times.append(fold_elapsed)
            done = (s_idx - 1) * N_FOLDS + fold
            eta  = (sum(lgbm_times) / len(lgbm_times)) * (total_models - done)
            print(f"  LGBM Seed {s_idx}/{n_seeds}, Fold {fold}/{N_FOLDS} done"
                  f" | fold: {_fmt(fold_elapsed)} | ETA: {_fmt(eta)} | {done}/{total_models}")

    lgbm_auc = roc_auc_score(y_train_full, lgbm_oof)
    print(f"LGBM [+FE] OOF AUC: {lgbm_auc:.6f}  (took {_fmt(time.time() - lgbm_start)})")
    np.save(os.path.join(RESULTS_DIR, f"{tag_lgbm}_oof.npy"),  lgbm_oof)
    np.save(os.path.join(RESULTS_DIR, f"{tag_lgbm}_test.npy"), lgbm_test)
    print(f"  Saved {tag_lgbm}_oof.npy + {tag_lgbm}_test.npy")

    # --- Optimal blend ---
    best_w, best_auc = 0.5, 0.0
    for w in np.arange(0.0, 1.01, 0.05):
        blended = w * cb_oof + (1 - w) * lgbm_oof
        auc = roc_auc_score(y_train_full, blended)
        if auc > best_auc:
            best_auc, best_w = auc, w
    print(f"\nOptimal blend [+FE]: CB {best_w:.2f} + LGBM {1-best_w:.2f}")
    print(f"Blended OOF AUC: {best_auc:.6f}")

    test_preds = best_w * cb_test + (1 - best_w) * lgbm_test
    submission_df = pd.DataFrame({"ID": test_ids.values, "NO_SHOW_FLG": test_preds})
    out_path = os.path.join(SUBMISSIONS_DIR, output_name)
    submission_df.to_csv(out_path, index=False)
    print(f"Submission saved to submissions/{output_name}  ({len(submission_df):,} rows)")
    print(f"Total runtime: {_fmt(time.time() - run_start)}")


def run_blend(output_name="submission_blend.csv", feature_subset=None):
    run_start = time.time()
    train_df, test_df, _ = load_data()
    X_train_full, y_train_full, _ = get_feature_target_split(train_df)
    X_test, test_ids = prepare_test_data(test_df)
    cat_features = get_cat_features(X_train_full)

    # Optionally restrict to a subset of features
    if feature_subset is not None:
        X_train_full = X_train_full[feature_subset]
        X_test = X_test[feature_subset]
        cat_features = [c for c in cat_features if c in feature_subset]
        print(f"Using top-{len(feature_subset)} features: {feature_subset}")

    # Fill NaN in categorical columns (required by both models)
    for col in cat_features:
        X_train_full[col] = X_train_full[col].fillna("Missing")
        X_test[col] = X_test[col].fillna("Missing")

    # LightGBM needs integer-encoded categoricals
    X_train_lgbm, X_test_lgbm = _encode_for_lgbm(X_train_full, X_test, cat_features)

    total_models = N_FOLDS * len(SEEDS)
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load tuned LGBM params (or defaults if tuning hasn't been run)
    lgbm_params = _load_lgbm_params()
    # Always inject GPU + silent settings on top of whatever params were loaded
    lgbm_params.update({"device": "gpu", "n_jobs": -1, "verbose": -1})

    # When running a feature-subset experiment, use v10's 10 seeds exactly
    # and skip the cache (different feature space = different predictions)
    if feature_subset is not None:
        active_seeds = SEEDS[:10]
        skip_cache = True
        exp_tag = f"top{len(feature_subset)}"
    else:
        active_seeds = SEEDS
        skip_cache = False
        exp_tag = None

    n_seeds = len(active_seeds)
    total_models = N_FOLDS * n_seeds
    tag = exp_tag if exp_tag else f"cb_{n_seeds}s"

    # -------------------------------------------------------------------------
    # CatBoost: N-seed × 5-fold ensemble (reuses cached 10-seed preds if present)
    # -------------------------------------------------------------------------
    SEEDS_CACHED = active_seeds[:10]   # original v10 seeds — may already be saved
    SEEDS_NEW    = active_seeds[10:]   # new seeds to train now

    cb_oof_cached_path  = os.path.join(RESULTS_DIR, "cb_v10_oof.npy")
    cb_test_cached_path = os.path.join(RESULTS_DIR, "cb_v10_test.npy")
    have_cache = (not skip_cache
                  and os.path.exists(cb_oof_cached_path)
                  and os.path.exists(cb_test_cached_path))

    cb_phase_start = time.time()

    if have_cache:
        print(f"\nFound cached CB preds for {len(SEEDS_CACHED)} seeds — training only "
              f"{len(SEEDS_NEW)} new seeds ({len(SEEDS_NEW) * N_FOLDS} models)...")
        cb_oof_cached  = np.load(cb_oof_cached_path)
        cb_test_cached = np.load(cb_test_cached_path)
        print(f"  Cached CB OOF AUC: {roc_auc_score(y_train_full, cb_oof_cached):.4f}")

        seeds_to_train = SEEDS_NEW
        n_seeds_to_train = len(SEEDS_NEW)
        new_total = N_FOLDS * n_seeds_to_train
    else:
        print(f"\nTraining CatBoost {N_FOLDS}-fold × {n_seeds}-seed "
              f"ensemble ({total_models} models)...")
        seeds_to_train = active_seeds
        n_seeds_to_train = n_seeds
        new_total = total_models

    cb_oof_new  = np.zeros(len(X_train_full))
    cb_test_new = np.zeros(len(X_test))
    cb_fold_times = []

    for s_idx, seed in enumerate(seeds_to_train, 1):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), 1):
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
            cb_oof_new[val_idx] += model.predict_proba(X_val)[:, 1] / n_seeds_to_train
            cb_test_new         += model.predict_proba(X_test)[:, 1] / new_total

            fold_elapsed = time.time() - fold_start
            cb_fold_times.append(fold_elapsed)
            models_done = (s_idx - 1) * N_FOLDS + fold
            eta = (sum(cb_fold_times) / len(cb_fold_times)) * (new_total - models_done)
            elapsed = time.time() - cb_phase_start
            print(f"  CB Seed {s_idx}/{n_seeds_to_train}, Fold {fold}/{N_FOLDS} done"
                  f" | fold: {_fmt(fold_elapsed)} | elapsed: {_fmt(elapsed)}"
                  f" | ETA: {_fmt(eta)} | {models_done}/{new_total} models")

    # Combine cached + new (equal weight per seed → simple average)
    if have_cache:
        cb_oof  = 0.5 * cb_oof_cached  + 0.5 * cb_oof_new
        cb_test = 0.5 * cb_test_cached + 0.5 * cb_test_new
    else:
        cb_oof  = cb_oof_new
        cb_test = cb_test_new

    cb_auc = roc_auc_score(y_train_full, cb_oof)
    print(f"CatBoost OOF AUC ({n_seeds} seeds): {cb_auc:.4f}  "
          f"(CB phase took {_fmt(time.time() - cb_phase_start)})")
    np.save(os.path.join(RESULTS_DIR, f"{tag}_oof.npy"),  cb_oof)
    np.save(os.path.join(RESULTS_DIR, f"{tag}_test.npy"), cb_test)
    print(f"  Saved {tag}_oof.npy + {tag}_test.npy")

    # -------------------------------------------------------------------------
    # LightGBM: N-seed × 5-fold ensemble
    # -------------------------------------------------------------------------
    print(f"\nTraining LightGBM {N_FOLDS}-fold × {n_seeds}-seed ensemble ({total_models} models)...")
    lgbm_oof = np.zeros(len(X_train_lgbm))
    lgbm_test = np.zeros(len(X_test_lgbm))
    lgbm_fold_times = []
    lgbm_phase_start = time.time()

    for s_idx, seed in enumerate(active_seeds, 1):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_lgbm, y_train_full), 1):
            fold_start = time.time()
            X_tr = X_train_lgbm.iloc[train_idx]
            y_tr = y_train_full.iloc[train_idx]
            X_val = X_train_lgbm.iloc[val_idx]

            model = lgb.LGBMClassifier(**lgbm_params, random_state=seed)
            model.fit(
                X_tr, y_tr,
                categorical_feature=cat_features,
            )
            lgbm_oof[val_idx] += model.predict_proba(X_val)[:, 1] / n_seeds
            lgbm_test += model.predict_proba(X_test_lgbm)[:, 1] / total_models

            fold_elapsed = time.time() - fold_start
            lgbm_fold_times.append(fold_elapsed)
            models_done = (s_idx - 1) * N_FOLDS + fold
            eta = (sum(lgbm_fold_times) / len(lgbm_fold_times)) * (total_models - models_done)
            elapsed = time.time() - lgbm_phase_start
            print(f"  LGBM Seed {s_idx}/{n_seeds}, Fold {fold}/{N_FOLDS} done"
                  f" | fold: {_fmt(fold_elapsed)} | elapsed: {_fmt(elapsed)}"
                  f" | ETA: {_fmt(eta)} | {models_done}/{total_models} models")

    lgbm_auc = roc_auc_score(y_train_full, lgbm_oof)
    print(f"LightGBM OOF AUC: {lgbm_auc:.4f}  (took {_fmt(time.time() - lgbm_phase_start)})")
    np.save(os.path.join(RESULTS_DIR, f"lgbm_{n_seeds}s_oof.npy"),  lgbm_oof)
    np.save(os.path.join(RESULTS_DIR, f"lgbm_{n_seeds}s_test.npy"), lgbm_test)
    print(f"  Saved lgbm_{n_seeds}s_oof.npy + lgbm_{n_seeds}s_test.npy")

    # -------------------------------------------------------------------------
    # Find optimal blend weight using OOF predictions
    # -------------------------------------------------------------------------
    best_w, best_auc = 0.5, 0.0
    for w in np.arange(0.0, 1.01, 0.05):
        blended = w * cb_oof + (1 - w) * lgbm_oof
        auc = roc_auc_score(y_train_full, blended)
        if auc > best_auc:  
            best_auc = auc
            best_w = w

    print(f"\nOptimal blend: CatBoost {best_w:.2f} + LightGBM {1 - best_w:.2f}")
    print(f"Blended OOF AUC: {best_auc:.4f}")

    # -------------------------------------------------------------------------
    # Build and save submission
    # -------------------------------------------------------------------------
    test_preds = best_w * cb_test + (1 - best_w) * lgbm_test
    submission_df = pd.DataFrame({"ID": test_ids.values, "NO_SHOW_FLG": test_preds})
    out_path = os.path.join(SUBMISSIONS_DIR, output_name)
    submission_df.to_csv(out_path, index=False)
    print(f"\nSubmission saved to submissions/{output_name}  ({len(submission_df):,} rows)")
    print(f"Predicted no-show rate: {test_preds.mean():.4f}")
    print(f"Total runtime: {_fmt(time.time() - run_start)}")


# =============================================================================
# CB Variant: train a single CatBoost model with different hyperparams
# =============================================================================

def run_cb_variant(depth=9, lr=0.02, iterations=2000, tag="cb_d9"):
    """Train a CatBoost variant (default: depth=9, lr=0.02, iter=2000).
    Uses the same 10 seeds as v10. Saves OOF + test preds as {tag}_oof/test.npy."""
    run_start = time.time()
    train_df, test_df, _ = load_data()
    X_train_full, y_train_full, _ = get_feature_target_split(train_df)
    X_test, test_ids = prepare_test_data(test_df)
    cat_features = get_cat_features(X_train_full)

    for col in cat_features:
        X_train_full[col] = X_train_full[col].fillna("Missing")
        X_test[col] = X_test[col].fillna("Missing")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    active_seeds = SEEDS[:10]       # same 10 seeds as v10 for apples-to-apples comparison
    n_seeds = len(active_seeds)
    total_models = N_FOLDS * n_seeds
    cb_params = {"iterations": iterations, "learning_rate": lr, "depth": depth}

    print(f"\nTraining CatBoost variant [{tag}]: depth={depth}, lr={lr}, iter={iterations}")
    print(f"  {n_seeds} seeds × {N_FOLDS} folds = {total_models} models")

    cb_oof  = np.zeros(len(X_train_full))
    cb_test = np.zeros(len(X_test))
    fold_times = []
    phase_start = time.time()

    for s_idx, seed in enumerate(active_seeds, 1):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), 1):
            fold_start = time.time()
            X_tr  = X_train_full.iloc[train_idx].copy()
            y_tr  = y_train_full.iloc[train_idx]
            X_val = X_train_full.iloc[val_idx].copy()

            model = CatBoostClassifier(
                **cb_params,
                eval_metric="Logloss",
                cat_features=cat_features,
                random_seed=seed,
                task_type="GPU",
                verbose=0,
            )
            model.fit(X_tr, y_tr)
            cb_oof[val_idx] += model.predict_proba(X_val)[:, 1] / n_seeds
            cb_test          += model.predict_proba(X_test)[:, 1] / total_models

            fold_elapsed = time.time() - fold_start
            fold_times.append(fold_elapsed)
            models_done = (s_idx - 1) * N_FOLDS + fold
            eta = (sum(fold_times) / len(fold_times)) * (total_models - models_done)
            elapsed = time.time() - phase_start
            print(f"  Seed {s_idx}/{n_seeds}, Fold {fold}/{N_FOLDS} done"
                  f" | fold: {_fmt(fold_elapsed)} | elapsed: {_fmt(elapsed)}"
                  f" | ETA: {_fmt(eta)} | {models_done}/{total_models} models")

    auc = roc_auc_score(y_train_full, cb_oof)
    print(f"\n[{tag}] OOF AUC: {auc:.6f}  (took {_fmt(time.time() - run_start)})")
    np.save(os.path.join(RESULTS_DIR, f"{tag}_oof.npy"),  cb_oof)
    np.save(os.path.join(RESULTS_DIR, f"{tag}_test.npy"), cb_test)
    print(f"  Saved {tag}_oof.npy + {tag}_test.npy")


# =============================================================================
# Blend variants: v10 + depth-9 CB — weight search + rank averaging
# =============================================================================

def run_blend_variants(output_name="submission_variants.csv"):
    """Load v10 + cb_d9 OOF preds, find optimal blend weight, try rank averaging.
    Saves the best-performing combination as a submission."""
    train_df, test_df, _ = load_data()
    _, y_train_full, _ = get_feature_target_split(train_df)
    _, test_ids = prepare_test_data(test_df)

    v10_oof_path  = os.path.join(RESULTS_DIR, "cb_v10_oof.npy")
    v10_test_path = os.path.join(RESULTS_DIR, "cb_v10_test.npy")
    d9_oof_path   = os.path.join(RESULTS_DIR, "cb_d9_oof.npy")
    d9_test_path  = os.path.join(RESULTS_DIR, "cb_d9_test.npy")

    for p in [v10_oof_path, v10_test_path, d9_oof_path, d9_test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}\nRun --cb-depth9 first.")

    v10_oof  = np.load(v10_oof_path)
    v10_test = np.load(v10_test_path)
    d9_oof   = np.load(d9_oof_path)
    d9_test  = np.load(d9_test_path)

    print(f"\nv10 OOF AUC  : {roc_auc_score(y_train_full, v10_oof):.6f}")
    print(f"d9  OOF AUC  : {roc_auc_score(y_train_full, d9_oof):.6f}")
    corr = np.corrcoef(v10_oof, d9_oof)[0, 1]
    print(f"Correlation  : {corr:.4f}  (lower = more diversity = better blend potential)")

    # --- Blend weight search ---
    print("\n--- Blend search (w * v10 + (1-w) * d9) ---")
    best_w, best_auc, best_test_preds = 0.5, 0.0, None
    for w in np.arange(0.0, 1.01, 0.05):
        blended_oof = w * v10_oof + (1 - w) * d9_oof
        auc = roc_auc_score(y_train_full, blended_oof)
        marker = " ← best" if auc > best_auc else ""
        print(f"  v10={w:.2f} + d9={1-w:.2f}  →  {auc:.6f}{marker}")
        if auc > best_auc:
            best_auc = auc
            best_w   = w
            best_test_preds = w * v10_test + (1 - w) * d9_test
    best_method = f"blend v10={best_w:.2f}+d9={1-best_w:.2f}"

    # --- Rank averaging ---
    print("\n--- Rank averaging ---")
    v10_rank_oof  = v10_oof.argsort().argsort().astype(float)
    d9_rank_oof   = d9_oof.argsort().argsort().astype(float)
    rank_oof_avg  = (v10_rank_oof + d9_rank_oof) / 2
    rank_auc = roc_auc_score(y_train_full, rank_oof_avg)
    marker = " ← best" if rank_auc > best_auc else ""
    print(f"  Rank avg OOF AUC: {rank_auc:.6f}{marker}")

    if rank_auc > best_auc:
        best_auc = rank_auc
        best_method = "rank_avg"
        v10_rank_test = v10_test.argsort().argsort().astype(float)
        d9_rank_test  = d9_test.argsort().argsort().astype(float)
        rank_test_avg = (v10_rank_test + d9_rank_test) / 2
        # Normalise to [0, 1] so submission values stay in probability range
        rank_test_avg = rank_test_avg / rank_test_avg.max()
        best_test_preds = rank_test_avg

    print(f"\nBest method : {best_method}")
    print(f"Best OOF AUC: {best_auc:.6f}")

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    submission_df = pd.DataFrame({"ID": test_ids.values, "NO_SHOW_FLG": best_test_preds})
    out_path = os.path.join(SUBMISSIONS_DIR, output_name)
    submission_df.to_csv(out_path, index=False)
    print(f"Submission saved to submissions/{output_name}  ({len(submission_df):,} rows)")


# =============================================================================
# Blend existing submission CSVs — rank averaging + fixed weight combos
# =============================================================================

def run_blend_subs(files, output_prefix="v15"):
    """Blend existing submission CSVs by rank averaging and fixed weight combos.
    No OOF available so weights cannot be optimized — generates all candidates.

    Args:
        files: list of (name, path) tuples, e.g.
               [("v10", "submissions/v10.csv"), ...]
        output_prefix: prefix for output filenames
    """
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

    dfs = {}
    for name, path in files:
        df = pd.read_csv(path).sort_values("ID").reset_index(drop=True)
        dfs[name] = df
        print(f"  Loaded {name}: {len(df):,} rows, mean={df['NO_SHOW_FLG'].mean():.4f}")

    base_ids = dfs[files[0][0]]["ID"].values
    for name, _ in files[1:]:
        assert np.array_equal(base_ids, dfs[name]["ID"].values), f"ID mismatch in {name}"

    names = [f[0] for f in files]
    preds = {n: dfs[n]["NO_SHOW_FLG"].values for n in names}

    print("\nPairwise correlations:")
    for i, a in enumerate(names):
        for b in names[i+1:]:
            corr = np.corrcoef(preds[a], preds[b])[0, 1]
            print(f"  {a} vs {b}: {corr:.4f}")

    saved = []

    # Rank averaging — parameter-free
    ranks = {n: preds[n].argsort().argsort().astype(float) for n in names}
    rank_avg = sum(ranks.values()) / len(names)
    rank_avg_norm = rank_avg / rank_avg.max()
    out = f"{output_prefix}_rank_avg.csv"
    pd.DataFrame({"ID": base_ids, "NO_SHOW_FLG": rank_avg_norm}).to_csv(
        os.path.join(SUBMISSIONS_DIR, out), index=False)
    print(f"\nSaved {out}  (rank averaging)")
    saved.append(out)

    # Fixed weight combos
    if len(names) == 3:
        weight_sets = [(0.6, 0.3, 0.1), (0.5, 0.3, 0.2), (0.7, 0.2, 0.1)]
    elif len(names) == 2:
        weight_sets = [(0.6, 0.4), (0.7, 0.3), (0.5, 0.5)]
    else:
        weight_sets = []

    for ws in weight_sets:
        blended = sum(w * preds[n] for w, n in zip(ws, names))
        label = "_".join(f"{int(round(w*10))}" for w in ws)
        out = f"{output_prefix}_w{label}.csv"
        pd.DataFrame({"ID": base_ids, "NO_SHOW_FLG": blended}).to_csv(
            os.path.join(SUBMISSIONS_DIR, out), index=False)
        combo = " + ".join(f"{w:.1f}×{n}" for w, n in zip(ws, names))
        print(f"Saved {out}  ({combo})")
        saved.append(out)

    print(f"\n{len(saved)} files saved. Submit all and compare LB.")


# =============================================================================
# Pseudo-labeling: use v10 high-confidence test preds as extra training data
# =============================================================================

def run_pseudo_label(
    pos_threshold=0.60,
    neg_threshold=0.02,
    output_name="submission_pseudo.csv",
    source_csv="v10.csv",
    save_tag="pl",
):
    """Pseudo-labeling using high-confidence test predictions from source_csv.

    Steps:
    1. Load blended test predictions from submissions/{source_csv}
    2. Select high-confidence rows:
       - predicted prob > pos_threshold  → label as 1 (no-show)
       - predicted prob < neg_threshold  → label as 0 (show)
    3. Append these pseudo-labeled test rows to the real training data
    4. Retrain v10 formula (CB+LGBM, 10 seeds × 5 folds) on expanded dataset
    5. OOF AUC evaluated on REAL training rows only (not pseudo-label rows)
       so the metric stays honest.

    save_tag: prefix for saved npy files, e.g. 'pl' → cb_pl_oof.npy, or
              'pl2' → cb_pl2_oof.npy for second-round pseudo-labeling.
    Saves cb_{save_tag}_oof/test.npy + lgbm_{save_tag}_oof/test.npy + submission.
    """
    run_start = time.time()
    train_df, test_df, _ = load_data()
    X_train_real, y_train_real, _ = get_feature_target_split(train_df)
    X_test, test_ids = prepare_test_data(test_df)
    cat_features = get_cat_features(X_train_real)

    for col in cat_features:
        X_train_real[col] = X_train_real[col].fillna("Missing")
        X_test[col] = X_test[col].fillna("Missing")

    # Load source test predictions
    src_path = os.path.join(SUBMISSIONS_DIR, source_csv)
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"{source_csv} not found at {src_path}")
    v10_preds = pd.read_csv(src_path).sort_values("ID").reset_index(drop=True)
    print(f"  Source predictions: {source_csv}")
    test_df_sorted = test_df.sort_values("ID").reset_index(drop=True)

    pos_mask = v10_preds["NO_SHOW_FLG"] >= pos_threshold
    neg_mask = v10_preds["NO_SHOW_FLG"] <= neg_threshold

    X_pseudo_pos = test_df_sorted.loc[pos_mask].drop(columns=["ID"])
    X_pseudo_neg = test_df_sorted.loc[neg_mask].drop(columns=["ID"])
    y_pseudo_pos = pd.Series(np.ones(pos_mask.sum(),  dtype=int), name="NO_SHOW_FLG")
    y_pseudo_neg = pd.Series(np.zeros(neg_mask.sum(), dtype=int), name="NO_SHOW_FLG")

    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    print(f"\nPseudo-label thresholds: pos >= {pos_threshold}, neg <= {neg_threshold}")
    print(f"  Selected {n_pos:,} positive pseudo-labels  ({n_pos/len(test_df)*100:.1f}% of test)")
    print(f"  Selected {n_neg:,} negative pseudo-labels  ({n_neg/len(test_df)*100:.1f}% of test)")
    print(f"  Real training rows:   {len(X_train_real):,}")

    # Combine real + pseudo
    for col in cat_features:
        for Xp in [X_pseudo_pos, X_pseudo_neg]:
            Xp[col] = Xp[col].fillna("Missing")

    X_combined = pd.concat([X_train_real, X_pseudo_pos, X_pseudo_neg], ignore_index=True)
    y_combined = pd.concat([y_train_real, y_pseudo_pos, y_pseudo_neg], ignore_index=True)
    n_real = len(X_train_real)  # indices 0..n_real-1 are real; rest are pseudo

    print(f"  Combined training set: {len(X_combined):,} rows")

    X_combined_lgbm, X_test_lgbm = _encode_for_lgbm(X_combined, X_test, cat_features)

    active_seeds = SEEDS[:10]
    n_seeds = len(active_seeds)
    total_models = N_FOLDS * n_seeds
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    lgbm_params = _load_lgbm_params()
    lgbm_params.update({"device": "gpu", "n_jobs": -1, "verbose": -1})

    # --- CatBoost ---
    print(f"\nTraining CatBoost [pseudo-label] {N_FOLDS}-fold × {n_seeds}-seed ({total_models} models)...")
    cb_oof_real = np.zeros(n_real)   # only track real rows
    cb_oof_cnt  = np.zeros(n_real)   # count how many seed/fold pairs covered each row
    cb_test     = np.zeros(len(X_test))
    cb_times    = []
    cb_start    = time.time()

    for s_idx, seed in enumerate(active_seeds, 1):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_combined, y_combined), 1):
            fold_start = time.time()
            X_tr  = X_combined.iloc[train_idx].copy()
            y_tr  = y_combined.iloc[train_idx]
            X_val = X_combined.iloc[val_idx].copy()

            model = CatBoostClassifier(
                **_CB_PARAMS, eval_metric="Logloss",
                cat_features=cat_features, random_seed=seed,
                task_type="GPU", verbose=0,
            )
            model.fit(X_tr, y_tr)

            # Only record OOF for real rows in val set
            real_val_mask = val_idx < n_real
            real_val_idx  = val_idx[real_val_mask]
            if len(real_val_idx) > 0:
                preds_val = model.predict_proba(X_combined.iloc[real_val_idx])[:, 1]
                cb_oof_real[real_val_idx] += preds_val
                cb_oof_cnt[real_val_idx]  += 1

            cb_test += model.predict_proba(X_test)[:, 1] / total_models

            fold_elapsed = time.time() - fold_start
            cb_times.append(fold_elapsed)
            done = (s_idx - 1) * N_FOLDS + fold
            eta  = (sum(cb_times) / len(cb_times)) * (total_models - done)
            print(f"  CB Seed {s_idx}/{n_seeds}, Fold {fold}/{N_FOLDS} done"
                  f" | fold: {_fmt(fold_elapsed)} | ETA: {_fmt(eta)} | {done}/{total_models}")

    # Normalise by count (some rows may appear in val more than once across seeds)
    cb_oof_cnt = np.maximum(cb_oof_cnt, 1)
    cb_oof_real /= cb_oof_cnt
    cb_auc = roc_auc_score(y_train_real, cb_oof_real)
    print(f"CB [pseudo] OOF AUC (real rows only): {cb_auc:.6f}  (took {_fmt(time.time() - cb_start)})")
    np.save(os.path.join(RESULTS_DIR, f"cb_{save_tag}_oof.npy"),  cb_oof_real)
    np.save(os.path.join(RESULTS_DIR, f"cb_{save_tag}_test.npy"), cb_test)

    # --- LightGBM ---
    print(f"\nTraining LightGBM [pseudo-label] {N_FOLDS}-fold × {n_seeds}-seed ({total_models} models)...")
    lgbm_oof_real = np.zeros(n_real)
    lgbm_oof_cnt  = np.zeros(n_real)
    lgbm_test     = np.zeros(len(X_test_lgbm))
    lgbm_times    = []
    lgbm_start    = time.time()

    for s_idx, seed in enumerate(active_seeds, 1):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_combined_lgbm, y_combined), 1):
            fold_start = time.time()
            X_tr  = X_combined_lgbm.iloc[train_idx]
            y_tr  = y_combined.iloc[train_idx]
            X_val = X_combined_lgbm.iloc[val_idx]

            model = lgb.LGBMClassifier(**lgbm_params, random_state=seed)
            model.fit(X_tr, y_tr, categorical_feature=cat_features)

            real_val_mask = val_idx < n_real
            real_val_idx  = val_idx[real_val_mask]
            if len(real_val_idx) > 0:
                preds_val = model.predict_proba(X_combined_lgbm.iloc[real_val_idx])[:, 1]
                lgbm_oof_real[real_val_idx] += preds_val
                lgbm_oof_cnt[real_val_idx]  += 1

            lgbm_test += model.predict_proba(X_test_lgbm)[:, 1] / total_models

            fold_elapsed = time.time() - fold_start
            lgbm_times.append(fold_elapsed)
            done = (s_idx - 1) * N_FOLDS + fold
            eta  = (sum(lgbm_times) / len(lgbm_times)) * (total_models - done)
            print(f"  LGBM Seed {s_idx}/{n_seeds}, Fold {fold}/{N_FOLDS} done"
                  f" | fold: {_fmt(fold_elapsed)} | ETA: {_fmt(eta)} | {done}/{total_models}")

    lgbm_oof_cnt = np.maximum(lgbm_oof_cnt, 1)
    lgbm_oof_real /= lgbm_oof_cnt
    lgbm_auc = roc_auc_score(y_train_real, lgbm_oof_real)
    print(f"LGBM [pseudo] OOF AUC (real rows only): {lgbm_auc:.6f}  (took {_fmt(time.time() - lgbm_start)})")
    np.save(os.path.join(RESULTS_DIR, f"lgbm_{save_tag}_oof.npy"),  lgbm_oof_real)
    np.save(os.path.join(RESULTS_DIR, f"lgbm_{save_tag}_test.npy"), lgbm_test)

    # --- Optimal blend ---
    best_w, best_auc = 0.5, 0.0
    for w in np.arange(0.0, 1.01, 0.05):
        blended = w * cb_oof_real + (1 - w) * lgbm_oof_real
        auc = roc_auc_score(y_train_real, blended)
        if auc > best_auc:
            best_auc, best_w = auc, w
    print(f"\nOptimal blend [pseudo]: CB {best_w:.2f} + LGBM {1-best_w:.2f}")
    print(f"Blended OOF AUC (real rows): {best_auc:.6f}  (v10 baseline: 0.778272)")

    test_preds = best_w * cb_test + (1 - best_w) * lgbm_test
    submission_df = pd.DataFrame({"ID": test_ids.values, "NO_SHOW_FLG": test_preds})
    submission_df.to_csv(os.path.join(SUBMISSIONS_DIR, output_name), index=False)
    print(f"Submission saved to submissions/{output_name}  ({len(submission_df):,} rows)")
    print(f"Total runtime: {_fmt(time.time() - run_start)}")


# =============================================================================
# Rank blend of top submission CSVs (OOF computed from saved .npy weights)
# =============================================================================

# Known OOF weight compositions for each named submission
_V19_W  = [("cb_20s", 0.497), ("cb_fe", 0.368), ("freq_lgbm", 0.080), ("lgbm_20s", 0.055)]
_V20_W  = [("cb_fe", 0.509), ("cb_20s", 0.323), ("lgbm_20s", 0.090),
           ("freq_lgbm", 0.038), ("lgbm_te", 0.020), ("freq_cb", 0.020)]
_V22_W  = [("cb_pl", 0.799), ("lgbm_pl", 0.141), ("cb_fe", 0.060)]
_V23_W  = [("cb_pl", 0.750), ("lgbm_pl", 0.160), ("cb_fe", 0.090)]
_V24_W  = [("cb_pl", 0.783), ("lgbm_pl", 0.138), ("cb_fe", 0.059), ("xgb", 0.020)]
_V29_W  = [("cb_pl2", 0.710), ("lgbm_pl2", 0.210), ("cb_fe", 0.080)]
_V30_W  = [("cb_pl", 0.6531), ("lgbm_pl2", 0.1633), ("cb_pl2", 0.1441),
           ("lgbm_pl", 0.0196), ("cb_fe", 0.0200)]
_XGB_W  = [("xgb", 1.0)]

_RANK_BLEND_COMBOS = {
    "v22_v20_v19":      [("v22", _V22_W), ("v20", _V20_W), ("v19", _V19_W)],
    "v23_v20_v19":      [("v23", _V23_W), ("v20", _V20_W), ("v19", _V19_W)],
    "v23_v20_v24":      [("v23", _V23_W), ("v20", _V20_W), ("v24", _V24_W)],
    "v23_v20_v19_v24":  [("v23", _V23_W), ("v20", _V20_W), ("v19", _V19_W), ("v24", _V24_W)],
    "v29_v20":          [("v29", _V29_W), ("v20", _V20_W)],
    "v30_v20":          [("v30", _V30_W), ("v20", _V20_W)],
    "v29_v23_v20":      [("v29", _V29_W), ("v23", _V23_W), ("v20", _V20_W)],
    "v30_v23_v20":      [("v30", _V30_W), ("v23", _V23_W), ("v20", _V20_W)],
    "xgb_v20":          [("xgb", _XGB_W), ("v20", _V20_W)],
    "xgb_v30":          [("xgb", _XGB_W), ("v30", _V30_W)],
}


def run_rank_blend(output_name="submission_rank_blend.csv", combo="v22_v20_v19"):
    """Rank-average a set of submissions reconstructed from saved .npy arrays.

    combo options (--rank-blend-combo):
      v22_v20_v19       — original winning blend (LB 0.78264)
      v23_v20_v19       — same structure, v22 → v23 (fine-tuned weights)
      v23_v20_v24       — upgraded anchor (v23) + xgb diversity (v24)
      v23_v20_v19_v24   — 4-way blend, maximum variance reduction
    """
    if combo not in _RANK_BLEND_COMBOS:
        raise ValueError(f"combo must be one of {list(_RANK_BLEND_COMBOS.keys())}, got '{combo}'")

    train_df, test_df, _ = load_data()
    _, y_train, _ = get_feature_target_split(train_df)
    _, test_ids = prepare_test_data(test_df)
    y = y_train.values

    def load_npy(tag, kind):
        path = os.path.join(RESULTS_DIR, f"{tag}_oof.npy" if kind == "oof" else f"{tag}_test.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}")
        return np.load(path)

    print(f"Rank blend: combo='{combo}' ({len(_RANK_BLEND_COMBOS[combo])} submissions)")
    oof_preds  = {}
    test_preds = {}
    for name, weights in _RANK_BLEND_COMBOS[combo]:
        oof_blend  = np.zeros(len(y))
        test_blend = np.zeros(len(test_ids))
        for tag, w in weights:
            oof_blend  += w * load_npy(tag, "oof")
            test_blend += w * load_npy(tag, "test")
        oof_preds[name]  = oof_blend
        test_preds[name] = test_blend
        auc = roc_auc_score(y, oof_blend)
        print(f"  {name} OOF AUC: {auc:.6f}")

    names = list(oof_preds.keys())
    print("\nPairwise OOF correlations:")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            corr = np.corrcoef(oof_preds[a], oof_preds[b])[0, 1]
            print(f"  {a} vs {b}: {corr:.6f}")

    oof_ranks      = {n: oof_preds[n].argsort().argsort().astype(float) for n in names}
    oof_rank_blend = sum(oof_ranks.values()) / len(names)
    rank_oof_auc   = roc_auc_score(y, oof_rank_blend)
    print(f"\nRank-blend OOF AUC  : {rank_oof_auc:.6f}")
    print(f"v23_rank_blend (LB) : 0.78264  (v22+v20+v19)")
    print(f"Net gain vs winner  : {rank_oof_auc - 0.780069:+.6f}  (OOF; LB may differ)")

    test_ranks      = {n: test_preds[n].argsort().argsort().astype(float) for n in names}
    test_rank_blend = sum(test_ranks.values()) / len(names)
    test_rank_norm  = test_rank_blend / test_rank_blend.max()

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    submission_df = pd.DataFrame({"ID": test_ids.values, "NO_SHOW_FLG": test_rank_norm})
    submission_df.to_csv(os.path.join(SUBMISSIONS_DIR, output_name), index=False)
    print(f"\nSubmission saved → submissions/{output_name}  ({len(submission_df):,} rows)")


def run_weighted_rank_blend():
    """Weighted rank blend of v23 + v20 + v24 using hand-tuned rank weights.

    Weights reflect signal strength / role:
      0.45 * rank(v23)  — strongest signal (pseudo-label anchor)
      0.35 * rank(v20)  — stable CB/LGBM blend
      0.20 * rank(v24)  — structural diversity (XGBoost)

    Also generates the simpler 2-way equal blend (v23 + v20) as a backup.
    Saves → submissions/v26_wrank_v23_v20_v24.csv
             submissions/v26_rank_v23_v20.csv
    """
    train_df, test_df, _ = load_data()
    _, y_train, _ = get_feature_target_split(train_df)
    _, test_ids = prepare_test_data(test_df)
    y = y_train.values

    def load_npy(tag, kind):
        path = os.path.join(RESULTS_DIR, f"{tag}_oof.npy" if kind == "oof" else f"{tag}_test.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}")
        return np.load(path)

    # --- Reconstruct each submission's OOF/test predictions ---
    combos = {
        "v23": _V23_W,
        "v20": _V20_W,
        "v24": _V24_W,
    }
    oof_preds  = {}
    test_preds = {}
    for name, weights in combos.items():
        oof_blend  = sum(w * load_npy(tag, "oof")  for tag, w in weights)
        test_blend = sum(w * load_npy(tag, "test") for tag, w in weights)
        oof_preds[name]  = oof_blend
        test_preds[name] = test_blend
        auc = roc_auc_score(y, oof_blend)
        print(f"  {name} OOF AUC: {auc:.6f}")

    print("\nPairwise OOF correlations:")
    names = list(oof_preds.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            corr = np.corrcoef(oof_preds[a], oof_preds[b])[0, 1]
            print(f"  {a} vs {b}: {corr:.6f}")

    # --- Convert to ranks ---
    oof_ranks  = {n: oof_preds[n].argsort().argsort().astype(float)  for n in names}
    test_ranks = {n: test_preds[n].argsort().argsort().astype(float) for n in names}

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

    # 3-way weighted (0.45 / 0.35 / 0.20)
    rank_weights = {"v23": 0.45, "v20": 0.35, "v24": 0.20}
    oof_w  = sum(rank_weights[n] * oof_ranks[n]  for n in names)
    test_w = sum(rank_weights[n] * test_ranks[n] for n in names)
    wrank_oof_auc = roc_auc_score(y, oof_w)
    test_w_norm   = test_w / test_w.max()
    print(f"\n3-way weighted rank (0.45/0.35/0.20) OOF AUC: {wrank_oof_auc:.6f}")
    print(f"Equal-weight (v25_rank_v23_v20_v24 LB)       : 0.78267")
    out3 = "v26_wrank_v23_v20_v24.csv"
    pd.DataFrame({"ID": test_ids.values, "NO_SHOW_FLG": test_w_norm}).to_csv(
        os.path.join(SUBMISSIONS_DIR, out3), index=False
    )
    print(f"Saved → submissions/{out3}")

    # 2-way equal backup (v23 + v20) / 2
    oof_2w  = (oof_ranks["v23"]  + oof_ranks["v20"])  / 2
    test_2w = (test_ranks["v23"] + test_ranks["v20"]) / 2
    rank2_oof_auc = roc_auc_score(y, oof_2w)
    test_2w_norm  = test_2w / test_2w.max()
    print(f"\n2-way equal rank (v23+v20) OOF AUC          : {rank2_oof_auc:.6f}")
    out2 = "v26_rank_v23_v20.csv"
    pd.DataFrame({"ID": test_ids.values, "NO_SHOW_FLG": test_2w_norm}).to_csv(
        os.path.join(SUBMISSIONS_DIR, out2), index=False
    )
    print(f"Saved → submissions/{out2}")


def run_xgb_v30_v20_rank_blends():
    """3-way equal and 3-way weighted rank blends of xgb + v30 + v20.

    Saves two files:
      v34_rank_xgb_v30_v20.csv          — equal 1/3 each
      v34_wrank_xgb_v30_v20.csv         — weighted 0.10/0.45/0.45
    """
    train_df, test_df, _ = load_data()
    _, y_train, _ = get_feature_target_split(train_df)
    _, test_ids = prepare_test_data(test_df)
    y = y_train.values

    def load_npy(tag, kind):
        path = os.path.join(RESULTS_DIR, f"{tag}_oof.npy" if kind == "oof" else f"{tag}_test.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}")
        return np.load(path)

    combos = {
        "xgb": _XGB_W,
        "v30": _V30_W,
        "v20": _V20_W,
    }
    oof_preds  = {}
    test_preds = {}
    for name, weights in combos.items():
        oof_blend  = sum(w * load_npy(tag, "oof")  for tag, w in weights)
        test_blend = sum(w * load_npy(tag, "test") for tag, w in weights)
        oof_preds[name]  = oof_blend
        test_preds[name] = test_blend
        print(f"  {name} OOF AUC: {roc_auc_score(y, oof_blend):.6f}")

    names = list(oof_preds.keys())
    print("\nPairwise OOF correlations:")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            corr = np.corrcoef(oof_preds[a], oof_preds[b])[0, 1]
            print(f"  {a} vs {b}: {corr:.6f}")

    oof_ranks  = {n: oof_preds[n].argsort().argsort().astype(float) for n in names}
    test_ranks = {n: test_preds[n].argsort().argsort().astype(float) for n in names}

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

    # Equal 3-way (1/3 each)
    oof_eq  = sum(oof_ranks[n]  for n in names) / 3
    test_eq = sum(test_ranks[n] for n in names) / 3
    print(f"\n3-way equal rank OOF AUC  : {roc_auc_score(y, oof_eq):.6f}")
    out_eq = "v34_rank_xgb_v30_v20.csv"
    pd.DataFrame({"ID": test_ids.values, "NO_SHOW_FLG": test_eq / test_eq.max()}).to_csv(
        os.path.join(SUBMISSIONS_DIR, out_eq), index=False
    )
    print(f"Saved → submissions/{out_eq}")

    # Weighted 3-way (xgb=0.10, v30=0.45, v20=0.45)
    rank_w = {"xgb": 0.10, "v30": 0.45, "v20": 0.45}
    oof_w  = sum(rank_w[n] * oof_ranks[n]  for n in names)
    test_w = sum(rank_w[n] * test_ranks[n] for n in names)
    print(f"3-way weighted rank OOF AUC: {roc_auc_score(y, oof_w):.6f}  (xgb=0.10, v30=0.45, v20=0.45)")
    out_w = "v34_wrank_xgb_v30_v20.csv"
    pd.DataFrame({"ID": test_ids.values, "NO_SHOW_FLG": test_w / test_w.max()}).to_csv(
        os.path.join(SUBMISSIONS_DIR, out_w), index=False
    )
    print(f"Saved → submissions/{out_w}")


def run_score_blend_v23_v20():
    """Score-average (probability space) blend of v23 and v20.

    Unlike rank blending, averages raw probabilities directly.
    Saves two files:
      v27_score_v23_v20.csv         — equal 50/50 score blend
      v27_score_v23_only.csv        — v23 standalone (isolation test)
    """
    train_df, test_df, _ = load_data()
    _, y_train, _ = get_feature_target_split(train_df)
    _, test_ids = prepare_test_data(test_df)
    y = y_train.values

    def load_npy(tag, kind):
        path = os.path.join(RESULTS_DIR, f"{tag}_oof.npy" if kind == "oof" else f"{tag}_test.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}")
        return np.load(path)

    oof_v23  = sum(w * load_npy(tag, "oof")  for tag, w in _V23_W)
    oof_v20  = sum(w * load_npy(tag, "oof")  for tag, w in _V20_W)
    test_v23 = sum(w * load_npy(tag, "test") for tag, w in _V23_W)
    test_v20 = sum(w * load_npy(tag, "test") for tag, w in _V20_W)

    print(f"  v23 OOF AUC : {roc_auc_score(y, oof_v23):.6f}")
    print(f"  v20 OOF AUC : {roc_auc_score(y, oof_v20):.6f}")

    # Equal score blend
    oof_blend  = (oof_v23  + oof_v20)  / 2
    test_blend = (test_v23 + test_v20) / 2
    print(f"  score blend OOF AUC: {roc_auc_score(y, oof_blend):.6f}")
    print(f"  rank  blend OOF AUC (v26 LB=0.78268): reference")

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    pd.DataFrame({"ID": test_ids.values, "NO_SHOW_FLG": test_blend}).to_csv(
        os.path.join(SUBMISSIONS_DIR, "v27_score_v23_v20.csv"), index=False
    )
    print("Saved → submissions/v27_score_v23_v20.csv")

    # v23 standalone
    pd.DataFrame({"ID": test_ids.values, "NO_SHOW_FLG": test_v23}).to_csv(
        os.path.join(SUBMISSIONS_DIR, "v27_score_v23_only.csv"), index=False
    )
    print("Saved → submissions/v27_score_v23_only.csv")


def run_score_blend_v29_v20(output_name="v32_score_v29_v20.csv"):
    """Equal 50/50 raw-probability score blend of v29 and v20.

    Analog of v27_score_v23_v20 but using pl2 fine-weight blend (v29) instead of v23.
    Saves → submissions/{output_name}
    """
    train_df, test_df, _ = load_data()
    _, y_train, _ = get_feature_target_split(train_df)
    _, test_ids = prepare_test_data(test_df)
    y = y_train.values

    def load_npy(tag, kind):
        path = os.path.join(RESULTS_DIR, f"{tag}_oof.npy" if kind == "oof" else f"{tag}_test.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}")
        return np.load(path)

    oof_v29  = sum(w * load_npy(tag, "oof")  for tag, w in _V29_W)
    oof_v20  = sum(w * load_npy(tag, "oof")  for tag, w in _V20_W)
    test_v29 = sum(w * load_npy(tag, "test") for tag, w in _V29_W)
    test_v20 = sum(w * load_npy(tag, "test") for tag, w in _V20_W)

    print(f"  v29 OOF AUC : {roc_auc_score(y, oof_v29):.6f}")
    print(f"  v20 OOF AUC : {roc_auc_score(y, oof_v20):.6f}")

    oof_blend  = (oof_v29  + oof_v20)  / 2
    test_blend = (test_v29 + test_v20) / 2
    print(f"  score blend OOF AUC: {roc_auc_score(y, oof_blend):.6f}")

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    pd.DataFrame({"ID": test_ids.values, "NO_SHOW_FLG": test_blend}).to_csv(
        os.path.join(SUBMISSIONS_DIR, output_name), index=False
    )
    print(f"Saved -> submissions/{output_name}")


def run_score_blend_v30_v20(output_name="v33_score_v30_v20.csv"):
    """Equal 50/50 raw-probability score blend of v30 and v20.

    Analog of run_score_blend_v29_v20 but using greedy pl2 blend (v30) instead of pl2 fine-weight (v29).
    Saves → submissions/{output_name}
    """
    train_df, test_df, _ = load_data()
    _, y_train, _ = get_feature_target_split(train_df)
    _, test_ids = prepare_test_data(test_df)
    y = y_train.values

    def load_npy(tag, kind):
        path = os.path.join(RESULTS_DIR, f"{tag}_oof.npy" if kind == "oof" else f"{tag}_test.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}")
        return np.load(path)

    oof_v30  = sum(w * load_npy(tag, "oof")  for tag, w in _V30_W)
    oof_v20  = sum(w * load_npy(tag, "oof")  for tag, w in _V20_W)
    test_v30 = sum(w * load_npy(tag, "test") for tag, w in _V30_W)
    test_v20 = sum(w * load_npy(tag, "test") for tag, w in _V20_W)

    print(f"  v30 OOF AUC : {roc_auc_score(y, oof_v30):.6f}")
    print(f"  v20 OOF AUC : {roc_auc_score(y, oof_v20):.6f}")

    oof_blend  = (oof_v30  + oof_v20)  / 2
    test_blend = (test_v30 + test_v20) / 2
    print(f"  score blend OOF AUC: {roc_auc_score(y, oof_blend):.6f}")

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    pd.DataFrame({"ID": test_ids.values, "NO_SHOW_FLG": test_blend}).to_csv(
        os.path.join(SUBMISSIONS_DIR, output_name), index=False
    )
    print(f"Saved -> submissions/{output_name}")


def run_sweep_rank_v23_v20():
    """Sweep weight w for: w*rank(v23) + (1-w)*rank(v20), w in [0.05, 0.95] step 0.05.

    Reports OOF AUC for each weight, prints the best, and saves the best-weight
    submission as v27_wrank_sweep_v23_v20.csv.
    """
    train_df, test_df, _ = load_data()
    _, y_train, _ = get_feature_target_split(train_df)
    _, test_ids = prepare_test_data(test_df)
    y = y_train.values

    def load_npy(tag, kind):
        path = os.path.join(RESULTS_DIR, f"{tag}_oof.npy" if kind == "oof" else f"{tag}_test.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}")
        return np.load(path)

    oof_v23  = sum(w * load_npy(tag, "oof")  for tag, w in _V23_W)
    oof_v20  = sum(w * load_npy(tag, "oof")  for tag, w in _V20_W)
    test_v23 = sum(w * load_npy(tag, "test") for tag, w in _V23_W)
    test_v20 = sum(w * load_npy(tag, "test") for tag, w in _V20_W)

    rank_oof_v23  = oof_v23.argsort().argsort().astype(float)
    rank_oof_v20  = oof_v20.argsort().argsort().astype(float)
    rank_test_v23 = test_v23.argsort().argsort().astype(float)
    rank_test_v20 = test_v20.argsort().argsort().astype(float)

    print(f"  v23 OOF AUC: {roc_auc_score(y, oof_v23):.6f}")
    print(f"  v20 OOF AUC: {roc_auc_score(y, oof_v20):.6f}")
    print(f"  equal (0.50) OOF AUC (v26 LB=0.78268): {roc_auc_score(y, rank_oof_v23 + rank_oof_v20):.6f}")
    print(f"\n{'w_v23':>6}  {'OOF AUC':>10}")
    print("-" * 20)

    best_w, best_auc = 0.5, 0.0
    for w in np.arange(0.05, 1.00, 0.05):
        w = round(w, 2)
        blended = w * rank_oof_v23 + (1 - w) * rank_oof_v20
        auc = roc_auc_score(y, blended)
        marker = " ← best" if auc > best_auc else ""
        print(f"  {w:.2f}   {auc:.6f}{marker}")
        if auc > best_auc:
            best_auc, best_w = auc, w

    print(f"\nBest weight: w_v23={best_w:.2f}, OOF AUC={best_auc:.6f}")
    test_blend = best_w * rank_test_v23 + (1 - best_w) * rank_test_v20
    test_norm  = test_blend / test_blend.max()

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    out = "v27_wrank_sweep_v23_v20.csv"
    pd.DataFrame({"ID": test_ids.values, "NO_SHOW_FLG": test_norm}).to_csv(
        os.path.join(SUBMISSIONS_DIR, out), index=False
    )
    print(f"Saved → submissions/{out}  (w_v23={best_w:.2f})")


def run_rank_blend_all():
    """Generate all rank blend combos at once and report OOF for each."""
    _OUTPUT_NAMES = {
        "v22_v20_v19":     "v23_rank_blend.csv",         # already submitted (reference)
        "v23_v20_v19":     "v25_rank_v23_v20_v19.csv",
        "v23_v20_v24":     "v25_rank_v23_v20_v24.csv",
        "v23_v20_v19_v24": "v25_rank_v23_v20_v19_v24.csv",
    }
    for combo, out in _OUTPUT_NAMES.items():
        print(f"\n{'='*60}")
        run_rank_blend(output_name=out, combo=combo)


# =============================================================================
# Greedy ensemble selection (hill-climbing over saved OOF predictions)
# =============================================================================

def run_greedy_ensemble(output_name="submission_greedy.csv", force_start=None):
    """Hill-climb over all saved OOF/test .npy pairs to find optimal ensemble.

    Algorithm (Caruana et al. 2004 "Ensemble Selection from Libraries"):
    1. Start with the single best model by OOF AUC (or force_start if given).
    2. In each round: for every candidate model, try blending the current
       ensemble with that model at weights w in linspace(0.02, 0.5).
       Keep the addition (model + weight) that maximally improves OOF AUC.
    3. Repeat until no candidate improves OOF AUC.
    4. Apply the found weights to test predictions and save submission.

    No training required — runs in under 1 second.
    force_start: label string to force as seed model (e.g. "cb_v10")
    """
    train_df, test_df, _ = load_data()
    _, y_train, _ = get_feature_target_split(train_df)
    _, test_ids = prepare_test_data(test_df)
    y = y_train.values

    # Candidate pool: (label, oof_file, test_file)
    CANDIDATES = [
        ("cb_v10",    "cb_v10_oof.npy",    "cb_v10_test.npy"),
        ("cb_20s",    "cb_20s_oof.npy",    "cb_20s_test.npy"),
        ("cb_d9",     "cb_d9_oof.npy",     "cb_d9_test.npy"),
        ("cb_fe",     "cb_fe_oof.npy",     "cb_fe_test.npy"),
        ("freq_cb",   "freq_cb_oof.npy",   "freq_cb_test.npy"),
        ("lgbm_20s",  "lgbm_20s_oof.npy",  "lgbm_20s_test.npy"),
        ("lgbm_fe",   "lgbm_fe_oof.npy",   "lgbm_fe_test.npy"),
        ("freq_lgbm", "freq_lgbm_oof.npy", "freq_lgbm_test.npy"),
        ("lgbm_te",   "lgbm_te_oof.npy",   "lgbm_te_test.npy"),
        ("cb_pl",     "cb_pl_oof.npy",     "cb_pl_test.npy"),
        ("lgbm_pl",   "lgbm_pl_oof.npy",   "lgbm_pl_test.npy"),
        ("cb_pl2",    "cb_pl2_oof.npy",    "cb_pl2_test.npy"),
        ("lgbm_pl2",  "lgbm_pl2_oof.npy",  "lgbm_pl2_test.npy"),
        ("xgb",       "xgb_oof.npy",       "xgb_test.npy"),
    ]

    print("\nLoading OOF arrays...")
    pool = []
    for label, oof_f, test_f in CANDIDATES:
        oof_path  = os.path.join(RESULTS_DIR, oof_f)
        test_path = os.path.join(RESULTS_DIR, test_f)
        if not os.path.exists(oof_path) or not os.path.exists(test_path):
            print(f"  Skipping {label} — files not found")
            continue
        oof_arr  = np.load(oof_path)
        test_arr = np.load(test_path)
        if len(oof_arr) != len(y):
            print(f"  Skipping {label} — OOF length {len(oof_arr)} != {len(y)}")
            continue
        auc = roc_auc_score(y, oof_arr)
        pool.append({"label": label, "oof": oof_arr, "test": test_arr, "solo_auc": auc})
        print(f"  {label:<12} OOF AUC = {auc:.6f}")

    if not pool:
        raise RuntimeError("No valid OOF files found in results/")

    # Seed ensemble with the best single model (or forced start)
    pool.sort(key=lambda x: x["solo_auc"], reverse=True)
    if force_start is not None:
        seed_candidates = [m for m in pool if m["label"] == force_start]
        if not seed_candidates:
            raise ValueError(f"force_start='{force_start}' not found in pool. "
                             f"Available: {[m['label'] for m in pool]}")
        best_model = seed_candidates[0]
        remaining  = [m for m in pool if m["label"] != force_start]
    else:
        best_model = pool[0]
        remaining  = pool[1:]

    ensemble_oof   = best_model["oof"].copy()
    ensemble_test  = best_model["test"].copy()
    ensemble_parts = [(best_model["label"], 1.0)]
    best_auc       = roc_auc_score(y, ensemble_oof)

    print(f"\nSeed model: '{best_model['label']}'  OOF AUC = {best_auc:.6f}")
    print("-" * 60)

    weights_to_try = np.round(np.concatenate([
        np.arange(0.02, 0.10, 0.02),
        np.arange(0.10, 0.55, 0.05),
    ]), 4)

    improved = True
    while improved and remaining:
        improved = False
        best_gain      = 0.0
        best_candidate = None
        best_w_add     = None
        best_new_oof   = None
        best_new_test  = None

        for cand in remaining:
            for w in weights_to_try:
                trial_oof = (1 - w) * ensemble_oof + w * cand["oof"]
                trial_auc = roc_auc_score(y, trial_oof)
                gain = trial_auc - best_auc
                if gain > best_gain:
                    best_gain      = gain
                    best_candidate = cand
                    best_w_add     = w
                    best_new_oof   = trial_oof
                    best_new_test  = (1 - w) * ensemble_test + w * cand["test"]

        if best_candidate is not None and best_gain > 1e-7:
            ensemble_parts = [(lbl, ww * (1 - best_w_add)) for lbl, ww in ensemble_parts]
            ensemble_parts.append((best_candidate["label"], best_w_add))
            ensemble_oof  = best_new_oof
            ensemble_test = best_new_test
            best_auc += best_gain
            remaining = [c for c in remaining if c["label"] != best_candidate["label"]]
            print(f"  +{best_candidate['label']:<12}  w={best_w_add:.2f}  "
                  f"OOF AUC = {best_auc:.6f}  (+{best_gain:.6f})")
            improved = True
        else:
            print("  No further improvement. Stopping.")

    print(f"\nFinal ensemble ({len(ensemble_parts)} models):")
    for lbl, ww in ensemble_parts:
        print(f"  {lbl:<12}  weight = {ww:.4f}")
    baseline_auc = pool[0]["solo_auc"]
    print(f"Final OOF AUC : {best_auc:.6f}")
    print(f"Baseline (cb_v10): {baseline_auc:.6f}")
    print(f"Net gain        : {best_auc - baseline_auc:+.6f}")

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    submission_df = pd.DataFrame({"ID": test_ids.values, "NO_SHOW_FLG": ensemble_test})
    submission_df.to_csv(os.path.join(SUBMISSIONS_DIR, output_name), index=False)
    print(f"\nSubmission saved → submissions/{output_name}  ({len(submission_df):,} rows)")

    # Return result dict so multistart can compare runs
    return {
        "auc":   best_auc,
        "parts": ensemble_parts,
        "oof":   ensemble_oof,
        "test":  ensemble_test,
    }


# =============================================================================
# Multi-start greedy: run greedy from every possible seed, keep best OOF
# =============================================================================

def run_greedy_multistart(output_name="submission_greedy_ms.csv"):
    """Run greedy ensemble selection from every candidate as starting model.

    Greedy hill-climbing is order-dependent: the choice of seed model can
    lead to different local optima. By trying all N starting models and
    keeping the best overall OOF AUC, we get a more robust result.

    No training required — runs in a few seconds.
    """
    train_df, test_df, _ = load_data()
    _, y_train, _ = get_feature_target_split(train_df)
    _, test_ids = prepare_test_data(test_df)
    y = y_train.values

    CANDIDATES = [
        ("cb_v10",    "cb_v10_oof.npy",    "cb_v10_test.npy"),
        ("cb_20s",    "cb_20s_oof.npy",    "cb_20s_test.npy"),
        ("cb_d9",     "cb_d9_oof.npy",     "cb_d9_test.npy"),
        ("cb_fe",     "cb_fe_oof.npy",     "cb_fe_test.npy"),
        ("freq_cb",   "freq_cb_oof.npy",   "freq_cb_test.npy"),
        ("lgbm_20s",  "lgbm_20s_oof.npy",  "lgbm_20s_test.npy"),
        ("lgbm_fe",   "lgbm_fe_oof.npy",   "lgbm_fe_test.npy"),
        ("freq_lgbm", "freq_lgbm_oof.npy", "freq_lgbm_test.npy"),
        ("lgbm_te",   "lgbm_te_oof.npy",   "lgbm_te_test.npy"),
        ("cb_pl",     "cb_pl_oof.npy",     "cb_pl_test.npy"),
        ("lgbm_pl",   "lgbm_pl_oof.npy",   "lgbm_pl_test.npy"),
        ("cb_pl2",    "cb_pl2_oof.npy",    "cb_pl2_test.npy"),
        ("lgbm_pl2",  "lgbm_pl2_oof.npy",  "lgbm_pl2_test.npy"),
        ("xgb",       "xgb_oof.npy",       "xgb_test.npy"),
    ]

    # Load available pool once
    pool = []
    for label, oof_f, test_f in CANDIDATES:
        oof_path  = os.path.join(RESULTS_DIR, oof_f)
        test_path = os.path.join(RESULTS_DIR, test_f)
        if not os.path.exists(oof_path) or not os.path.exists(test_path):
            continue
        oof_arr  = np.load(oof_path)
        test_arr = np.load(test_path)
        if len(oof_arr) != len(y):
            continue
        pool.append({"label": label, "oof": oof_arr, "test": test_arr,
                     "solo_auc": roc_auc_score(y, oof_arr)})

    if not pool:
        raise RuntimeError("No valid OOF files found.")

    available_labels = [m["label"] for m in pool]
    print(f"Pool: {available_labels}\n")

    weights_to_try = np.round(np.concatenate([
        np.arange(0.02, 0.10, 0.02),
        np.arange(0.10, 0.55, 0.05),
    ]), 4)

    def _run_from(seed_label):
        seed     = next(m for m in pool if m["label"] == seed_label)
        ens_oof  = seed["oof"].copy()
        ens_test = seed["test"].copy()
        parts    = [(seed_label, 1.0)]
        cur_auc  = roc_auc_score(y, ens_oof)
        rem      = [m for m in pool if m["label"] != seed_label]

        improved = True
        while improved and rem:
            improved = False
            best_gain = 0.0
            best_cand = best_w_add = best_new_oof = best_new_test = None
            for cand in rem:
                for w in weights_to_try:
                    trial_oof = (1 - w) * ens_oof + w * cand["oof"]
                    gain = roc_auc_score(y, trial_oof) - cur_auc
                    if gain > best_gain:
                        best_gain     = gain
                        best_cand     = cand
                        best_w_add    = w
                        best_new_oof  = trial_oof
                        best_new_test = (1 - w) * ens_test + w * cand["test"]
            if best_cand is not None and best_gain > 1e-7:
                parts    = [(lbl, ww * (1 - best_w_add)) for lbl, ww in parts]
                parts.append((best_cand["label"], best_w_add))
                ens_oof  = best_new_oof
                ens_test = best_new_test
                cur_auc += best_gain
                rem = [m for m in rem if m["label"] != best_cand["label"]]
                improved = True

        return {"auc": cur_auc, "parts": parts, "oof": ens_oof, "test": ens_test,
                "seed": seed_label}

    best_result = None
    for m in pool:
        res = _run_from(m["label"])
        model_str = " + ".join(f"{lbl}×{ww:.3f}" for lbl, ww in res["parts"])
        marker = ""
        if best_result is None or res["auc"] > best_result["auc"]:
            best_result = res
            marker = "  ← best so far"
        print(f"  Start={m['label']:<12}  OOF={res['auc']:.6f}  models={len(res['parts'])}{marker}")

    print(f"\n{'='*60}")
    print(f"Best start: '{best_result['seed']}'  OOF AUC = {best_result['auc']:.6f}")
    print(f"Final ensemble ({len(best_result['parts'])} models):")
    for lbl, ww in best_result["parts"]:
        print(f"  {lbl:<12}  weight = {ww:.4f}")

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    submission_df = pd.DataFrame({"ID": test_ids.values, "NO_SHOW_FLG": best_result["test"]})
    submission_df.to_csv(os.path.join(SUBMISSIONS_DIR, output_name), index=False)
    print(f"\nSubmission saved → submissions/{output_name}  ({len(submission_df):,} rows)")


# =============================================================================
# Fine weight grid search around a fixed set of models
# =============================================================================

def run_fine_weight_search(output_name="submission_fine_weights.csv", step=0.02, combo="v19"):
    """Dense weight grid search over a fixed model combo.

    combo='v19' (default): cb_20s + cb_fe + freq_lgbm + lgbm_20s  (4 models)
    combo='v22': cb_pl + lgbm_pl + cb_fe  (3 models from v22 greedy)

    step=0.02 → runs in < 1 second.
    step=0.01 → 3-model: ~161k combos ~5s; 4-model: same.
    """
    train_df, test_df, _ = load_data()
    _, y_train, _ = get_feature_target_split(train_df)
    _, test_ids = prepare_test_data(test_df)
    y = y_train.values

    _COMBOS = {
        "v19": [
            ("cb_20s",    "cb_20s_oof.npy",    "cb_20s_test.npy"),
            ("cb_fe",     "cb_fe_oof.npy",     "cb_fe_test.npy"),
            ("freq_lgbm", "freq_lgbm_oof.npy", "freq_lgbm_test.npy"),
            ("lgbm_20s",  "lgbm_20s_oof.npy",  "lgbm_20s_test.npy"),
        ],
        "v22": [
            ("cb_pl",   "cb_pl_oof.npy",   "cb_pl_test.npy"),
            ("lgbm_pl", "lgbm_pl_oof.npy", "lgbm_pl_test.npy"),
            ("cb_fe",   "cb_fe_oof.npy",   "cb_fe_test.npy"),
        ],
        "v22_pl2": [
            ("cb_pl2",   "cb_pl2_oof.npy",   "cb_pl2_test.npy"),
            ("lgbm_pl2", "lgbm_pl2_oof.npy", "lgbm_pl2_test.npy"),
            ("cb_fe",    "cb_fe_oof.npy",     "cb_fe_test.npy"),
        ],
    }
    if combo not in _COMBOS:
        raise ValueError(f"combo must be one of {list(_COMBOS.keys())}, got '{combo}'")
    MODEL_FILES = _COMBOS[combo]
    print(f"Fine weight search: combo='{combo}' ({len(MODEL_FILES)} models), step={step}")

    oof_arrays  = []
    test_arrays = []
    labels      = []
    for label, oof_f, test_f in MODEL_FILES:
        oof_path  = os.path.join(RESULTS_DIR, oof_f)
        test_path = os.path.join(RESULTS_DIR, test_f)
        if not os.path.exists(oof_path):
            raise FileNotFoundError(f"Missing {oof_f} — run the model first")
        oof_arrays.append(np.load(oof_path))
        test_arrays.append(np.load(test_path))
        labels.append(label)

    n = len(labels)
    # Stack for fast matrix ops: shape (n_models, n_train)
    OOF  = np.stack(oof_arrays,  axis=0)
    TEST = np.stack(test_arrays, axis=0)

    # Pre-sort y once for a fast Wilcoxon/AUC calculation.
    # roc_auc_score on 169k rows is ~30ms each; with 161k combos that's 80 min.
    # Instead we use a rank-based trick: AUC = mean_rank(positives) / n_neg - offset.
    # Specifically: AUC = (sum of ranks of positives among all sorted scores) / (n_pos*n_neg) - ...
    # We precompute sorted order once per weight combo via argsort on the blended vector.
    # Still faster via a closed-form: after blending, rank-sum of positives gives AUC.
    #
    # Fastest approach for simplex grid: enumerate all weight combos as numpy array,
    # batch-compute blended OOF matrix (n_combos × n_train), then vectorised rank-AUC.
    # At step=0.02 → ~1140 combos → fast. At step=0.01 → ~161k → still manageable.

    grid = np.round(np.arange(0.0, 1.0 + step / 2, step), 10)

    # Build all weight combos on the simplex — works for any n models
    n_steps = int(round(1.0 / step))
    n_m = len(labels)

    def _enum_simplex(dims, budget):
        if dims == 1:
            yield (budget,)
            return
        for k in range(budget + 1):
            for rest in _enum_simplex(dims - 1, budget - k):
                yield (k,) + rest

    combos = [[k * step for k in combo] for combo in _enum_simplex(n_m, n_steps)]
    W = np.array(combos, dtype=np.float64)
    print(f"Searching {len(W):,} weight combinations (step={step}, {len(labels)} models)...")

    # Blended OOF matrix: (n_combos, n_train) — may be large; process in chunks
    # to stay within memory. chunk_size rows at a time.
    y_arr  = y.astype(np.float64)
    n_pos  = y_arr.sum()
    n_neg  = len(y_arr) - n_pos

    # Vectorised AUC via Mann-Whitney U statistic:
    #   AUC = (U) / (n_pos * n_neg)
    # where U = sum of ranks of positives - n_pos*(n_pos+1)/2
    # We compute ranks per combo using argsort. Still O(n_combos * n * log n).
    # For 161k combos this is ~10s with numpy.
    pos_idx = np.where(y_arr == 1)[0]
    chunk   = 2000  # process this many combos at once

    best_auc = 0.0
    best_w   = W[0].copy()

    for start in range(0, len(W), chunk):
        wchunk  = W[start:start + chunk]            # (c, 4)
        blended = wchunk @ OOF                      # (c, n_train)
        # argsort each row, then rank positives
        order   = np.argsort(blended, axis=1)       # (c, n_train)
        ranks   = np.empty_like(order, dtype=np.int64)
        np.put_along_axis(ranks, order,
                          np.arange(1, OOF.shape[1] + 1, dtype=np.int64)[None, :],
                          axis=1)                   # 1-based ranks
        rank_sum = ranks[:, pos_idx].sum(axis=1)    # (c,)
        u_stat   = rank_sum - n_pos * (n_pos + 1) / 2
        aucs     = u_stat / (n_pos * n_neg)
        idx      = int(np.argmax(aucs))
        if aucs[idx] > best_auc:
            best_auc = float(aucs[idx])
            best_w   = wchunk[idx].copy()

    print(f"Searched {len(W):,} weight combinations (step={step})")
    _BASELINES = {"v19": 0.778990, "v22": 0.780069}
    baseline = _BASELINES.get(combo, 0.0)
    print(f"\nBest weights:")
    for lbl, ww in zip(labels, best_w):
        print(f"  {lbl:<12}  weight = {ww:.4f}")
    print(f"Best OOF AUC : {best_auc:.6f}")
    print(f"Baseline ({combo}) : {baseline:.6f}")
    print(f"Net gain     : {best_auc - baseline:+.6f}")

    test_preds = best_w @ TEST
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    submission_df = pd.DataFrame({"ID": test_ids.values, "NO_SHOW_FLG": test_preds})
    submission_df.to_csv(os.path.join(SUBMISSIONS_DIR, output_name), index=False)
    print(f"\nSubmission saved → submissions/{output_name}  ({len(submission_df):,} rows)")


# =============================================================================
# XGBoost candidate (10 seeds × 5 folds, saves xgb_oof.npy / xgb_test.npy)
# =============================================================================

def run_xgb_candidate(n_seeds=10, tag="xgb"):
    """Train XGBoost 10-seed × 5-fold, save OOF + test arrays for greedy pool.

    Uses GPU histogram if available, falls back to CPU hist automatically.
    XGBoost uses a different second-order approximation and regularization than
    CB/LGBM — structurally different predictions = greedy diversity.
    """
    import xgboost as xgb

    train_df, test_df, _ = load_data()
    X_train, y_train, _ = get_feature_target_split(train_df)
    X_test,  test_ids   = prepare_test_data(test_df)
    cat_features = get_cat_features(X_train)

    # Ordinal-encode cats for XGBoost
    X_train, X_test = _encode_for_lgbm(X_train, X_test, cat_features)

    y = y_train.values
    n_real = len(y)
    seeds  = SEEDS[:n_seeds]
    total_models = n_seeds * N_FOLDS

    _XGB_PARAMS = {
        "n_estimators":      1000,
        "learning_rate":     0.05,
        "max_depth":         6,
        "min_child_weight":  5,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "reg_alpha":         0.1,
        "reg_lambda":        1.0,
        "eval_metric":       "auc",
        "early_stopping_rounds": 50,
        "tree_method":       "hist",
        "device":            "cuda",   # auto-fallback to cpu if no GPU
        "verbosity":         0,
    }

    print(f"\nTraining XGBoost [{tag}] {N_FOLDS}-fold × {n_seeds}-seed ({total_models} models)...")
    run_start = time.time()

    oof_preds = np.zeros(n_real)
    oof_cnt   = np.zeros(n_real)
    test_preds_accum = np.zeros(len(test_ids))
    done = 0

    for seed in seeds:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        fold_start = time.time()
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y[tr_idx],             y[val_idx]

            params = _XGB_PARAMS.copy()
            params["random_state"] = seed

            model = xgb.XGBClassifier(**params)
            try:
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
            except Exception:
                # Fallback to CPU if CUDA not available
                params["device"] = "cpu"
                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )

            oof_preds[val_idx] += model.predict_proba(X_val)[:, 1]
            oof_cnt[val_idx]   += 1
            test_preds_accum   += model.predict_proba(X_test)[:, 1]
            done += 1

        fold_elapsed = time.time() - fold_start
        eta = (time.time() - run_start) / done * (total_models - done)
        print(f"  seed={seed} | fold: {_fmt(fold_elapsed)} | ETA: {_fmt(eta)} | {done}/{total_models}")

    oof_cnt   = np.maximum(oof_cnt, 1)
    oof_preds /= oof_cnt
    test_preds_accum /= total_models

    auc = roc_auc_score(y, oof_preds)
    print(f"\nXGBoost [{tag}] OOF AUC: {auc:.6f}  (took {_fmt(time.time() - run_start)})")
    print(f"v10 baseline: 0.778272  |  net: {auc - 0.778272:+.6f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(os.path.join(RESULTS_DIR, f"{tag}_oof.npy"),  oof_preds)
    np.save(os.path.join(RESULTS_DIR, f"{tag}_test.npy"), test_preds_accum)
    print(f"Saved {tag}_oof.npy and {tag}_test.npy  →  ready for greedy ensemble")
