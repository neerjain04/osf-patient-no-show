# =============================================================================
# tune.py
# -----------------------------------------------------------------------------
# Automated hyperparameter search using Optuna.
#
# What is Optuna?
#   Optuna is a "define-by-run" hyperparameter optimisation framework.
#   Instead of grid search (try every combination) or random search (random
#   samples), Optuna uses Bayesian optimisation (TPE sampler) to intelligently
#   focus future trials on the regions of the search space that looked promising
#   in previous trials. This means you get better results with fewer trials.
#
# How it works here:
#   1. Each "trial" picks a new set of hyperparameters from the search space.
#   2. We train CatBoost with 3-fold CV on those hyperparameters and return the
#      mean AUC (purposely using 3 folds instead of 5 to keep each trial fast).
#   3. Optuna records whether that was a good or bad score, then proposes the
#      next set of hyperparameters more likely to beat the current best.
#   4. After N_TRIALS trials, the best parameters are saved to results/ and
#      printed so you can paste them into train_baseline.py / submission.py.
#
# Why not tune on 5 folds?
#   With 50 trials × 5 folds = 250 full CatBoost training runs. On 168k rows
#   that takes ~2 hours. Using 3 folds × 50 trials = 150 runs, which is still
#   highly informative and finishes in about 30–40 minutes.
#
# Output saved to results/:
#   best_params.json        — the optimal hyperparameter set found
#   optuna_results.csv      — all trials with their AUC scores (for inspection)
# =============================================================================

import os
import json
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

from data_utils import load_data, get_feature_target_split, get_cat_features

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N_TRIALS = 100   # number of Optuna trials; more = better, but slower
N_FOLDS = 3      # folds per trial; 3 gives better AUC estimates now that GPU makes each trial fast


def objective(trial, X, y, cat_features):
    """
    The objective function Optuna calls for each trial.

    trial.suggest_* functions define the search space:
      - suggest_int(name, low, high)          → sample an integer in [low, high]
      - suggest_float(name, low, high, log=True) → sample a float; log=True
            samples in log space so it explores small values (0.01) as carefully
            as large ones (0.1), which is important for learning_rate.

    Returns the mean validation AUC across all folds (higher = better).
    Optuna will try to MAXIMISE this value.
    """

    # --- Hyperparameter search space ---
    # depth: deeper trees capture more complex patterns but overfit more easily.
    # Cap at 7 — depth 8 makes trials 2x slower with minimal AUC gain during search.
    depth = trial.suggest_int("depth", 4, 7)

    # iterations: set a hard cap of 800 — early_stopping_rounds=50 will stop well
    # before this if the model has converged, so the ceiling rarely matters.
    # Keeping this low is the single biggest speed-up: 800 vs 2000 = ~3x faster.
    iterations = trial.suggest_int("iterations", 300, 800)

    # learning_rate: step size for each tree's correction. Lower = more stable
    # but needs more iterations. Sampled in log space to explore 0.01 and 0.1
    # with equal density.
    learning_rate = trial.suggest_float("learning_rate", 0.005, 0.15, log=True)

    # l2_leaf_reg: L2 regularisation on leaf weights. Higher values prevent
    # leaves from memorising individual training examples (reduces overfitting).
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 10.0)

    # bagging_temperature: controls the randomness of subsample selection
    # (Bayesian bootstrap). 0 = full dataset each iteration, 1 = standard
    # bootstrap, >1 = more aggressive subsampling.
    bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 2.0)

    # --- 3-fold cross-validation ---
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # Fill missing categoricals with the literal string "Missing" — CatBoost
        # requires all categorical columns to be non-null strings.
        for col in cat_features:
            X_train[col] = X_train[col].fillna("Missing")
            X_val[col] = X_val[col].fillna("Missing")

        model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            bagging_temperature=bagging_temperature,
            eval_metric="Logloss",  # AUC not supported on GPU; Logloss works fine for early stopping
            cat_features=cat_features,
            random_seed=42,
            task_type="GPU",
            verbose=0,
            early_stopping_rounds=50,
        )

        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,  # keep the checkpoint with best eval AUC
        )

        preds = model.predict_proba(X_val)[:, 1]
        fold_scores.append(roc_auc_score(y_val, preds))

        # Optuna pruning: if this fold already looks much worse than the current
        # best trial, abort early to save time. This is called "pruning".
        # report() tells the pruner the intermediate value after each fold.
        trial.report(np.mean(fold_scores), fold_idx)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(fold_scores)


def run_tuning(n_trials=N_TRIALS):
    """
    Runs the full Optuna study and saves the best parameters.

    Steps:
      1. Load training data
      2. Create an Optuna study (direction="maximize" = find highest AUC)
      3. Run n_trials trials
      4. Print and save the best parameters
      5. Save all trial results to optuna_results.csv for inspection
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data — we only need the training set for tuning.
    train_df, _, _ = load_data()
    X, y, _ = get_feature_target_split(train_df)
    cat_features = get_cat_features(X)

    # Use a stratified 30% subsample for tuning. This is ~3x faster per trial
    # with negligible loss in param-ranking accuracy — Optuna only needs to
    # distinguish good params from bad, not compute a final model.
    from sklearn.model_selection import train_test_split
    X, _, y, _ = train_test_split(
        X, y, train_size=0.30, stratify=y, random_state=42
    )
    print(f"Tuning on {len(X):,} rows (30% stratified subsample of full training set).")

    print(f"\nStarting Optuna hyperparameter search: {n_trials} trials, {N_FOLDS}-fold CV")
    print("Each trial trains {n_folds} CatBoost models. Estimated time: ~45 min.\n".format(n_folds=N_FOLDS))

    # optuna.logging.set_verbosity controls how much Optuna prints.
    # WARNING suppresses the per-trial detail so only our print() calls show.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # MedianPruner: prunes a trial if, after the first fold, its AUC is in the
    # bottom half of all trials so far. Saves time on clearly bad params.
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)

    # TPESampler (default): Tree-structured Parzen Estimator — Bayesian method
    # that builds a probabilistic model of the objective and samples from
    # high-probability-of-improvement regions.
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=pruner,
    )

    # lambda wraps the objective so we can pass extra arguments (X, y, cat_features)
    # while Optuna only expects a single `trial` argument.
    study.optimize(
        lambda trial: objective(trial, X, y, cat_features),
        n_trials=n_trials,
        show_progress_bar=True,  # shows a tqdm progress bar in the terminal
    )

    # --- Results ---
    best_params = study.best_params
    best_score = study.best_value

    print(f"\n{'='*60}")
    print(f"Best CV AUC: {best_score:.6f}")
    print(f"Best hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")

    # Save best params to JSON so submission.py / train_baseline.py can load
    # them programmatically, or you can copy them manually.
    params_path = os.path.join(RESULTS_DIR, "best_params.json")
    with open(params_path, "w") as f:
        json.dump({"best_cv_auc": best_score, **best_params}, f, indent=2)
    print(f"\nBest parameters saved to: {params_path}")

    # Save all trials to CSV so you can inspect which combinations worked.
    trials_df = study.trials_dataframe()
    trials_path = os.path.join(RESULTS_DIR, "optuna_results.csv")
    trials_df.to_csv(trials_path, index=False)
    print(f"All trial results saved to: {trials_path}")

    # Print the exact lines to paste into train_baseline.py / submission.py
    print(f"\n--- Paste these into CatBoostClassifier() ---")
    print(f"    iterations={best_params['iterations']},")
    print(f"    learning_rate={best_params['learning_rate']:.5f},")
    print(f"    depth={best_params['depth']},")
    print(f"    l2_leaf_reg={best_params['l2_leaf_reg']:.4f},")
    print(f"    border_count={best_params['border_count']},")
    print(f"    bagging_temperature={best_params['bagging_temperature']:.4f},")
    print(f"---------------------------------------------\n")

    return best_params
