# =============================================================================
# train_baseline.py
# -----------------------------------------------------------------------------
# Trains the first real model: a CatBoost classifier evaluated with 5-fold
# Stratified Cross-Validation.
#
# Why CatBoost as the baseline?
#   - All 20 features in this dataset are categorical strings.
#   - CatBoost handles categorical features natively using ordered target
#     encoding, which is more accurate than OrdinalEncoder or one-hot encoding.
#   - No manual preprocessing or encoding pipeline needed.
#
# Why Stratified K-Fold?
#   - The target is heavily imbalanced (~5% no-shows).
#   - StratifiedKFold ensures every fold has roughly the same class ratio,
#     preventing a fold from having too few positive examples to evaluate on.
#
# Outputs saved to results/:\
#   oof_predictions.csv    — Out-of-Fold predictions for every training row
#   feature_importance.csv — Which features the last fold's model relied on most
# =============================================================================

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

from data_utils import load_data, get_feature_target_split, get_cat_features

# Where to write result files. os.makedirs will create this folder if needed.
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N_FOLDS = 5  # number of cross-validation folds; 5 is a standard choice


def train_baseline():
    # Step 1: Load data and split into features + target.
    # We discard test_df and meta_df here because baseline training only
    # needs the labelled training set.
    train_df, _, _ = load_data()
    X, y, ids = get_feature_target_split(train_df)

    # Detect which columns are categorical strings so CatBoost can handle them.
    cat_features = get_cat_features(X)

    # StratifiedKFold keeps the no-show ratio consistent across all folds.
    # shuffle=True randomises which rows go into which fold.
    # random_state=42 makes the fold split reproducible.
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # oof_preds stores the model's predicted probability for each training row
    # when that row was in the VALIDATION set (i.e. the model had never seen it).
    # This gives an unbiased estimate of real-world performance.
    oof_preds = np.zeros(len(X))
    fold_scores = []   # ROC-AUC for each individual fold
    last_model = None  # kept to extract feature importances after the loop

    print(f"\nTraining CatBoost baseline ({N_FOLDS}-fold StratifiedKFold)...")

    # Step 2: Cross-validation loop.
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        # Split this fold's rows into train and validation sets.
        X_train = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # CatBoost requires categorical columns to be strings, not NaN.
        # Replace any missing values with the literal string "Missing" so
        # CatBoost treats them as a valid category rather than throwing an error.
        for col in cat_features:
            X_train[col] = X_train[col].fillna("Missing")
            X_val[col] = X_val[col].fillna("Missing")

        # Step 3: Build and train the CatBoost model.
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
        )
        # eval_set lets CatBoost evaluate the model on the validation set after
        # each iteration. early_stopping_rounds=50 stops training if the AUC on
        # the validation set hasn't improved for 50 consecutive trees — this
        # prevents overfitting and saves time.
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

        # Step 4: Predict probabilities for the validation rows.
        # predict_proba returns [[prob_class_0, prob_class_1], ...]
        # We take column [:, 1] which is the probability of no-show (class 1).
        val_preds = model.predict_proba(X_val)[:, 1]

        # Store these predictions at the correct positions in the OOF array.
        oof_preds[val_idx] = val_preds

        # ROC-AUC measures discrimination: probability that the model ranks a
        # random no-show higher than a random show-up. 0.5 = random, 1.0 = perfect.
        score = roc_auc_score(y_val, val_preds)
        fold_scores.append(score)
        last_model = model
        print(f"  Fold {fold}: ROC-AUC = {score:.4f}")

    # Step 5: Summarise results across all folds.
    mean_auc = np.mean(fold_scores)
    std_auc = np.std(fold_scores)
    # Overall OOF AUC uses every training row exactly once as a validation row,
    # giving a single stable estimate of generalisation performance.
    overall_oof_auc = roc_auc_score(y, oof_preds)
    print(f"\nCatBoost Baseline")
    print(f"  Mean CV AUC : {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  Overall OOF AUC: {overall_oof_auc:.4f}")

    # Step 6: Save outputs to the results/ folder.
    os.makedirs(RESULTS_DIR, exist_ok=True)  # create folder if it doesn't exist

    # Save OOF predictions — useful for stacking or post-analysis.
    oof_df = pd.DataFrame({"ID": ids.values, "oof_pred": oof_preds, "y_true": y.values})
    oof_df.to_csv(os.path.join(RESULTS_DIR, "oof_predictions.csv"), index=False)
    print("  OOF predictions saved to results/oof_predictions.csv")

    if last_model is not None:
        # Feature importance shows which columns contributed most to the model's
        # predictions. Higher value = feature was used more to split trees.
        importance_df = pd.DataFrame({
            "feature": X.columns,
            "importance": last_model.get_feature_importance(),
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        importance_df.to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)
        print("  Feature importances saved to results/feature_importance.csv")

    return mean_auc, std_auc


if __name__ == "__main__":
    train_baseline()
