# Patient No-Show Prediction — Kaggle Competition

**Final Placement: 🥈 2nd Place (Private Leaderboard)**  
**Public ROC-AUC: 0.78271 · Private ROC-AUC: 0.78270**  
**45 submissions over the competition period**

---

## Competition Overview

Predict whether a patient will miss their scheduled medical appointment (*no-show*) using appointment metadata and patient history. The evaluation metric is **ROC-AUC**.

| | |
|---|---|
| Dataset | 168,982 training rows · ~20 categorical features |
| Target | `NO_SHOW_FLG` (binary: 1 = no-show) |
| Class balance | ~5% no-show (imbalanced) |
| Metric | ROC-AUC |

---

## Final Pipeline

```
Raw Data
    │
    ▼
Feature Engineering (frequency encoding, interaction features)
    │
    ▼
Train Base Models — 10 seeds × 5 folds each (50 models per algorithm)
  ├── CatBoost  (native categorical encoding)
  ├── LightGBM  (ordinal + target encoding variants)
  └── XGBoost   (ordinal encoding)
    │
    ▼
Generate OOF Predictions
    │
    ▼
Optuna Hyperparameter Search (CatBoost + LightGBM)
    │
    ▼
Pseudo-Labeling — Round 1 (~95k high-confidence test rows added)
    │
    ▼
Pseudo-Labeling — Round 2 (~119k rows, stricter thresholds)
    │
    ▼
Greedy Ensemble Selection (hill-climbing over OOF AUC)
    │
    ▼
Rank Averaging (normalises probability scales across models)
    │
    ▼
Final Submission  →  2nd Place  (0.78270 private AUC)
```

---

## Key Techniques

- **CatBoost baseline** — all 20 features are categorical strings; CatBoost's ordered target encoding was the single strongest baseline (CV AUC 0.7736)
- **50-model ensembling** — 10 random seeds × 5-fold CV reduces variance significantly vs a single-seed model
- **Optuna hyperparameter optimisation** — TPE sampler over depth / iterations / learning rate / l2\_leaf\_reg / bagging\_temperature
- **Pseudo-labeling (2 rounds)** — high-confidence test predictions added as extra training rows; boosted LB from 0.78218 → 0.78250+
- **Greedy ensemble selection** — hill-climbing over OOF AUC to find the optimal subset and weights across all saved model arrays
- **Rank averaging** — converts each model's raw probabilities to percentile ranks before averaging, neutralising calibration differences between CatBoost / LightGBM / XGBoost

---

## Experiment Timeline

| Version | Technique | Public AUC |
|---|---|---|
| v1–v3 | CatBoost baseline, submission tooling | 0.78056 |
| v4 | 3-model stacking (CB + LGBM + XGB) | 0.78066 |
| v8 | CB + LGBM blend, 10-seed × 5-fold | 0.78162 |
| v10 | 50-model CB + LGBM (lr=0.01, iter=3000) | **0.78218** |
| v11 | Target-encoded LGBM + CB blend | 0.78205 |
| v14 | XGBoost added to ensemble | 0.78222 |
| v15 | Rank averaging sweep | 0.78230 |
| v20 | Greedy ensemble (multi-model) | 0.78245 |
| v22 | Greedy + pseudo-labels round 1 (~95k rows) | 0.78250 |
| v23 | Fine-tuned greedy weights on v22 | 0.78260 |
| v26 | Rank(v23, v20) | 0.78268 |
| v29 | Pseudo-labels round 2 (~119k rows) | 0.78269 |
| **v31** | **Rank(v30, v20)** | **0.78271** ← final |

---

## Model Comparison (5-Fold CV on Training Data)

| Model | CV ROC-AUC | Notes |
|---|---|---|
| Logistic Regression | 0.7253 | Baseline reference |
| Random Forest | 0.7507 | |
| XGBoost | 0.7680 | |
| LightGBM | 0.7714 | |
| **CatBoost** | **0.7736** | Best single model — native categorical encoding |
| CatBoost (Optuna-tuned) | 0.7744 | Best params: depth=7, lr=0.034, l2=7.49 |

---

## Repository Structure

```
patient-no-show-kaggle/
│
├── README.md
│
├── src/                          ← all source code
│   ├── main.py                   ← CLI entry point (controls the whole pipeline)
│   ├── data_utils.py             ← data loading, feature splits, encoding helpers
│   ├── train_baseline.py         ← CatBoost 5-fold CV baseline
│   ├── tune.py                   ← Optuna search for CatBoost
│   ├── tune_lgbm.py              ← Optuna search for LightGBM
│   ├── blend.py                  ← CB + LGBM blend (10-seed × 5-fold)
│   ├── blend_te_lgbm.py          ← target-encoded LGBM variant
│   ├── stack.py                  ← 3-model stacking with Logistic Regression meta-model
│   └── submission.py             ← generates final Kaggle submission CSV
│
├── experiments/                  ← per-version experiment logs
│   ├── v10_50model_ensemble.md
│   ├── v20_greedy_ensemble.md
│   ├── v22_pseudo_label_r1.md
│   ├── v26_rank_blend.md
│   ├── v29_pseudo_label_r2.md
│   └── v31_final_submission.md
│
├── outputs/
│   ├── feature_importance.csv    ← top features by CatBoost gain
│   ├── model_results.csv         ← per-fold AUC for all models
│   └── leaderboard_progression/  ← score progression charts
│
├── slides/
│   └── competition_presentation/ ← 14-chart visual walkthrough of the pipeline
│
├── requirements.txt
└── .gitignore
```

> **Data not included** — raw competition data is not redistributed per Kaggle rules. Download `train.csv`, `test.csv`, and `metaData.csv` from the competition page and place them in the project root.

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Train CatBoost baseline (5-fold CV)
python src/main.py --baseline

# Run Optuna hyperparameter search (100 trials)
python src/main.py --tune --trials 100

# Run 50-model CB+LGBM blend and generate submission
python src/main.py --blend --output v10.csv

# Run full stacking ensemble
python src/main.py --stack --output stack.csv
```

---

## Key Findings

1. **Pseudo-labeling was the single biggest LB jump** (+0.00032 from v10 → v22). Threshold tuning mattered — too aggressive hurt diversity.
2. **Rank averaging consistently outperformed probability averaging** when blending models with different calibration scales (CatBoost vs LightGBM vs XGBoost).
3. **Optuna-tuned LGBM underperformed defaults** on the LB despite better CV scores — a reminder that CV alone doesn't guarantee LB improvement on categorical-heavy datasets.
4. **Greedy ensemble selection found weights that no hand-tuned blend matched.** The winning combination kept CatBoost-heavy and used LightGBM only where it added OOF diversity.
5. **The final 0.00015 margin** came down to rank blending v30 (pseudo-label r2) with v20 (greedy multi-model) — neither alone scored as high.

---

## Presentation

A 14-chart visual walkthrough of the full pipeline is available in `/slides/competition_presentation/`.

Charts cover: dataset overview · model comparison · Optuna convergence · feature importance · pseudo-labeling workflow · score progression · ensemble diagram · rank vs probability blending · leaderboard result · full submission history.

---

## Topics

`machine-learning` `kaggle` `catboost` `lightgbm` `xgboost` `python` `data-science` `ensemble-methods` `pseudo-labeling` `optuna` `hyperparameter-optimization` `feature-engineering`
