# Patient No-Show Prediction — Kaggle Competition

**Final Placement: 🥈 2nd Place (Private Leaderboard)**  
**Public ROC-AUC: 0.78271 · Private ROC-AUC: 0.78270**

---

## Highlights

- 🥈 2nd place overall on the private leaderboard
- Final private ROC-AUC: **0.78270** (leader: 0.78285 — margin of 0.00015)
- 50-model CatBoost ensemble (10 random seeds × 5-fold CV) as the backbone
- Two rounds of pseudo-labeling, augmenting the training set by ~214k high-confidence rows
- Greedy ensemble selection (Caruana et al. 2004) over all saved OOF arrays
- Final solution: rank blend of a pseudo-labeled greedy ensemble and a base greedy ensemble

---

## Core Contributions

This project focused on **systematic experimentation and ensemble optimisation for categorical tabular data**. Rather than optimising a single model, the workflow explored how ensemble construction strategy, training data augmentation, and prediction blending interact.

Key findings:

- CatBoost's **ordered target encoding** captured most interaction and frequency signals internally, making explicit feature engineering redundant — and sometimes harmful
- **Pseudo-labeling** produced the single largest leaderboard gain of the competition (+0.00027 AUC), provided thresholds were strict enough to avoid noisy labels
- **Greedy ensemble selection** consistently outperformed manually tuned blend weights, suggesting that exhaustive search over saved OOF arrays is a more reliable strategy than human intuition
- **Rank averaging** improved robustness when blending models trained on different data distributions (original vs. pseudo-labeled), by removing calibration differences before combining predictions

---

## Methodology

The workflow followed four major stages:

**1. Baseline benchmarking** — Five model families (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost) were compared under identical 5-fold CV conditions to identify the strongest foundation. CatBoost dominated due to its native handling of high-cardinality categorical features.

**2. Hyperparameter optimisation** — Optuna was used to search the CatBoost and LightGBM hyperparameter spaces (100 trials each). Notably, tuned LightGBM underperformed default parameters, suggesting that the 5-fold CV signal was too noisy to guide LGBM tuning reliably.

**3. Pseudo-label augmentation** — High-confidence test predictions (positive: prob > 0.60, negative: prob < 0.02) were added to the training set and models were retrained. Two rounds of this were run. Strict thresholds were essential; loosening them degraded performance by introducing noisy labels.

**4. Ensemble selection and rank blending** — Greedy ensemble selection (hill-climbing over OOF ROC-AUC, with replacement) was used to find the optimal weighted combination across all saved model arrays. The final submission rank-averaged two independently trained greedy ensembles — one on original data, one on pseudo-labeled data — to balance stability and diversity.

Cross-validation was used throughout all experiments to reduce leaderboard overfitting and ensure stable generalisation.

---

## Competition Overview

Predict whether a patient will miss their scheduled medical appointment (*no-show*) using appointment metadata and patient history. The evaluation metric is **ROC-AUC**.

| | |
|---|---|
| Dataset | 168,982 training rows · 20 categorical features |
| Target | `NO_SHOW_FLG` (binary: 1 = no-show) |
| Class balance | ~5% no-show (strongly imbalanced) |
| Metric | ROC-AUC |

---

## Final Pipeline

```
Raw Data (168,982 rows · 20 categorical features)
    │
    ▼
Train CatBoost + LightGBM
  10 random seeds × 5-fold CV = 50 models per algorithm
    │
    ▼
Generate OOF Predictions for all model variants
    │
    ▼
Pseudo-Labeling Round 1
  Select high-confidence test rows:
    398 predicted no-shows  (prob > 0.60)
    94,639 predicted show-ups (prob < 0.02)
  → Augment training set to ~264k rows
    │
    ▼
Retrain CatBoost + LightGBM on augmented data (50 models each)
    │
    ▼
Pseudo-Labeling Round 2
  Select high-confidence test rows from updated model:
    461 predicted no-shows  (prob > 0.60)
    119,128 predicted show-ups (prob < 0.02)
  → Augment training set to ~288k rows
    │
    ▼
Retrain CatBoost + LightGBM on augmented data (50 models each)
    │
    ▼
Greedy Ensemble Selection
  Hill-climbing over OOF ROC-AUC across all saved model arrays
  Two independent runs:
    Base greedy (v20)       OOF AUC: 0.779003
    PL2 greedy (v30)        OOF AUC: 0.780235
    │
    ▼
Rank Average (v30, v20)
    │
    ▼
Final Submission — 3rd Public (0.78271) → 2nd Private (0.78270)
```

---

## Ablation Results

Each technique below is measured as the AUC change from its most comparable prior submission.

| Technique | AUC Change | Notes |
|---|---|---|
| Multi-seed ensembling (v1 → v2) | +0.00110 | Averaging 10 seeds substantially reduces prediction variance |
| CB + LGBM blend (v2 → v8) | +0.00048 | Adding a diverse model family improves ensemble calibration |
| Slower CatBoost convergence (v8 → v10) | +0.00004 | Lower learning rate, more iterations; small but consistent |
| Pseudo-labeling Round 1 (v10 → v18) | **+0.00027** | Largest single gain — strict thresholds were key |
| Greedy ensemble selection (v18 → v19) | +0.00004 | Systematic search beats manual weight tuning |
| Multi-start greedy (v19 → v20) | +0.00007 | Multiple random restarts reduce sensitivity to greedy path |
| Pseudo-labeling Round 2 + rank blend (v20 → v31) | +0.00015 | Final submission combines both greedy runs via rank averaging |
| Feature engineering (7 interactions) | −0.00029 | CatBoost's encoding already captures interaction signals |
| Frequency encoding | −0.00167 | Redundant with CatBoost's internal encoding |

---

## Model Comparison (5-Fold CV)

| Model | CV ROC-AUC | Notes |
|---|---|---|
| Logistic Regression | 0.7253 | Linear baseline |
| Random Forest | 0.7507 | |
| XGBoost | 0.7680 | |
| LightGBM | 0.7714 | |
| **CatBoost** | **0.7736** | Best single model — native ordered target encoding |
| CatBoost (Optuna-tuned) | 0.7744 | Best params: depth=7, lr=0.034, iter=757, l2=7.49 |

CatBoost's advantage stems from its ordered target statistics, which encode categorical features without target leakage. On a dataset with 20 high-cardinality categorical columns, this was a decisive structural advantage over tree-based models that require explicit encoding.

---

## Key Techniques

- **Multi-seed ensembling** — training 10 random seeds per fold (50 models total per algorithm) reduces prediction variance substantially. Each seed produces a slightly different model; averaging smooths out noise without requiring additional data.

- **Pseudo-labeling** — high-confidence test-set predictions (prob > 0.60 for the positive class, prob < 0.02 for the negative) are treated as additional labelled training data. The key constraint is threshold strictness: looser thresholds introduce too many ambiguous labels and degrade performance.

- **Greedy ensemble selection** — models are added to the ensemble one at a time, each time keeping the addition only if it improves OOF ROC-AUC (Caruana et al., 2004). This is more reliable than hand-tuning blend weights because it directly optimises the OOF metric over a combinatorially large model pool.

- **Rank averaging** — before blending, each model's raw probability outputs are replaced by their percentile ranks. This removes calibration scale differences between models trained on different data distributions (e.g., original vs. pseudo-labeled training sets), making the final average more stable.

---

## Experiment Timeline

| Experiment | Technique | Public AUC |
|---|---|---|
| Baseline | CatBoost 5-fold CV | 0.78056 |
| Multi-seed ensemble | CatBoost, 10 seeds | 0.78166 |
| Dual-model blend | CatBoost + LightGBM, 10-seed × 5-fold | 0.78214 |
| Slower convergence | CB lr=0.01, iter=3000 | 0.78218 |
| Pseudo-labeling round 1 | +95k high-confidence rows | 0.78245 |
| Greedy ensemble | Hill-climbing over OOF AUC | 0.78249 |
| Multi-start greedy | Multiple random restarts | 0.78256 |
| Rank blend | rank(greedy-pl2) + rank(base greedy) | 0.78268 |
| **Final submission** | **Rank blend: v30 + v20** | **0.78271** |

---

## What Didn't Work

The following directions were explored and abandoned. Each failed for a structurally interesting reason.

| Technique | Public AUC | Finding |
|---|---|---|
| Interaction feature engineering (7 cross-features) | 0.78189 | CatBoost's ordered target encoding captures interaction signals internally; explicit features added noise |
| Frequency encoding as primary features | 0.78051 | Frequency information was already embedded in CatBoost's encoding; the explicit version conflicted rather than complemented |
| Optuna-tuned LightGBM | 0.78199 | Tuned parameters overfit to 5-fold CV; default LGBM parameters generalised better to the leaderboard |
| Target-encoded LightGBM blend | 0.78203 | Duplicated CatBoost's existing signal; no ensemble diversity gain |
| 3-model stacking (CB + LGBM + XGB → Logistic Regression meta) | 0.77991 | The meta-model overfits on OOF predictions with a small inner fold; direct blending was more robust |

One consistent pattern across failures: approaches that added information CatBoost was already capturing internally tended to hurt rather than help.

---

## Insights

1. **Pseudo-labeling required strict thresholds to be effective.** The +0.00027 gain came from keeping only the highest-confidence test predictions. When thresholds were loosened to add more rows, performance declined — the noise from ambiguous labels outweighed the benefit of additional data.

2. **Greedy ensemble selection consistently beat manual weight tuning.** Multi-start greedy (0.78256) outperformed every manually weighted blend at the same model pool. Exhaustive OOF-guided search is more reliable than human intuition for weight optimisation.

3. **Feature engineering consistently hurt.** Interaction features and frequency encoding both fell below the pre-engineering baseline. This was the most surprising result: CatBoost's internal encoding was already capturing most of the signal these features were designed to add.

4. **Hyperparameter tuning was not always beneficial.** Optuna-tuned LightGBM underperformed default parameters on the leaderboard. The 5-fold CV signal was not reliable enough to distinguish between similarly-performing hyperparameter configurations, and tuned parameters overfit to that signal.

5. **The private leaderboard confirmed the ensemble was not overfit to the public test split.** The final submission ranked 3rd on the public leaderboard (0.78271) but rose to 2nd on the private (0.78270). The 2nd-place public competitor dropped to 3rd on the private (0.78283 → 0.78253), while our model held. The private margin to 1st place (0.78285) was 0.00015.

---

## Lessons Learned

The strongest improvements did not come from model complexity or feature engineering — they came from:

- **reducing variance** through seed averaging across a fixed fold structure
- **leveraging unlabelled test data carefully** via pseudo-labeling with strict confidence thresholds
- **systematic ensemble search** rather than manual weight tuning
- **stabilising predictions across distribution shifts** using rank normalisation before blending

A recurring observation was that this dataset rewarded *ensemble diversity* and *variance reduction* far more than individual model performance. The delta between a single CatBoost model (0.7736 CV AUC) and the final ensemble (0.78270 private AUC) was substantially larger than the delta from any single modelling improvement.

The competition also highlighted how quickly leaderboard overfitting can occur when iterating directly against public LB feedback. Cross-validation remained the more reliable guide throughout.

---

## Repository Structure

```
osf-patient-no-show/
│
├── README.md
├── requirements.txt
├── .gitignore
├── generate_slides_images_v2.py   ← generates the 14 chart PNGs in slide_images/
│
├── experiments/                   ← per-experiment write-ups with methodology notes
│   ├── v10_50model_ensemble.md
│   ├── v20_greedy_ensemble.md
│   ├── v22_pseudo_label_r1.md
│   ├── v26_rank_blend.md
│   ├── v29_pseudo_label_r2.md
│   └── v31_final_submission.md
│
└── osf_no_show_project/
    ├── main.py                    ← CLI entry point
    ├── data_utils.py              ← data loading, feature splits, encoding helpers
    ├── train_baseline.py          ← CatBoost 5-fold CV baseline
    ├── tune.py                    ← Optuna hyperparameter search (CatBoost)
    ├── tune_lgbm.py               ← Optuna hyperparameter search (LightGBM)
    ├── blend.py                   ← CatBoost + LightGBM blend (10-seed × 5-fold)
    ├── blend_te_lgbm.py           ← target-encoded LightGBM variant
    ├── stack.py                   ← 3-model stacking (CB + LGBM + XGB → LogReg meta)
    ├── submission.py              ← submission CSV generator
    ├── outputs/
    │   ├── model_results.csv      ← per-fold AUC for all model families
    │   ├── optuna_results.csv     ← all Optuna trial scores
    │   └── feature_importance/feature_importance.csv
    ├── results/                   ← saved OOF/test arrays and best hyperparameters
    │   ├── best_params.json       ← Optuna best CatBoost params (CV AUC 0.7744)
    │   └── best_lgbm_params.json  ← Optuna best LightGBM params
    ├── slide_images/              ← 14 presentation chart PNGs
    └── submissions/               ← all submission CSVs
```

> **Data not included** — competition data is not redistributed per Kaggle rules. Download `train.csv`, `test.csv`, and `metaData.csv` from the competition page and place them in the repo root.

---

## Quickstart

```bash
pip install -r requirements.txt

# CatBoost 5-fold CV baseline
python osf_no_show_project/main.py --baseline

# Optuna hyperparameter search (100 trials)
python osf_no_show_project/main.py --tune --trials 100

# 50-model CB+LGBM blend
python osf_no_show_project/main.py --blend --output submission.csv

# 3-model stacking ensemble
python osf_no_show_project/main.py --stack --output submission_stack.csv
```

---

## Presentation

A 14-chart visual walkthrough of the full pipeline is in `osf_no_show_project/slide_images/`. Charts cover dataset overview, model benchmarking, Optuna convergence, feature importance, pseudo-labeling workflow, score progression, ensemble architecture, rank vs. probability blending, leaderboard result, and full submission history.

---

## Topics

`machine-learning` `kaggle` `catboost` `lightgbm` `python` `data-science` `ensemble-methods` `pseudo-labeling` `optuna` `hyperparameter-optimization`
