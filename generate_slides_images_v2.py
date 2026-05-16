"""
Generate all presentation images for the OSF No-Show Prediction project.
Outputs one PNG per slide into  osf_no_show_project/slide_images/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score, roc_curve

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT   = os.path.dirname(os.path.abspath(__file__))
RES    = os.path.join(ROOT, "osf_no_show_project", "results")
OUT    = os.path.join(ROOT, "osf_no_show_project", "slide_images")
os.makedirs(OUT, exist_ok=True)

TRAIN_CSV = os.path.join(ROOT, "train.csv")

# ── Colour palette ───────────────────────────────────────────────────────────
BLUE   = "#2563EB"
DBLUE  = "#1E3A8A"
LGRAY  = "#F1F5F9"
DGRAY  = "#334155"
GREEN  = "#16A34A"
RED    = "#DC2626"
AMBER  = "#D97706"
PURPLE = "#7C3AED"
TEAL   = "#0D9488"

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "font.size":         12,
})

# =============================================================================
# SLIDE 2a – Class Distribution (pie + bar)
# =============================================================================
def slide2_class_distribution():
    df = pd.read_csv(TRAIN_CSV)
    counts = df["NO_SHOW_FLG"].value_counts().sort_index()
    total  = counts.sum()
    labels = ["Showed Up (0)", "No-Show (1)"]
    sizes  = [counts[0], counts[1]]
    pcts   = [s / total * 100 for s in sizes]
    colors = [BLUE, RED]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")

    wedge_props = dict(linewidth=2, edgecolor="white")
    wedges, texts, autotexts = axes[0].pie(
        sizes, labels=None, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=wedge_props, pctdistance=0.65,
        textprops={"fontsize": 15, "fontweight": "bold"},
    )
    for at in autotexts:
        at.set_color("white")
    axes[0].set_title("Target Class Distribution\n(168,982 training rows)",
                      fontsize=14, fontweight="bold", pad=15)
    axes[0].legend(wedges, [f"{l}  ({s:,}  |  {p:.1f}%)"
                             for l, s, p in zip(labels, sizes, pcts)],
                   loc="lower center", fontsize=11,
                   bbox_to_anchor=(0.5, -0.12), frameon=False)

    bars = axes[1].bar(labels, sizes, color=colors, width=0.4,
                       edgecolor="white", linewidth=1.5)
    for bar, s, p in zip(bars, sizes, pcts):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1500,
                     f"{s:,}\n({p:.1f}%)",
                     ha="center", va="bottom", fontsize=13, fontweight="bold")
    axes[1].set_title("Class Count Comparison", fontsize=14, fontweight="bold", pad=10)
    axes[1].set_ylabel("Number of Rows", fontsize=12)
    axes[1].set_ylim(0, max(sizes) * 1.15)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    axes[1].tick_params(axis="x", labelsize=12)

    fig.suptitle("Dataset Overview — OSF No-Show Prediction",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT, "slide2_class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# SLIDE 2b – Dataset Feature Overview Table
# =============================================================================
def slide2_dataset_table():
    data = {
        "Metric": [
            "Training rows", "Test rows", "Total features",
            "Feature type", "Target column", "Metric",
            "Class ratio (no-show)", "CV strategy",
        ],
        "Value": [
            "168,982", "72,006", "20",
            "Categorical (pre-bucketed ranges)", "NO_SHOW_FLG (0 / 1)",
            "ROC-AUC",
            "~5 %", "5-Fold Stratified K-Fold",
        ],
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")

    tbl = ax.table(cellText=df.values, colLabels=df.columns,
                   cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl.scale(1, 2.0)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if r == 0:
            cell.set_facecolor(DBLUE)
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor(LGRAY)
        else:
            cell.set_facecolor("white")

    ax.set_title("Dataset at a Glance", fontsize=16, fontweight="bold",
                 pad=16, color=DGRAY)
    plt.tight_layout()
    path = os.path.join(OUT, "slide2_dataset_table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# SLIDE 3 – Model Comparison Bar Chart
# =============================================================================
def slide3_model_comparison():
    model_results = pd.read_csv(os.path.join(RES, "model_results.csv"))
    models = model_results["model"].tolist()
    means  = model_results["mean_auc"].tolist()
    stds   = model_results["std_auc"].tolist()

    colors = [LGRAY, LGRAY, LGRAY, LGRAY, BLUE]
    edge   = [DGRAY] * 4 + [DBLUE]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(models, means, color=colors, edgecolor=edge,
                  linewidth=1.5, width=0.55,
                  yerr=stds, capsize=6,
                  error_kw=dict(elinewidth=1.5, ecolor=DGRAY))

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.0005,
                f"{m:.4f}", ha="center", va="bottom",
                fontsize=13, fontweight="bold",
                color=BLUE if bar.get_facecolor()[0] < 0.5 else DGRAY)

    ax.set_ylim(0.68, 0.795)
    ax.set_ylabel("5-Fold CV ROC-AUC", fontsize=13)
    ax.set_title("Baseline Model Comparison — 5-Fold CV ROC-AUC",
                 fontsize=15, fontweight="bold", pad=12)
    ax.axhline(0.78, color=GREEN, linestyle="--", linewidth=1.5, alpha=0.6,
               label="Target threshold (0.78)")
    ax.legend(fontsize=11, frameon=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))
    ax.tick_params(axis="x", labelsize=12)

    best_idx = means.index(max(means))
    ax.annotate("Best baseline\n← CatBoost",
                xy=(best_idx, max(means) + stds[best_idx] + 0.001),
                xytext=(best_idx - 1.1, max(means) + 0.006),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.8),
                fontsize=11, color=RED, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUT, "slide3_model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# SLIDE 3b – CatBoost 5-Fold CV Detail
# =============================================================================
def slide3_catboost_folds():
    row = pd.read_csv(os.path.join(RES, "model_results.csv"))
    row = row[row["model"] == "CatBoost"].iloc[0]
    folds = [row["fold_1"], row["fold_2"], row["fold_3"], row["fold_4"], row["fold_5"]]
    mean_val = row["mean_auc"]

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = list(range(1, 6))
    bars = ax.bar(xs, folds, color=[BLUE] * 5, edgecolor=DBLUE,
                  linewidth=1.5, width=0.5)
    for bar, v in zip(bars, folds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
                f"{v:.4f}", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=DBLUE)

    ax.axhline(mean_val, color=RED, linestyle="--", linewidth=2,
               label=f"Mean AUC = {mean_val:.4f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Fold {i}" for i in xs], fontsize=12)
    ax.set_ylim(0.755, 0.786)
    ax.set_ylabel("ROC-AUC", fontsize=13)
    ax.set_title("CatBoost — 5-Fold CV Per-Fold Scores",
                 fontsize=15, fontweight="bold", pad=12)
    ax.legend(fontsize=12, frameon=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))

    plt.tight_layout()
    path = os.path.join(OUT, "slide3_catboost_folds.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# SLIDE 3c – Optuna Tuning Convergence
# =============================================================================
def slide3_optuna_convergence():
    opt = pd.read_csv(os.path.join(RES, "optuna_results.csv"))
    opt = opt[opt["state"] == "COMPLETE"].copy()
    opt["best_so_far"] = opt["value"].cummax()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(opt["number"], opt["value"], color=BLUE, alpha=0.45, s=35,
               label="Trial AUC", zorder=2)
    ax.plot(opt["number"], opt["best_so_far"], color=RED, lw=2.5,
            label="Best AUC so far", zorder=3)

    best_row = opt.loc[opt["value"].idxmax()]
    ax.annotate(f"Best: {best_row['value']:.5f}\n(trial {int(best_row['number'])})",
                xy=(best_row["number"], best_row["value"]),
                xytext=(best_row["number"] + 5, best_row["value"] - 0.003),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
                fontsize=11, color=RED, fontweight="bold")

    ax.set_xlabel("Optuna Trial #", fontsize=13)
    ax.set_ylabel("3-Fold CV ROC-AUC", fontsize=13)
    ax.set_title("Optuna Hyperparameter Search — CatBoost (100 trials)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=11, frameon=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.4f}"))

    plt.tight_layout()
    path = os.path.join(OUT, "slide3_optuna_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# SLIDE 4 – Feature Importance Bar Chart
# =============================================================================
def slide4_feature_importance():
    fi = pd.read_csv(os.path.join(RES, "feature_importance.csv"))
    fi = fi.sort_values("importance", ascending=True)

    colors = [GREEN if i >= len(fi) - 3 else BLUE for i in range(len(fi))]

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(fi["feature"], fi["importance"],
                   color=colors, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, fi["importance"]):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", ha="left", fontsize=10.5)

    ax.set_xlabel("Feature Importance (%)", fontsize=13)
    ax.set_title("CatBoost Feature Importance — Top 20 Features",
                 fontsize=15, fontweight="bold", pad=12)
    ax.set_xlim(0, fi["importance"].max() * 1.18)

    top3 = mpatches.Patch(color=GREEN, label="Top 3 features")
    rest = mpatches.Patch(color=BLUE,  label="Remaining features")
    ax.legend(handles=[top3, rest], fontsize=11, frameon=False, loc="lower right")

    plt.tight_layout()
    path = os.path.join(OUT, "slide4_feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# SLIDE 4b – What Didn't Work Table  (actual LB scores)
# =============================================================================
def slide4_what_didnt_work():
    # Baseline for comparison: v10 = public LB 0.78218
    data = {
        "Technique": [
            "7 Interaction features\n(NoShowRate\u00d7DaysBetween etc.)",
            "Frequency encoding\n(count-proportion per category)",
            "LightGBM Optuna tuning\n(100 trials, best CV=0.7601)",
            "Target-encoded LGBM blend\n(TE with cv=5)",
            "XGBoost standalone\n(10-seed \u00d7 5-fold, OOF=0.7745)",
            "3-model Stacking\n(CB+LGBM+XGB \u2192 LogReg meta)",
        ],
        "Version": ["v16", "v17", "v9", "v11", "v24 (greedy)", "submission_stack"],
        "LB Score": ["0.78189", "0.78051", "0.78199", "0.78203", "0.78256", "0.77991"],
        "vs v10 (0.78218)": [
            "-0.00029 \u2193",
            "-0.00167 \u2193\u2193",
            "-0.00019 \u2193",
            "-0.00015 \u2193",
            "+0.00038 (greedy kept 2%)",
            "-0.00227 \u2193\u2193",
        ],
        "Why it didn't help": [
            "CatBoost already captures interactions\nvia ordered target encoding",
            "CatBoost's encoding already captures\ncategory frequency signal",
            "Tuned LGBM worse than defaults;\ncategorical data favours CB",
            "Target encoding duplicates CB's\ninternal encoding work",
            "OOF=0.7745 \u2014 weakest model;\nhigh correlation with CB (0.937+)",
            "Meta-model overfits on small OOF;\nweaker than simple blend",
        ],
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.axis("off")

    tbl = ax.table(cellText=df.values, colLabels=df.columns,
                   cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 2.6)

    col_widths = [0.20, 0.07, 0.08, 0.14, 0.35]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if c < len(col_widths):
            cell.set_width(col_widths[c])
        if r == 0:
            cell.set_facecolor(DBLUE)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            if c == 2:
                try:
                    val = float(cell.get_text().get_text())
                    cell.set_facecolor("#FEE2E2" if val < 0.78218 else "#DCFCE7")
                except ValueError:
                    pass
            elif r % 2 == 0:
                cell.set_facecolor("#FEF2F2")
            else:
                cell.set_facecolor("white")

    ax.set_title(
        "Experiments That Did NOT Help  (baseline v10 = 0.78218 public LB)",
        fontsize=13, fontweight="bold", pad=14, color=RED)
    plt.tight_layout()
    path = os.path.join(OUT, "slide4_what_didnt_work.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# SLIDE 5 – Pseudo-Labeling Diagram  (corrected: ~95k rows round 1, ~119k round 2)
# =============================================================================
def slide5_pseudo_label_diagram():
    fig, ax = plt.subplots(figsize=(13, 6.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.8)
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    def flow_box(x, y, w, h, color, label, sublabel="", fontsize=11):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                              facecolor=color, edgecolor=DGRAY, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + (0.22 if sublabel else 0),
                label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold",
                color="white" if color != LGRAY else DGRAY)
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.32, sublabel,
                    ha="center", va="center", fontsize=9,
                    color="white" if color != LGRAY else DGRAY)

    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.15",
                                   color=DGRAY, lw=2))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.22, label, ha="center", va="bottom",
                    fontsize=10, color=DGRAY, style="italic")

    # Flow diagram (top half)
    flow_box(0.3, 3.5, 2.4, 1.6, BLUE, "\u2460 Train Model",
             "168k training rows\n10 seeds \u00d7 5 folds CB+LGBM", fontsize=10)
    arrow(2.7, 4.3, 3.7, 4.3, "predict")
    flow_box(3.7, 3.5, 2.4, 1.6, TEAL, "\u2461 Predict Test Set",
             "72,006 test rows\n\u2192 probabilities 0\u20131", fontsize=10)
    arrow(6.1, 4.3, 7.1, 4.3, "filter")
    flow_box(7.1, 3.5, 2.6, 1.6, PURPLE, "\u2462 Filter High-Conf.",
             "Prob > 0.60  \u2192 NO-SHOW (1)\nProb < 0.02  \u2192 SHOW (0)", fontsize=10)
    arrow(9.7, 4.3, 10.7, 4.3, "append")
    flow_box(10.7, 3.5, 2.0, 1.6, GREEN, "\u2463 Retrain",
             "train + pseudo\n\u2192 new CB+LGBM", fontsize=10)

    # Round 1 info box
    r1_rect = FancyBboxPatch((0.3, 0.3), 6.0, 2.55,
                             boxstyle="round,pad=0.12", facecolor=LGRAY,
                             edgecolor=DGRAY, linewidth=1.5)
    ax.add_patch(r1_rect)
    ax.text(3.3, 2.73, "Round 1 \u2014 V18  (source: v10.csv predictions)",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color=DBLUE)
    ax.text(0.55, 2.45, "\u2022 398 high-confidence NO-SHOWs  (prob > 0.60)",
            fontsize=10.5, color=RED)
    ax.text(0.55, 2.02, "\u2022 94,639 high-confidence SHOW-UPs  (prob < 0.02)",
            fontsize=10.5, color=BLUE)
    ax.text(0.55, 1.59, "\u2022 Total pseudo rows:  +95,037  (168k \u2192 263k train rows)",
            fontsize=10.5, color=DGRAY, fontweight="bold")
    ax.text(0.55, 1.12, "\u2022 CB OOF on real rows: 0.779638 \u2192 blended OOF: 0.780060",
            fontsize=10, color=GREEN)
    ax.text(0.55, 0.65, "\u2022 Public LB jump:  0.78218 \u2192 0.78245  (+0.00027)",
            fontsize=10, color=GREEN, fontweight="bold")

    # Round 2 info box
    r2_rect = FancyBboxPatch((6.5, 0.3), 6.2, 2.55,
                             boxstyle="round,pad=0.12", facecolor=LGRAY,
                             edgecolor=DGRAY, linewidth=1.5)
    ax.add_patch(r2_rect)
    ax.text(9.6, 2.73, "Round 2 \u2014 V29/V30  (source: v23_fine_v22.csv)",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color=DBLUE)
    ax.text(6.75, 2.45, "\u2022 461 high-confidence NO-SHOWs  (prob > 0.60)",
            fontsize=10.5, color=RED)
    ax.text(6.75, 2.02, "\u2022 119,128 high-confidence SHOW-UPs  (prob < 0.02)",
            fontsize=10.5, color=BLUE)
    ax.text(6.75, 1.59, "\u2022 Total pseudo rows:  +119,589  (168k \u2192 287k train rows)",
            fontsize=10.5, color=DGRAY, fontweight="bold")
    ax.text(6.75, 1.12, "\u2022 Greedy pl2 OOF: 0.780235  |  public LB: 0.78254",
            fontsize=10, color=GREEN)
    ax.text(6.75, 0.65, "\u2022 After rank blend with v20:  final LB  0.78271  (2nd place)",
            fontsize=10, color=GREEN, fontweight="bold")

    ax.set_title("Pseudo-Labeling Strategy \u2014 Two Rounds of Test-Set Augmentation",
                 fontsize=14, fontweight="bold", pad=10, color=DBLUE)
    plt.tight_layout()
    path = os.path.join(OUT, "slide5_pseudo_labeling.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# SLIDE 6a – Score Progression (actual Kaggle public LB scores)
# =============================================================================
def slide6_score_progression():
    stages = [
        ("v1\nCatBoost\n5-fold",            0.78056),
        ("v2\nCatBoost\n5-fold+ensemble",    0.78166),
        ("v8\nCB+LGBM\nblend",               0.78214),
        ("v10\nCB lr=0.01\n3k iter",         0.78218),
        ("v18\nPseudo-\nlabeling (+95k)",    0.78245),
        ("v20\nGreedy\nmulti-start",         0.78256),
        ("v23\nFine\nweights",               0.78254),
        ("v26\nRank blend\nv23+v20",         0.78268),
        ("v31/v33\nFinal rank\nblend",       0.78271),
    ]
    labels = [s[0] for s in stages]
    scores = [s[1] for s in stages]
    phases = [BLUE, BLUE, TEAL, TEAL, PURPLE, PURPLE, PURPLE, GREEN, GREEN]

    fig, ax = plt.subplots(figsize=(14, 6))
    xs = list(range(len(stages)))

    ax.plot(xs, scores, color=DGRAY, linewidth=1.8, zorder=1, alpha=0.6)
    for x, y, c in zip(xs, scores, phases):
        ax.scatter(x, y, color=c, s=120, zorder=3, edgecolors="white", linewidths=1.8)
        va  = "bottom" if x % 2 == 0 else "top"
        off = 0.00025 if va == "bottom" else -0.00025
        ax.text(x, y + off, f"{y:.5f}", ha="center", va=va,
                fontsize=10, fontweight="bold", color=c)

    ax.axvspan(3.5, 4.5, alpha=0.10, color=PURPLE)
    ax.text(4.0, 0.78228, "Pseudo-label\nbreakthrough",
            ha="center", fontsize=9, color=PURPLE, style="italic")

    ax.annotate("Best: 0.78271\n(2nd place)",
                xy=(8, 0.78271), xytext=(7.0, 0.78278),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.8),
                fontsize=10, color=GREEN, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Kaggle Public LB  ROC-AUC", fontsize=13)
    ax.set_ylim(0.7793, 0.7832)
    ax.set_title("Score Progression \u2014 Actual Kaggle Public Leaderboard Scores",
                 fontsize=14, fontweight="bold", pad=12)

    p1 = mpatches.Patch(color=BLUE,   label="CatBoost baseline phase")
    p2 = mpatches.Patch(color=TEAL,   label="CB+LGBM blend phase")
    p3 = mpatches.Patch(color=PURPLE, label="Pseudo-label + greedy phase")
    p4 = mpatches.Patch(color=GREEN,  label="Rank-blend final phase")
    ax.legend(handles=[p1, p2, p3, p4], fontsize=10, frameon=False, loc="lower right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.5f}"))

    plt.tight_layout()
    path = os.path.join(OUT, "slide6_score_progression.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# SLIDE 6b – Final Ensemble Diagram  (v31_rank_v30_v20 = 0.78271)
# =============================================================================
def slide6_ensemble_diagram():
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def box(x, y, w, h, color, line1, line2="", fontsize=10):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                              facecolor=color, edgecolor="white", linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + (0.25 if line2 else 0),
                line1, ha="center", va="center",
                fontsize=fontsize, fontweight="bold",
                color="white" if color not in (LGRAY,) else DGRAY)
        if line2:
            ax.text(x + w / 2, y + h / 2 - 0.30, line2, ha="center", va="center",
                    fontsize=8.5,
                    color="white" if color not in (LGRAY,) else DGRAY)

    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->,head_width=0.22,head_length=0.14",
                                   color=DGRAY, lw=2.0))

    # Model A: v30 greedy pl2 ensemble
    box(0.15, 3.4, 3.8, 2.6, BLUE, "v30  Greedy pl2 Ensemble",
        "cb_pl  \u00d70.653\nlgbm_pl2 \u00d70.163   cb_pl2 \u00d70.144\n"
        "lgbm_pl  \u00d70.020   cb_fe  \u00d70.020\nOOF AUC = 0.780235",
        fontsize=9)

    # Model B: v20 greedy multistart (no pseudo)
    box(0.15, 0.4, 3.8, 2.6, TEAL, "v20  Greedy Multistart Ensemble",
        "cb_fe     \u00d70.509   cb_20s  \u00d70.323\n"
        "lgbm_20s \u00d70.090   freq_lgbm \u00d70.038\n"
        "lgbm_te  \u00d70.020   freq_cb  \u00d70.020\nOOF AUC = 0.779003",
        fontsize=9)

    ax.text(2.05, 3.2, "OOF correlation v30 vs v20 = 0.9915",
            ha="center", fontsize=9, color=AMBER, style="italic")

    # Rank normalisation
    box(5.0, 3.4, 2.8, 2.2, PURPLE, "Rank-Normalise A",
        "sort 72,006 test preds\nassign rank 1..N\nnormalise to [0, 1]", fontsize=9)
    box(5.0, 0.4, 2.8, 2.2, PURPLE, "Rank-Normalise B",
        "sort 72,006 test preds\nassign rank 1..N\nnormalise to [0, 1]", fontsize=9)

    arr(3.95, 4.7, 5.0, 4.7)
    arr(3.95, 1.7, 5.0, 1.7)

    # Equal-weight average
    box(9.0, 2.1, 2.8, 1.8, AMBER, "Equal-Weight Average",
        "(rank_A + rank_B) / 2\nParameter-free", fontsize=9)
    arr(7.8, 4.5, 8.8, 3.05)
    arr(7.8, 1.5, 8.8, 2.40)

    # Final submission
    box(12.0, 2.1, 1.85, 1.8, GREEN, "Final Sub",
        "v31_rank_v30_v20\nLB = 0.78271\n(2nd Place)", fontsize=9)
    arr(11.8, 3.0, 12.0, 3.0)

    ax.set_title(
        "Final Ensemble:  rank_avg(v30 greedy-pl2,  v20 greedy-multistart)  \u2192  0.78271 Public LB",
        fontsize=12, fontweight="bold", pad=12, color=DBLUE)
    plt.tight_layout()
    path = os.path.join(OUT, "slide6_ensemble_diagram.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# SLIDE 7 – Rank Averaging vs Probability Averaging
# =============================================================================
def slide7_rank_vs_prob():
    # Design two models with different probability scales so rank avg clearly wins.
    # Model A: full [0,1] range.  Model B: compressed to [0, 0.25] range.
    # Prob avg is dominated by Model A's scale; rank avg normalises both.
    np.random.seed(7)
    n = 600
    true_y = np.random.binomial(1, 0.25, n)

    # Model A — full scale, moderate discrimination
    raw_a = np.where(true_y == 1,
                     np.random.beta(3, 2, n),
                     np.random.beta(2, 4, n))
    model_a = np.clip(raw_a, 0, 1)

    # Model B — compressed scale [0, 0.25], similar discrimination
    raw_b = np.where(true_y == 1,
                     np.random.beta(3, 2, n) * 0.25,
                     np.random.beta(2, 4, n) * 0.25)
    model_b = np.clip(raw_b, 0, 1)

    prob_avg = (model_a + model_b) / 2
    ranks_a  = pd.Series(model_a).rank(pct=True).values
    ranks_b  = pd.Series(model_b).rank(pct=True).values
    rank_avg = (ranks_a + ranks_b) / 2

    auc_a   = roc_auc_score(true_y, model_a)
    auc_b   = roc_auc_score(true_y, model_b)
    auc_pa  = roc_auc_score(true_y, prob_avg)
    auc_ra  = roc_auc_score(true_y, rank_avg)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for preds, label, color in [
        (model_a,  f"Model A  (AUC={auc_a:.3f})",  BLUE),
        (model_b,  f"Model B  (AUC={auc_b:.3f})",  TEAL),
        (prob_avg, f"Prob Avg (AUC={auc_pa:.3f})", AMBER),
        (rank_avg, f"Rank Avg (AUC={auc_ra:.3f})", GREEN),
    ]:
        fpr, tpr, _ = roc_curve(true_y, preds)
        lw = 2.5 if "Rank" in label and "Avg" in label else 1.5
        ls = "-" if "Avg" in label else "--"
        axes[0].plot(fpr, tpr, label=label, lw=lw, linestyle=ls, color=color)

    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    axes[0].set_xlabel("False Positive Rate", fontsize=12)
    axes[0].set_ylabel("True Positive Rate", fontsize=12)
    axes[0].set_title("ROC Curves \u2014 Rank Avg vs Prob Avg", fontsize=13,
                      fontweight="bold")
    axes[0].legend(fontsize=10, frameon=False)

    labels_bar = ["Model A", "Model B", "Prob Avg", "Rank Avg"]
    aucs_bar   = [auc_a, auc_b, auc_pa, auc_ra]
    colors_bar = [BLUE, TEAL, AMBER, GREEN]
    bars = axes[1].bar(labels_bar, aucs_bar, color=colors_bar,
                       edgecolor="white", linewidth=1.5, width=0.5)
    for bar, v in zip(bars, aucs_bar):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.002,
                     f"{v:.4f}", ha="center", va="bottom",
                     fontsize=12, fontweight="bold")
    axes[1].set_ylim(min(aucs_bar) - 0.02, max(aucs_bar) + 0.03)
    axes[1].set_ylabel("ROC-AUC", fontsize=12)
    axes[1].set_title("AUC Comparison\n(Rank Avg \u2265 Prob Avg)",
                      fontsize=13, fontweight="bold")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))

    fig.suptitle("Why Rank Averaging \u2014 Illustrated on Simulated Data",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT, "slide7_rank_vs_prob.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# SLIDE 8 – Leaderboard Results Table
# =============================================================================
def slide8_leaderboard():
    lb_data = {
        "#": ["1", "2 \u2605", "3", "\u2026"],
        "Team": ["Joshua Tiffany", "NeerJain04", "(3rd place)", "\u2026"],
        "Public Score": ["0.78277", "0.78271", "~0.7820", "\u2026"],
        "Entries": ["57", "45", "\u2014", "\u2014"],
        "Gap from #1": ["\u2014", "+0.00006", "\u2014", "\u2014"],
    }
    df = pd.DataFrame(lb_data)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(14)
    tbl.scale(1, 2.4)

    row2_color = "#FEF3C7"
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if r == 0:
            cell.set_facecolor(DBLUE)
            cell.set_text_props(color="white", fontweight="bold")
        elif r == 2:
            cell.set_facecolor(row2_color)
            cell.set_text_props(fontweight="bold", color="#92400E")
        else:
            cell.set_facecolor("white")

    ax.set_title("Final Leaderboard Position",
                 fontsize=16, fontweight="bold", pad=14, color=DBLUE)
    note = "\u2605 NeerJain04 \u2014 2nd place,  trailing leader by only 0.00006 AUC (45 submissions)"
    ax.text(0.5, -0.05, note, transform=ax.transAxes,
            ha="center", fontsize=11, color=DGRAY, style="italic")
    plt.tight_layout()
    path = os.path.join(OUT, "slide8_leaderboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# SLIDE 8b – Full Submission History
# =============================================================================
def slide8_submission_history():
    subs = [
        ("submission",        0.78056),
        ("submission_v2",     0.78166),
        ("submission_v3",     0.78121),
        ("sub_stack",         0.77991),
        ("v4",                0.78154),
        ("v5",                0.78136),
        ("v6",                0.78178),
        ("v7",                0.78188),
        ("v8",                0.78214),
        ("v9",                0.78199),
        ("v10",               0.78218),
        ("v11",               0.78203),
        ("v12_pure_cb",       0.78191),
        ("v13",               0.78191),
        ("v14",               0.78216),
        ("v15_rank",          0.78217),
        ("v15_w5_3_2",        0.78217),
        ("v15_w6_3_1",        0.78218),
        ("v15_w7_2_1",        0.78218),
        ("v16",               0.78189),
        ("v17",               0.78051),
        ("v18_pseudo",        0.78245),
        ("v19_greedy",        0.78249),
        ("v20_greedy_ms",     0.78256),
        ("v21_fine_wts",      0.78253),
        ("v22_greedy_pl",     0.78250),
        ("v23_fine_v22",      0.78254),
        ("v23_rank_blend",    0.78264),
        ("v24_xgb_ms",        0.78256),
        ("v25_r_v23v20v19",   0.78264),
        ("v25_r+v24",         0.78267),
        ("v25_r_v23v20v24",   0.78267),
        ("v26_rank_v23v20",   0.78268),
        ("v26_wrank",         0.78267),
        ("v27_score_v23v20",  0.78268),
        ("v27_score_v23only", 0.78264),
        ("v29_pl2_fine",      0.78252),
        ("v30_greedy_pl2",    0.78254),
        ("v31_rank_v30v20",   0.78271),
        ("v31_r_v30v23v20",   0.78268),
        ("v31_rank_v29v20",   0.78269),
        ("v32_score_v29v20",  0.78270),
        ("v33_r_xgbv20",      0.78242),
        ("v33_r_xgbv30",      0.78257),
        ("v33_score_v30v20",  0.78271),
    ]

    names  = [s[0] for s in subs]
    scores = [s[1] for s in subs]
    xs     = list(range(len(subs)))

    best_so_far = []
    best = 0.0
    for s in scores:
        best = max(best, s)
        best_so_far.append(best)

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.fill_between(xs, [min(scores) - 0.0002] * len(xs),
                    best_so_far, alpha=0.06, color=GREEN)
    ax.scatter(xs, scores, color=BLUE, s=35, zorder=3, alpha=0.8)
    ax.plot(xs, best_so_far, color=GREEN, lw=2.5, label="Personal best so far", zorder=4)

    ax.axvline(21, color=PURPLE, linestyle="--", lw=1.5, alpha=0.7,
               label="v18: pseudo-labeling starts")
    ax.text(21.2, 0.7802, "v18\npseudo-\nlabeling", fontsize=8, color=PURPLE)

    best_idx = scores.index(max(scores))
    ax.annotate(f"Best: {max(scores):.5f}",
                xy=(best_idx, max(scores)),
                xytext=(best_idx - 6, max(scores) + 0.00018),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5),
                fontsize=10, color=GREEN, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=7.5)
    ax.set_ylabel("Public LB ROC-AUC", fontsize=12)
    ax.set_ylim(0.7792, 0.7832)
    ax.set_title("All 45 Submissions \u2014 Public Leaderboard Score History",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=11, frameon=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.5f}"))

    plt.tight_layout()
    path = os.path.join(OUT, "slide8_submission_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# SLIDE 9 – Key Takeaways Summary Table  (actual LB scores)
# =============================================================================
def slide9_takeaways():
    data = {
        "What": [
            "CatBoost on raw categoricals",
            "10-seed \u00d7 5-fold ensembling",
            "CB+LGBM blend (v8\u2192v10)",
            "Pseudo-labeling round 1 (v18)",
            "Pseudo-labeling round 2 (v29/v30)",
            "Greedy multistart ensemble (v20)",
            "Fine-weight grid search (v23)",
            "Rank averaging final blend (v31/v33)",
            "Freq / interaction features (v16, v17)",
            "LGBM Optuna tuning (v9)",
            "3-model stacking (submission_stack)",
        ],
        "Best LB": [
            "0.78166 (v2)", "0.78188 (v7)", "0.78218 (v10)",
            "0.78245 (v18)", "0.78254 (v30)", "0.78256 (v20)",
            "0.78254 (v23)", "0.78271 (v31/v33 \u2014 BEST)",
            "0.78189 / 0.78051", "0.78199 (v9)", "0.77991",
        ],
        "\u2713 / \u2717": ["\u2713", "\u2713", "\u2713", "\u2713", "\u2713",
                            "\u2713", "\u2713", "\u2713\u2713",
                            "\u2717", "\u2717", "\u2717\u2717"],
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns,
                   cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 1.85)

    col_widths = [0.38, 0.26, 0.10]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if c < len(col_widths):
            cell.set_width(col_widths[c])
        if r == 0:
            cell.set_facecolor(DBLUE)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            if c == 2:
                sym = cell.get_text().get_text()
                if "\u2713\u2713" in sym:
                    cell.set_facecolor("#DCFCE7")
                elif "\u2713" in sym:
                    cell.set_facecolor("#F0FDF4")
                elif "\u2717\u2717" in sym:
                    cell.set_facecolor("#FEE2E2")
                elif "\u2717" in sym:
                    cell.set_facecolor("#FEF2F2")
            elif r % 2 == 0:
                cell.set_facecolor(LGRAY)
            else:
                cell.set_facecolor("white")

    ax.set_title("What Worked vs What Didn\u2019t \u2014 Actual Public LB Scores",
                 fontsize=14, fontweight="bold", pad=14, color=DBLUE)
    plt.tight_layout()
    path = os.path.join(OUT, "slide9_takeaways.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# RUN ALL
# =============================================================================
if __name__ == "__main__":
    print("Generating all slide images...\n")
    slide2_class_distribution()
    slide2_dataset_table()
    slide3_model_comparison()
    slide3_catboost_folds()
    slide3_optuna_convergence()
    slide4_feature_importance()
    slide4_what_didnt_work()
    slide5_pseudo_label_diagram()
    slide6_score_progression()
    slide6_ensemble_diagram()
    slide7_rank_vs_prob()
    slide8_leaderboard()
    slide8_submission_history()
    slide9_takeaways()
    print(f"\nAll done. Images saved to: osf_no_show_project/slide_images/")
