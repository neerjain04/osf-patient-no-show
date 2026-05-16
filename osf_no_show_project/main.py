# =============================================================================
# main.py
# -----------------------------------------------------------------------------
# The control centre for the entire pipeline.
# Instead of manually running individual Python files, you run this one file
# and pass flags to tell it what to do.
#
# Usage examples:
#   python main.py --baseline              # train CatBoost baseline + save OOF
#   python main.py --submit                # generate submissions/submission.csv
#   python main.py --submit --output v2.csv  # custom submission filename
#   python main.py --baseline --submit     # chain: baseline then submission
#   python main.py --tune                  # Optuna hyperparameter search (50 trials)
#   python main.py --tune --trials 100     # run 100 trials instead of 50
#   python main.py --stack                 # 3-model stacking ensemble (CatBoost+LightGBM+XGBoost)
#   python main.py --stack --output v4.csv # custom filename for stack submission
#   python main.py --help                  # show all available flags
# =============================================================================

import os
import argparse
import sys


def main():
    # argparse builds a proper CLI parser with --help documentation built in.
    parser = argparse.ArgumentParser(
        description="OSF No-Show Prediction Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Each add_argument call defines one flag the user can pass.
    # action="store_true" means the flag is a boolean switch (no value needed).
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run CatBoost baseline with 5-fold CV and save OOF predictions",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Generate Kaggle submission file using 5-fold CatBoost ensemble",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        # Allows versioning submissions without editing code, e.g. --output v2.csv
        help="Filename for the submission (default: submission.csv)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter search and save best params to results/best_params.json",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna trials to run (default: 50; more = better but slower)",
    )
    parser.add_argument(
        "--stack",
        action="store_true",
        help="Run 3-model stacking ensemble (CatBoost + LightGBM + XGBoost) and save submission",
    )
    parser.add_argument(
        "--blend",
        action="store_true",
        help="Run CatBoost + LightGBM blend ensemble (10-seed x 5-fold each) and save submission",
    )
    parser.add_argument(
        "--tune-lgbm",
        action="store_true",
        help="Run Optuna hyperparameter search for LightGBM and save best params to results/best_lgbm_params.json",
    )
    parser.add_argument(
        "--lgbm-trials",
        type=int,
        default=100,
        help="Number of Optuna trials for LightGBM tuning (default: 100)",
    )
    parser.add_argument(
        "--blend-te",
        action="store_true",
        help="Run CatBoost + Target-Encoded LightGBM blend/stack (v11 strategy)",
    )
    parser.add_argument(
        "--top10",
        action="store_true",
        help="Run v10 blend formula using only the top 10 features by importance",
    )
    parser.add_argument(
        "--blend-subs",
        action="store_true",
        help="Blend v8+v10+v14 submission CSVs via rank averaging and fixed weight combos",
    )
    parser.add_argument(
        "--blend-freq",
        action="store_true",
        help="Run CB+LGBM blend with frequency-encoded features added (10 seeds x 5 folds, trains from scratch)",
    )
    parser.add_argument(
        "--blend-fe",
        action="store_true",
        help="Run CB+LGBM blend with 7 interaction features added (10 seeds x 5 folds, trains from scratch)",
    )
    parser.add_argument(
        "--cb-depth9",
        action="store_true",
        help="Train CatBoost variant depth=9, lr=0.02, iter=2000 (same 10 seeds as v10). Saves cb_d9_oof/test.npy",
    )
    parser.add_argument(
        "--blend-variants",
        action="store_true",
        help="Blend v10 + depth-9 CB using OOF weight search + rank averaging",
    )
    parser.add_argument(
        "--greedy-ensemble",
        action="store_true",
        help="Greedy ensemble selection (hill-climbing) over all saved OOF .npy files. No training — runs instantly.",
    )
    parser.add_argument(
        "--greedy-multistart",
        action="store_true",
        help="Run greedy ensemble from every possible seed model; keep best OOF result.",
    )
    parser.add_argument(
        "--fine-weights",
        action="store_true",
        help="Dense weight grid search over a fixed model combo (see --fine-combo).",
    )
    parser.add_argument(
        "--fine-step",
        type=float,
        default=0.02,
        help="Weight step size for --fine-weights grid search (default: 0.02).",
    )
    parser.add_argument(
        "--fine-combo",
        type=str,
        default="v19",
        help="Model combo for --fine-weights: 'v19' (4-model) or 'v22' (3-model cb_pl+lgbm_pl+cb_fe). Default: v19.",
    )
    parser.add_argument(
        "--pseudo-label",
        action="store_true",
        help="Retrain v10 formula with high-confidence test pseudo-labels appended (10 seeds x 5 folds)",
    )
    parser.add_argument(
        "--pos-thresh",
        type=float,
        default=0.60,
        help="Minimum predicted probability to pseudo-label a test row as positive (default: 0.60)",
    )
    parser.add_argument(
        "--neg-thresh",
        type=float,
        default=0.02,
        help="Maximum predicted probability to pseudo-label a test row as negative (default: 0.02)",
    )
    parser.add_argument(
        "--pseudo-source",
        type=str,
        default="v10.csv",
        help="Submission CSV to use as pseudo-label source (default: v10.csv)",
    )
    parser.add_argument(
        "--save-tag",
        type=str,
        default="pl",
        help="Tag for saved npy files, e.g. 'pl2' for second round (default: pl)",
    )
    parser.add_argument(
        "--rank-blend",
        action="store_true",
        help="Rank-average a set of submissions (see --rank-blend-combo).",
    )
    parser.add_argument(
        "--rank-blend-combo",
        type=str,
        default="v22_v20_v19",
        help="Which combo to rank-blend: v22_v20_v19 | v23_v20_v19 | v23_v20_v24 | v23_v20_v19_v24",
    )
    parser.add_argument(
        "--rank-blend-all",
        action="store_true",
        help="Generate all rank blend combos at once (saves v25_rank_*.csv files).",
    )
    parser.add_argument(
        "--weighted-rank-blend",
        action="store_true",
        help="Weighted rank blend: 0.45*rank(v23)+0.35*rank(v20)+0.20*rank(v24) → v26_wrank_v23_v20_v24.csv; also saves 2-way v26_rank_v23_v20.csv.",
    )
    parser.add_argument(
        "--sweep-rank-v23-v20",
        action="store_true",
        help="Sweep w in w*rank(v23)+(1-w)*rank(v20) over [0.05,0.95]; saves best as v27_wrank_sweep_v23_v20.csv.",
    )
    parser.add_argument(
        "--score-blend-v23-v20",
        action="store_true",
        help="Score-average (prob space) blend of v23+v20 and v23 standalone → v27_score_v23_v20.csv + v27_score_v23_only.csv.",
    )
    parser.add_argument(
        "--score-blend-v29-v20",
        action="store_true",
        help="Score-average (prob space) blend of v29+v20 → v32_score_v29_v20.csv.",
    )
    parser.add_argument(
        "--score-blend-v30-v20",
        action="store_true",
        help="Score-average (prob space) blend of v30+v20 → v33_score_v30_v20.csv.",
    )
    parser.add_argument(
        "--xgb-v30-v20",
        action="store_true",
        help="3-way rank blends of xgb+v30+v20: equal (v34_rank) and weighted 0.10/0.45/0.45 (v34_wrank).",
    )
    parser.add_argument(
        "--xgb",
        action="store_true",
        help="Train XGBoost 10-seed x 5-fold and save xgb_oof.npy/xgb_test.npy for greedy pool.",
    )

    args = parser.parse_args()

    # If the user ran `python main.py` with no flags, print help and exit cleanly
    # rather than doing nothing silently.
    if not any([args.baseline, args.submit, args.tune, args.stack, args.blend, args.tune_lgbm, args.blend_te, args.top10, args.cb_depth9, args.blend_variants, args.blend_fe, args.blend_subs, args.blend_freq, args.pseudo_label, args.greedy_ensemble, args.greedy_multistart, args.fine_weights, args.rank_blend, args.rank_blend_all, args.weighted_rank_blend, args.sweep_rank_v23_v20, args.score_blend_v23_v20, args.score_blend_v29_v20, args.score_blend_v30_v20, args.xgb_v30_v20, args.xgb]):
        parser.print_help()
        sys.exit(0)

    # Imports are done inside each branch so that if catboost is not installed,
    # the error message is clear and specific.
    if args.baseline:
        from train_baseline import train_baseline
        train_baseline()

    if args.submit:
        from submission import generate_submission
        generate_submission(output_name=args.output)

    if args.tune:
        from tune import run_tuning
        run_tuning(n_trials=args.trials)

    if args.stack:
        from stack import run_stacking
        run_stacking(output_name=args.output)

    if args.blend:
        from blend import run_blend
        run_blend(output_name=args.output)

    if args.tune_lgbm:
        from tune_lgbm import run_lgbm_tuning
        run_lgbm_tuning(n_trials=args.lgbm_trials)

    if args.blend_te:
        from blend_te_lgbm import run_blend_te
        run_blend_te(output_name=args.output)

    if args.top10:
        from blend import run_blend
        _TOP10 = [
            "PATIENT_NOSHOWRATE_CATEGORY",
            "PATIENT_AVG_APPT2DOC_CATEGORY",
            "AGE_CATEGORY",
            "PATIENT_EMPLOYMENT_STATUS_CATEGORY",
            "DAYS_BETWEEN_CATEGORY",
            "PATIENT_MARITAL_STATUS_CATEGORY",
            "LENGTH",
            "PROV_NOSHOWRATE_CATEGORY",
            "DEPT_NOSHOW_RATE_CATEGORY",
            "VISIT_TYPE",
        ]
        run_blend(output_name=args.output, feature_subset=_TOP10)

    if args.blend_subs:
        from blend import run_blend_subs
        _SUBS_DIR = os.path.join(os.path.dirname(__file__), "submissions")
        run_blend_subs(
            files=[
                ("v10", os.path.join(_SUBS_DIR, "v10.csv")),
                ("v14", os.path.join(_SUBS_DIR, "v14.csv")),
                ("v8",  os.path.join(_SUBS_DIR, "v8.csv")),
            ],
            output_prefix="v15",
        )

    if args.blend_freq:
        from blend import run_blend_freq
        run_blend_freq(output_name=args.output)

    if args.blend_fe:
        from blend import run_blend_fe
        run_blend_fe(output_name=args.output)

    if args.cb_depth9:
        from blend import run_cb_variant
        run_cb_variant(depth=9, lr=0.02, iterations=2000, tag="cb_d9")

    if args.blend_variants:
        from blend import run_blend_variants
        run_blend_variants(output_name=args.output)

    if args.greedy_ensemble:
        from blend import run_greedy_ensemble
        run_greedy_ensemble(output_name=args.output)

    if args.greedy_multistart:
        from blend import run_greedy_multistart
        run_greedy_multistart(output_name=args.output)

    if args.fine_weights:
        from blend import run_fine_weight_search
        run_fine_weight_search(output_name=args.output, step=args.fine_step, combo=args.fine_combo)

    if args.pseudo_label:
        from blend import run_pseudo_label
        run_pseudo_label(
            pos_threshold=args.pos_thresh,
            neg_threshold=args.neg_thresh,
            output_name=args.output,
            source_csv=args.pseudo_source,
            save_tag=args.save_tag,
        )

    if args.rank_blend:
        from blend import run_rank_blend
        run_rank_blend(output_name=args.output, combo=args.rank_blend_combo)

    if args.rank_blend_all:
        from blend import run_rank_blend_all
        run_rank_blend_all()

    if args.weighted_rank_blend:
        from blend import run_weighted_rank_blend
        run_weighted_rank_blend()

    if args.sweep_rank_v23_v20:
        from blend import run_sweep_rank_v23_v20
        run_sweep_rank_v23_v20()

    if args.score_blend_v23_v20:
        from blend import run_score_blend_v23_v20
        run_score_blend_v23_v20()

    if args.score_blend_v29_v20:
        from blend import run_score_blend_v29_v20
        run_score_blend_v29_v20(output_name=args.output if args.output else "v32_score_v29_v20.csv")

    if args.score_blend_v30_v20:
        from blend import run_score_blend_v30_v20
        run_score_blend_v30_v20(output_name=args.output if args.output else "v33_score_v30_v20.csv")

    if args.xgb_v30_v20:
        from blend import run_xgb_v30_v20_rank_blends
        run_xgb_v30_v20_rank_blends()

    if args.xgb:
        from blend import run_xgb_candidate
        run_xgb_candidate()


if __name__ == "__main__":
    main()
