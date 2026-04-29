"""
    Entry point for the genetic algorithm.
    Supports single runs and batch runs from an Excel table.

    Usage:
        Single run:  python main.py --mode single [--seed 42] [--label my_run]
        Batch mode:  python main.py --mode batch [--excel path/to/Beste-Features.xlsx]

    Note:
        In single run mode, all hyperparameters (target, regression, GA settings,
        island model settings, etc.) are read directly from config.py.
        Only the random seed and an optional run label can be passed via CLI.
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import random
import config
import timeit
import argparse
from pathlib import Path
from datetime import datetime
from data_processing import data_processor
from utils import filesystem as fs
import run_island_model as island_runner
from algorithms.utils.terminal_logger import init_feature_freq_log
import xlsxwriter
import traceback


# ── MAPPING-Tables ──────────────────────────────────────────────────────────
TARGET_MAP = {
    'IB':      ['IB_AVG'],
    'Density': ['DENSITY_AVG'],
    'MOR':     ['MOR_AVG'],
}

RULES_MAP = {
    'No exp':     'simple',
    'No rules':   'simple',
    'With Rules': 'with_rules',
    'With rules': 'with_rules',
}

CXPB_MUTPB_MAP = {
    'simple':     (0.8, 0.1),
    'with_rules': (0.3, 0.5),
}


# ── overwrite config (Batch only) ────────────────────────────────────────────
def apply_config(targets, regression, rule_type, seeds):
    cxpb, mutpb = CXPB_MUTPB_MAP[rule_type]

    config.targets    = targets
    config.regression = [regression]
    config.rule_type  = rule_type
    config.cxpb       = cxpb
    config.mutpb      = mutpb
    config.seeds      = seeds

    print(f"  config updated:")
    print(f"    targets={config.targets}")
    print(f"    regression={config.regression}")
    print(f"    rule_type={config.rule_type}")
    print(f"    cxpb={config.cxpb}, mutpb={config.mutpb}")


# ── single run ───────────────────────────────────────────────────────────────
def main(seeds, run_label=None):
    """
    Executes the GA for a single run.
    All hyperparameters are read from config.py.
    Seeds and run_label can be passed directly.
    """
    if len(seeds) < config.validation_runs:
        raise ValueError("Not enough seeds defined.")

    base_results_folder = fs.create_results_base_folder(run_label=run_label)

    for validation_run in range(config.validation_runs):
        run_folder        = fs.create_run_subfolder(base_results_folder, validation_run, config.pop_size)
        results_file_path = fs.get_results_file_path(run_folder, f"run{validation_run+1}")

        if not results_file_path.exists():
            workbook = xlsxwriter.Workbook(str(results_file_path))
            workbook.add_worksheet()
            workbook.close()

        for i, target in enumerate(config.targets):
            print(f"\nRunning validation run {validation_run + 1} for target: {target}")

            random.seed(seeds[validation_run][i])
            np.random.seed(random.randint(0, 2**32 - 1))

            processor = data_processor.DataProcessor(
                target, config.material_type, group_type=config.group_type
            )
            dataset = processor.run()

            init_feature_freq_log(dataset.df_preprocessed.shape[1])

            start = timeit.default_timer()
            island_runner.execute(
                num_islands=config.num_islands,
                dataset=dataset,
                results_folder=run_folder,
                results_file_path=results_file_path
            )
            stop = timeit.default_timer()

            print(
                f"Validation run {validation_run + 1} for target {target} "
                f"completed in {round(stop - start)} seconds."
            )


# ── BATCH-MODE ───────────────────────────────────────────────────────────────
def run_batch(excel_path="Beste-Features.xlsx"):
    df_runs = pd.read_excel(excel_path, sheet_name="Tabelle1", engine="calamine")
    total   = len(df_runs)
    errors  = []

    print(f"{'='*60}")
    print(f"Batch mode started: {total} runs")
    print(f"Source: {excel_path}")
    print(f"{'='*60}\n")

    for run_idx, row in df_runs.iterrows():
        quality    = row["Qualityparameter"]
        seed_val   = int(row["Seed"])
        regression = row["Regression"]
        rules_raw  = row["Rules"]

        targets   = TARGET_MAP[quality]
        rule_type = RULES_MAP[rules_raw]
        seeds     = [[seed_val] * 3] * 3
        run_label = (
            f"{quality}_Seed{seed_val}_"
            f"{regression}_"
            f"{rules_raw.replace(' ', '_')}"
        )

        print(f"\n{'='*60}")
        print(f"[{run_idx+1}/{total}]  {run_label}")
        print(f"{'='*60}")

        try:
            apply_config(
                targets    = targets,
                regression = regression,
                rule_type  = rule_type,
                seeds      = seeds,
            )
            main(seeds=seeds, run_label=run_label)
            print(f"\n  {run_label} completed.")

        except Exception as e:
            traceback.print_exc()
            print(f"Error in {run_label}: {e}")
            errors.append({"run": run_label, "error": str(e)})
            with open("batch_errors.log", "a") as f:
                f.write(f"{datetime.now()} | {run_label} | {e}\n")
            continue

    print(f"\n{'='*60}")
    print(f"Batch completed: {total - len(errors)}/{total} successful")
    if errors:
        print(f"\nFailed runs ({len(errors)}):")
        for err in errors:
            print(f"  • {err['run']}: {err['error']}")
    print(f"{'='*60}")


# ── ARGUMENT PARSER ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Genetic Algorithm Feature Selection for Particleboard Quality Prediction",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--mode',
        choices=['single', 'batch'],
        required=True,
        help=(
            "Execution mode:\n"
            "  single  → run the GA once using all settings from config.py\n"
            "  batch   → run multiple configurations from an Excel file"
        )
    )

    # ── Single-run arguments ─────────────────────────────────────────────────
    single_group = parser.add_argument_group(
        'Single Run Options',
        'Only used with --mode single. All other settings are read from config.py.'
    )

    single_group.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    single_group.add_argument(
        '--label',
        type=str,
        default=None,
        help="Optional label for the results folder (default: auto-generated from config)"
    )

    # ── Batch-mode arguments ─────────────────────────────────────────────────
    batch_group = parser.add_argument_group(
        'Batch Mode Options',
        'Only used with --mode batch.'
    )

    batch_group.add_argument(
        '--excel',
        type=str,
        default=None,
        help="Path to the Excel file with batch run configurations\n(default: algorithms/Beste-Features.xlsx)"
    )

    return parser.parse_args()


# ── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    BASE_DIR = Path(__file__).parent
    args     = parse_args()

    if args.mode == 'batch':
        excel_path = Path(args.excel) if args.excel else BASE_DIR / "algorithms" / "Beste-Features.xlsx"
        run_batch(excel_path=excel_path)

    elif args.mode == 'single':
        # All hyperparameters come from config.py
        # Seeds are built from the CLI seed for each target and validation run
        seeds      = [[args.seed] * len(config.targets)] * config.validation_runs
        run_label  = args.label or f"single_{'_'.join(config.targets)}_Seed{args.seed}"

        print(f"{'='*60}")
        print(f"Single run started")
        print(f"  target(s)    : {config.targets}")
        print(f"  regression   : {config.regression}")
        print(f"  rule_type    : {config.rule_type}")
        print(f"  seed         : {args.seed}")
        print(f"  run_label    : {run_label}")
        print(f"  All other hyperparameters are read from config.py")
        print(f"{'='*60}\n")

        main(seeds=seeds, run_label=run_label)