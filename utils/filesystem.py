from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results_data"

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def create_results_base_folder(run_label=None) -> Path:
    ensure_dir(RESULTS_DIR)
    if run_label:
        result_path = RESULTS_DIR / run_label
    else:
        timestamp   = datetime.now().strftime('%m-%d-%H-%M-%S')
        result_path = RESULTS_DIR / timestamp
    ensure_dir(result_path)
    return result_path

def create_run_subfolder(base_folder: Path, run_index: int, pop_size: int) -> Path:
    subfolder = base_folder / f"size{pop_size}_run{run_index+1}"
    ensure_dir(subfolder)
    return subfolder

def get_results_file_path(run_folder: Path, suffix: str) -> Path:
    current_date = datetime.now().strftime('%Y-%m-%d')
    return run_folder / f"results_{suffix}_{current_date}.xlsx"

def get_preprocessed_file_path(run_folder: Path) -> Path:
    current_date = datetime.now().strftime('%Y-%m-%d')
    return run_folder / f"preprocessed_data_{current_date}.xlsx"
