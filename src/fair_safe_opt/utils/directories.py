from datetime import datetime
from pathlib import Path
import shutil


def create_report_dir(architecture, method, seed):
    """Create a out_directory for today, and one inside for this run"""
    datestr = datetime.now().strftime("%Y.%m.%d")
    timestr = datetime.now().strftime("%H%M")
    date_dir = Path.cwd() / "results" / datestr

    name = f"{architecture}_seed{seed}_{method}_{timestr}"
    out_dir = date_dir / name

    if Path(out_dir).exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir
