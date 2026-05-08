from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"


def ensure_directories() -> None:
    """Create directories used by the project if they do not exist."""
    for path in [PROCESSED_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def print_step(message: str) -> None:
    """Print a simple progress banner for the CLI workflow."""
    print(f"\n{'=' * 20} {message} {'=' * 20}")
