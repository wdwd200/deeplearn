from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase5_common import PHASE4_ROOT, PHASE5_FIGURES, PHASE5_SOURCE_DATA  # noqa: E402
from plot_paper_figures import plot_paper_figures  # noqa: E402

EXPECTED_FIGURES = [
    "figure_1_system_model",
    "figure_2_modeling_pipeline",
    "figure_3_training_curves",
    "figure_4_algorithm_rate_comparison",
    "figure_5_scenario_comparison",
    "figure_6_ablation_rate",
    "figure_7_constraint_violations",
    "figure_8_representative_trajectories",
    "figure_9_rate_over_time",
]


def _ensure_figures() -> None:
    missing = [
        name
        for name in EXPECTED_FIGURES
        if not (PHASE5_FIGURES / f"{name}.png").exists() or not (PHASE5_FIGURES / f"{name}.pdf").exists()
    ]
    if missing:
        plot_paper_figures()


def test_paper_figures_exist_with_png_and_pdf_outputs() -> None:
    before = {path: path.stat().st_mtime_ns for path in (PHASE4_ROOT / "figures").glob("*.png")}
    _ensure_figures()
    after = {path: path.stat().st_mtime_ns for path in (PHASE4_ROOT / "figures").glob("*.png")}

    assert before == after
    for name in EXPECTED_FIGURES:
        png_path = PHASE5_FIGURES / f"{name}.png"
        pdf_path = PHASE5_FIGURES / f"{name}.pdf"
        assert png_path.exists()
        assert pdf_path.exists()
        assert png_path.stat().st_size > 0
        assert pdf_path.stat().st_size > 0


def test_png_dpi_is_at_least_600() -> None:
    _ensure_figures()
    for name in EXPECTED_FIGURES:
        png_path = PHASE5_FIGURES / f"{name}.png"
        with Image.open(png_path) as image:
            dpi = image.info.get("dpi", (0.0, 0.0))[0]
        assert dpi >= 599.0


def test_figure_source_data_exists() -> None:
    _ensure_figures()
    assert (PHASE5_SOURCE_DATA / "representative_trajectories.csv").exists()
    assert (PHASE5_SOURCE_DATA / "representative_rate_trace.csv").exists()
