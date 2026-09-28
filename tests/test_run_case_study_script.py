from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from test_pipeline_smoke import (
    _write_smoke_fixtures,
    _write_smoke_fixtures_with_factors_dir,
)

from portfolio_bl.config import load_config
from portfolio_bl.pipeline import run_case_study


def _import_run_case_study_script() -> ModuleType:
    """Import scripts/run_case_study.py without adding it to sys.path."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_case_study.py"
    spec = importlib.util.spec_from_file_location("run_case_study", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_case_study_script = _import_run_case_study_script()

_METADATA_ROWS = ["person_label", "as_of_date", "n_assets", "universe", "risk_free_rate"]


def _add_ff5_daily(factors_dir: Path, seed: int = 11) -> None:
    """Write an ff5_daily.csv derived from the fixture's ff3_daily.csv.

    ``run_case_study.py`` always attempts factor attribution (CAPM, FF3 and
    FF5) once ``data.factors_dir`` is configured, so the FF5 file has to
    exist alongside the FF3 files the smoke fixture already writes.
    """
    ff3 = pd.read_csv(factors_dir / "ff3_daily.csv")
    rng = np.random.default_rng(seed)
    n = len(ff3)
    pd.DataFrame(
        {
            "date": ff3["date"],
            "Mkt-RF": ff3["Mkt-RF"],
            "SMB": ff3["SMB"],
            "HML": ff3["HML"],
            "RMW": rng.normal(0.0, 0.3, n),
            "CMA": rng.normal(0.0, 0.3, n),
            "RF": ff3["RF"],
        }
    ).to_csv(factors_dir / "ff5_daily.csv", index=False)


def test_main_with_factors_dir_writes_metadata_and_factor_attribution(tmp_path, monkeypatch):
    config_path = _write_smoke_fixtures_with_factors_dir(tmp_path)
    _add_ff5_daily(config_path.parent / "factors")

    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_case_study.py",
            "--person", "buffett",
            "--config", str(config_path),
            "--output-dir", str(out_dir),
        ],
    )
    run_case_study_script.main()

    person_dir = out_dir / "buffett"
    written = pd.read_csv(person_dir / "metadata.csv", index_col=0)["value"]
    assert list(written.index) == _METADATA_ROWS

    app_config = load_config(config_path)
    expected = run_case_study(app_config, person_key="buffett").risk_free_rate
    written_rf = float(written["risk_free_rate"])
    assert written_rf == pytest.approx(expected)
    assert written_rf > 0.0

    assert (person_dir / "factor_attribution.csv").exists()


def test_main_without_factors_dir_zero_rf_and_no_factor_attribution(tmp_path, monkeypatch):
    _, _, config_path = _write_smoke_fixtures(tmp_path)

    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_case_study.py",
            "--person", "buffett",
            "--config", str(config_path),
            "--output-dir", str(out_dir),
        ],
    )
    run_case_study_script.main()

    person_dir = out_dir / "buffett"
    written = pd.read_csv(person_dir / "metadata.csv", index_col=0)["value"]
    assert list(written.index) == _METADATA_ROWS
    assert float(written["risk_free_rate"]) == 0.0

    assert not (person_dir / "factor_attribution.csv").exists()
