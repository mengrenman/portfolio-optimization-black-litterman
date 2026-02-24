from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for the rolling backtest engine.

    Attributes:
        lookback_periods: Number of historical periods used to estimate the
            covariance matrix and expected returns at each rebalance date.
        rebalance_frequency: Pandas offset alias for rebalancing (e.g. ``'ME'``
            for month-end, ``'QE'`` for quarter-end).
        risk_aversion: Risk-aversion coefficient λ used in the equilibrium
            return formula π = λ Σ w_mkt.
        tau: Scalar controlling the uncertainty of the prior distribution in
            the Black-Litterman model. Smaller values imply stronger trust in
            the equilibrium prior.
        view_confidence: Analyst confidence in the views expressed to the
            Black-Litterman model. Range (0, 1]; higher values reduce view
            uncertainty (Ω) and place more weight on the views relative to the
            equilibrium prior. Configurable via the ``backtest.view_confidence``
            YAML key.
    """

    lookback_periods: int = 12
    rebalance_frequency: str = "ME"
    risk_aversion: float = 2.5
    tau: float = 0.05
    view_confidence: float = 0.65


@dataclass(frozen=True)
class CaseStudyConfig:
    """Configuration for a single person's case study.

    Attributes:
        key: Unique identifier used as the CLI ``--person`` argument and as
            the output directory name.
        person_label: Human-readable label used in report titles and plots.
        disclosure_aliases: Lowercase name variants that identify this
            person's rows in the disclosures CSV.
    """

    key: str
    person_label: str
    disclosure_aliases: tuple[str, ...]


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration.

    Attributes:
        disclosures_path: Absolute path to the disclosures CSV.
        prices_path: Absolute path to the prices CSV.
        backtest: Backtest and model hyper-parameters.
        case_studies: Mapping from case-study key to its configuration.
    """

    disclosures_path: Path
    prices_path: Path
    backtest: BacktestConfig
    case_studies: dict[str, CaseStudyConfig]


def load_config(path: str | Path) -> AppConfig:
    """Load application configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A fully-populated :class:`AppConfig` instance.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the YAML is malformed, is not a top-level mapping, or
            contains no case studies.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Malformed YAML in configuration file {config_path}: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise ValueError(
            f"Configuration file {config_path} must contain a YAML mapping at the top level."
        )

    data_cfg = raw.get("data", {})
    bt_cfg = raw.get("backtest", {})
    case_cfg = raw.get("case_studies", {})

    backtest = BacktestConfig(
        lookback_periods=int(bt_cfg.get("lookback_periods", 12)),
        rebalance_frequency=str(bt_cfg.get("rebalance_frequency", "ME")),
        risk_aversion=float(bt_cfg.get("risk_aversion", 2.5)),
        tau=float(bt_cfg.get("tau", 0.05)),
        view_confidence=float(bt_cfg.get("view_confidence", 0.65)),
    )

    case_studies: dict[str, CaseStudyConfig] = {}
    for key, item in case_cfg.items():
        aliases = item.get("disclosure_aliases", [key])
        case_studies[str(key)] = CaseStudyConfig(
            key=str(key),
            person_label=str(item.get("person_label", key.title())),
            disclosure_aliases=tuple(str(a).strip().lower() for a in aliases),
        )

    if not case_studies:
        raise ValueError("No case studies found in config.")

    root = config_path.parent.parent
    disclosures_path = (
        root / data_cfg.get("disclosures_path", "data/raw/disclosures/disclosures.csv")
    ).resolve()
    prices_path = (
        root / data_cfg.get("prices_path", "data/raw/prices/prices.csv")
    ).resolve()

    logger.debug(
        "Loaded config: %d case studies, disclosures=%s, prices=%s",
        len(case_studies),
        disclosures_path,
        prices_path,
    )

    return AppConfig(
        disclosures_path=disclosures_path,
        prices_path=prices_path,
        backtest=backtest,
        case_studies=case_studies,
    )
