from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from portfolio_bl.models.views import View

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
            YAML key. Also the default confidence for explicit views that do
            not set their own.
        use_sample_mean_views: When ``True`` (default) the Black-Litterman
            strategy expresses one absolute view per asset equal to the
            lookback sample mean, on top of any explicit views. When ``False``
            only the explicit views configured per case study are used, and a
            case study with no views falls back to the equilibrium prior.
    """

    lookback_periods: int = 12
    rebalance_frequency: str = "ME"
    risk_aversion: float = 2.5
    tau: float = 0.05
    view_confidence: float = 0.65
    use_sample_mean_views: bool = True


@dataclass(frozen=True)
class CaseStudyConfig:
    """Configuration for a single person's case study.

    Attributes:
        key: Unique identifier used as the CLI ``--person`` argument and as
            the output directory name.
        person_label: Human-readable label used in report titles and plots.
        disclosure_aliases: Lowercase name variants that identify this
            person's rows in the disclosures CSV.
        views: Explicit Black-Litterman views for this case study, each one
            becoming a row of the pick matrix. Empty by default.
    """

    key: str
    person_label: str
    disclosure_aliases: tuple[str, ...]
    views: tuple[View, ...] = ()


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


def _parse_bool(value: object, key: str) -> bool:
    """Parse a YAML boolean, accepting common textual spellings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "on", "1"}:
            return True
        if lowered in {"false", "no", "off", "0"}:
            return False
    raise ValueError(f"'{key}' must be a boolean, got {value!r}.")


_VIEW_KEYS = frozenset({"assets", "annual_return", "confidence", "label"})


def _parse_views(case_key: str, raw_views: object) -> tuple[View, ...]:
    """Parse the optional ``views`` list of one case study into :class:`View`s.

    Each entry must be a mapping with ``assets`` (ticker → coefficient) and
    ``annual_return``, plus optional ``confidence`` and ``label``. Unknown keys
    are rejected so that a misspelled ``confidence`` cannot silently fall back
    to the global default. Errors are
    prefixed with the case-study key and the 1-based view position so the
    offending YAML block is easy to find.
    """
    if raw_views is None:
        return ()
    if not isinstance(raw_views, list):
        raise ValueError(f"Case study '{case_key}': 'views' must be a list of mappings.")  # noqa: TRY004 - loader raises ValueError

    views: list[View] = []
    for position, item in enumerate(raw_views, start=1):
        prefix = f"Case study '{case_key}' view {position}"
        if not isinstance(item, dict):
            raise ValueError(f"{prefix}: each view must be a mapping.")  # noqa: TRY004 - loader raises ValueError
        missing = [k for k in ("assets", "annual_return") if k not in item]
        if missing:
            raise ValueError(f"{prefix}: missing required key(s): {', '.join(missing)}.")
        unknown = sorted(str(k) for k in set(item) - _VIEW_KEYS)
        if unknown:
            raise ValueError(
                f"{prefix}: unknown key(s): {', '.join(unknown)}. "
                f"Valid keys: {', '.join(sorted(_VIEW_KEYS))}."
            )
        try:
            view = View(
                assets=item["assets"],
                annual_return=item["annual_return"],
                confidence=item.get("confidence"),
                label=item.get("label"),
            )
        except ValueError as exc:
            raise ValueError(f"{prefix}: {exc}") from exc
        views.append(view)
    return tuple(views)


def load_config(path: str | Path) -> AppConfig:
    """Load application configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A fully-populated :class:`AppConfig` instance.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the YAML is malformed, is not a top-level mapping,
            contains no case studies, or defines a malformed view.
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
        use_sample_mean_views=_parse_bool(
            bt_cfg.get("use_sample_mean_views", True), "backtest.use_sample_mean_views"
        ),
    )

    case_studies: dict[str, CaseStudyConfig] = {}
    for key, item in case_cfg.items():
        # YAML keys are not always strings (a bare 2024: parses as an int), so
        # normalise once instead of calling str-only methods on the raw key.
        key_str = str(key)
        aliases = item.get("disclosure_aliases", [key_str])
        case_studies[key_str] = CaseStudyConfig(
            key=key_str,
            person_label=str(item.get("person_label", key_str.title())),
            disclosure_aliases=tuple(str(a).strip().lower() for a in aliases),
            views=_parse_views(key_str, item.get("views")),
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
