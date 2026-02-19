from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class BacktestConfig:
    lookback_periods: int = 12
    rebalance_frequency: str = "ME"
    risk_aversion: float = 2.5
    tau: float = 0.05


@dataclass(frozen=True)
class CaseStudyConfig:
    key: str
    person_label: str
    disclosure_aliases: tuple[str, ...]


@dataclass(frozen=True)
class AppConfig:
    disclosures_path: Path
    prices_path: Path
    backtest: BacktestConfig
    case_studies: dict[str, CaseStudyConfig]



def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data_cfg = raw.get("data", {})
    bt_cfg = raw.get("backtest", {})
    case_cfg = raw.get("case_studies", {})

    backtest = BacktestConfig(
        lookback_periods=int(bt_cfg.get("lookback_periods", 12)),
        rebalance_frequency=str(bt_cfg.get("rebalance_frequency", "ME")),
        risk_aversion=float(bt_cfg.get("risk_aversion", 2.5)),
        tau=float(bt_cfg.get("tau", 0.05)),
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
    disclosures_path = (root / data_cfg.get("disclosures_path", "data/raw/disclosures/disclosures.csv")).resolve()
    prices_path = (root / data_cfg.get("prices_path", "data/raw/prices/prices.csv")).resolve()

    return AppConfig(
        disclosures_path=disclosures_path,
        prices_path=prices_path,
        backtest=backtest,
        case_studies=case_studies,
    )
