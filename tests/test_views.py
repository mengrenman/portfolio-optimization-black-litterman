from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from portfolio_bl.config import load_config
from portfolio_bl.data.prices import load_prices_csv, to_return_matrix
from portfolio_bl.models._numeric import mean_diagonal, relative_ridge
from portfolio_bl.models.black_litterman import (
    black_litterman_posterior,
    diagonal_omega_from_confidence,
    implied_equilibrium_returns,
)
from portfolio_bl.models.views import View, build_view_matrices
from portfolio_bl.pipeline import run_case_study

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "XOM"]
COV = np.array([[0.04, 0.01, 0.00], [0.01, 0.05, 0.01], [0.00, 0.01, 0.03]])
COV_DF = pd.DataFrame(COV, index=TICKERS, columns=TICKERS)
W_MKT = pd.Series([0.5, 0.3, 0.2], index=TICKERS)


# ---------------------------------------------------------------------------
# View
# ---------------------------------------------------------------------------


def test_view_normalises_tickers_and_coerces_numbers() -> None:
    view = View(assets={" aapl ": "1"}, annual_return="0.08", confidence="0.5")
    assert view.assets == {"AAPL": 1.0}
    assert view.annual_return == 0.08
    assert view.confidence == 0.5
    assert view.tickers == ("AAPL",)
    assert not view.is_relative


def test_view_relative_flag_and_describe() -> None:
    view = View(assets={"CVX": 1.0, "OXY": -1.0}, annual_return=0.02)
    assert view.is_relative
    assert view.confidence is None
    assert "CVX:+1" in view.describe() and "OXY:-1" in view.describe()
    assert "2.00%" in view.describe()

    labelled = View(assets={"CVX": 1.0}, annual_return=0.02, label="CVX bull")
    assert labelled.describe() == "CVX bull"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"assets": {}, "annual_return": 0.05}, "at least one asset"),
        ({"assets": ["AAPL"], "annual_return": 0.05}, "non-empty mapping"),
        ({"assets": {"AAPL": 0.0, "MSFT": 0.0}, "annual_return": 0.05}, "cannot all be zero"),
        ({"assets": {"AAPL": "x"}, "annual_return": 0.05}, "must be numeric"),
        ({"assets": {"AAPL": float("inf")}, "annual_return": 0.05}, "must be finite"),
        ({"assets": {"aapl": 1.0, "AAPL": 1.0}, "annual_return": 0.05}, "more than once"),
        ({"assets": {"AAPL": 1.0}, "annual_return": "high"}, "annual_return must be numeric"),
        ({"assets": {"AAPL": 1.0}, "annual_return": float("nan")}, "annual_return must be finite"),
        ({"assets": {"AAPL": 1.0}, "annual_return": 0.05, "confidence": 0.0}, r"in \(0, 1\]"),
        ({"assets": {"AAPL": 1.0}, "annual_return": 0.05, "confidence": 1.5}, r"in \(0, 1\]"),
    ],
)
def test_view_validation_errors(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        View(**kwargs)


# ---------------------------------------------------------------------------
# build_view_matrices
# ---------------------------------------------------------------------------


def test_view_rejects_non_string_ticker() -> None:
    """YAML resolves bare ON/OFF/YES/NO to booleans; they must not become 'TRUE'/'FALSE'."""
    parsed = yaml.safe_load("assets: {ON: 1.0, AAPL: -1.0}")
    with pytest.raises(ValueError, match="must be a string"):
        View(assets=parsed["assets"], annual_return=0.05)

    quoted = yaml.safe_load("assets: {'ON': 1.0, AAPL: -1.0}")
    assert View(assets=quoted["assets"], annual_return=0.05).assets == {"ON": 1.0, "AAPL": -1.0}


def test_build_view_matrices_absolute_and_relative() -> None:
    views = [
        View(assets={"AAPL": 1.0}, annual_return=0.252, confidence=0.9),
        View(assets={"MSFT": 1.0, "XOM": -1.0}, annual_return=0.0252),
    ]
    built = build_view_matrices(views, TICKERS, periods_per_year=252, default_confidence=0.4)

    np.testing.assert_array_equal(built.p_matrix, [[1.0, 0.0, 0.0], [0.0, 1.0, -1.0]])
    np.testing.assert_allclose(built.q_views, [0.001, 0.0001])
    np.testing.assert_allclose(built.confidences, [0.9, 0.4])
    assert built.kept == (0, 1)
    assert built.dropped == ()
    assert built.n_views == 2


def test_build_view_matrices_respects_column_order() -> None:
    view = View(assets={"XOM": 2.0, "AAPL": -1.0}, annual_return=0.12)
    built = build_view_matrices([view], ["MSFT", "XOM", "AAPL"], 12, 0.5)
    np.testing.assert_array_equal(built.p_matrix, [[0.0, 2.0, -1.0]])
    np.testing.assert_allclose(built.q_views, [0.01])


def test_build_view_matrices_drops_views_with_missing_ticker() -> None:
    views = [
        View(assets={"AAPL": 1.0}, annual_return=0.05),
        View(assets={"AAPL": 1.0, "NVDA": -1.0}, annual_return=0.02),
    ]
    built = build_view_matrices(views, TICKERS, 12, 0.5)
    assert built.kept == (0,)
    assert built.dropped == (1,)
    assert built.p_matrix.shape == (1, 3)
    assert built.n_views == 1


def test_build_view_matrices_with_no_views_is_empty() -> None:
    built = build_view_matrices([], TICKERS, 12, 0.5)
    assert built.p_matrix.shape == (0, 3)
    assert built.q_views.shape == (0,)
    assert built.confidences.shape == (0,)
    assert built.n_views == 0


def test_build_view_matrices_rejects_non_positive_periods() -> None:
    with pytest.raises(ValueError, match="periods_per_year"):
        build_view_matrices([], TICKERS, 0, 0.5)


# ---------------------------------------------------------------------------
# diagonal_omega_from_confidence with per-view confidences
# ---------------------------------------------------------------------------


def test_omega_accepts_per_view_confidence_vector() -> None:
    p = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, -1.0]])
    omega = diagonal_omega_from_confidence(COV, p, tau=0.05, confidence=[0.9, 0.3])

    assert omega.shape == (2, 2)
    projected = np.diag(p @ (0.05 * COV) @ p.T)
    expected = projected * np.array([(1 - 0.9) / 0.9, (1 - 0.3) / 0.3])
    np.testing.assert_allclose(np.diag(omega), expected)
    assert omega[0, 1] == 0.0 and omega[1, 0] == 0.0


def test_omega_scalar_matches_broadcast_vector() -> None:
    p = np.eye(3)
    scalar = diagonal_omega_from_confidence(COV, p, tau=0.05, confidence=0.6)
    vector = diagonal_omega_from_confidence(COV, p, tau=0.05, confidence=[0.6, 0.6, 0.6])
    np.testing.assert_allclose(scalar, vector)


def test_omega_rejects_wrong_length_confidence() -> None:
    with pytest.raises(ValueError, match="one confidence per view"):
        diagonal_omega_from_confidence(COV, np.eye(3), tau=0.05, confidence=[0.5, 0.5])


def test_omega_with_no_views_is_empty() -> None:
    omega = diagonal_omega_from_confidence(COV, np.zeros((0, 3)), tau=0.05, confidence=0.65)
    assert omega.shape == (0, 0)


# ---------------------------------------------------------------------------
# black_litterman_posterior with a general pick matrix
# ---------------------------------------------------------------------------


def test_posterior_with_no_views_returns_prior() -> None:
    pi = np.array([0.05, 0.04, 0.03])
    mu, cov = black_litterman_posterior(pi, COV, np.zeros((0, 3)), np.zeros(0), tau=0.05)
    np.testing.assert_allclose(mu, pi)
    # The ridge is relative to the matrix scale, not an absolute constant.
    expected = 1.05 * (COV + np.diag(relative_ridge(COV, 1e-6)))
    np.testing.assert_allclose(cov, expected)


def _realised_confidence(cov: np.ndarray, pi: np.ndarray, row: np.ndarray, c: float) -> float:
    """How far the posterior actually moves from the prior toward the view."""
    p = row.reshape(1, -1)
    base = float((p @ pi)[0])
    q = np.array([base + 0.01])
    omega = diagonal_omega_from_confidence(cov, p, tau=0.05, confidence=c)
    mu, _ = black_litterman_posterior(pi, cov, p, q, tau=0.05, omega=omega)
    return (float((p @ mu)[0]) - base) / (q[0] - base)


@pytest.mark.parametrize("c", [0.1, 0.3, 0.65, 0.9, 0.99, 1.0])
def test_realised_confidence_matches_configured_confidence(c: float) -> None:
    """The configured confidence must be delivered regardless of asset volatility.

    A fixed absolute ridge on Omega used to make the realised value depend on
    how volatile the asset was, spanning 0.14 to 0.64 for a configured 0.65.

    The tolerance is 1e-4 because the relative ridge is itself a 1e-6
    perturbation; with ridge=0 the identity holds to machine precision. What
    matters is that the realised value no longer depends on the asset.
    """
    # Variances spanning six orders of magnitude, as bond funds and a meme stock would.
    cov = np.diag([1e-8, 1e-6, 1e-4, 1e-2]).astype(float)
    pi = np.array([0.001, 0.002, 0.003, 0.004])
    realised = [_realised_confidence(cov, pi, np.eye(4)[i], c) for i in range(4)]
    for value in realised:
        assert value == pytest.approx(c, abs=1e-4)
    # The point of the fix: identical across assets, not merely close to c.
    assert max(realised) - min(realised) < 1e-9


def test_realised_confidence_holds_for_relative_views() -> None:
    """Relative views between similar assets were the worst-affected case."""
    cov = np.array(
        [[1.0e-8, 0.9e-8, 0.0], [0.9e-8, 1.0e-8, 0.0], [0.0, 0.0, 1.0e-2]], dtype=float
    )
    pi = np.array([0.001, 0.001, 0.004])
    row = np.array([1.0, -1.0, 0.0])
    for c in (0.3, 0.65, 1.0):
        assert _realised_confidence(cov, pi, row, c) == pytest.approx(c, abs=1e-4)


def _stacked_realised(cov: np.ndarray, pi: np.ndarray, q: np.ndarray, c: float) -> np.ndarray:
    """Per-asset realised fraction when one view per asset is active at once."""
    p = np.eye(len(pi))
    omega = diagonal_omega_from_confidence(cov, p, tau=0.05, confidence=c)
    mu, _ = black_litterman_posterior(pi, cov, p, q, tau=0.05, omega=omega)
    return (mu - pi) / (q - pi)


def test_per_view_calibration_requires_uncorrelated_view_projections() -> None:
    """Per-view calibration holds exactly when P(tau*Sigma)P' is diagonal.

    Omega is diagonal by construction, so each view realises its configured
    fraction only when the views' projections are uncorrelated under the prior.
    That covers a single view, and the identity block over a diagonal Sigma. It
    fails as soon as two view rows touch the same asset, even when Sigma itself
    is diagonal, and it fails for the identity block over a correlated Sigma,
    which is the shipped default. The posterior is pooling information across
    correlated evidence, which is ordinary Bayesian updating rather than a
    defect, but it is not a per-view guarantee. This test pins all four corners
    of that boundary so the limitation cannot be documented away again.
    """
    correlated = np.array(
        [[4.0e-4, 3.4e-4, 1.0e-5], [3.4e-4, 4.0e-4, 1.0e-5], [1.0e-5, 1.0e-5, 9.0e-4]]
    )
    pi = np.array([3.0e-4, 3.0e-4, 5.0e-4])
    q = pi + np.array([2.0e-4, -2.0e-4, 1.0e-4])
    c = 0.65

    # Correlated Sigma: at least one asset misses its configured fraction badly.
    spread = _stacked_realised(correlated, pi, q, c)
    assert np.abs(spread - c).max() > 0.2, (
        "expected stacked views over a correlated Sigma to break per-asset calibration"
    )

    # The same stack over a diagonal Sigma is exactly calibrated, which isolates
    # the off-diagonal terms as the cause rather than the regularisation.
    diagonal = np.diag(np.diag(correlated))
    np.testing.assert_allclose(_stacked_realised(diagonal, pi, q, c), c, atol=1e-4)

    # And a single view on the same correlated Sigma is still exact.
    for i in range(3):
        row = np.zeros(3)
        row[i] = 1.0
        assert _realised_confidence(correlated, pi, row, c) == pytest.approx(c, abs=1e-4)

    # A diagonal Sigma is NOT sufficient on its own: overlapping pick rows
    # correlate the view projections even when the assets are uncorrelated.
    overlapping = np.vstack([np.eye(3), [1.0, -1.0, 0.0]])
    omega = diagonal_omega_from_confidence(diagonal, overlapping, tau=0.05, confidence=c)
    q_ov = overlapping @ q
    mu_ov, _ = black_litterman_posterior(
        pi, diagonal, overlapping, q_ov, tau=0.05, omega=omega
    )
    realised_ov = (overlapping @ mu_ov - overlapping @ pi) / (q_ov - overlapping @ pi)
    assert np.abs(realised_ov - c).max() > 0.1, (
        "overlapping pick rows over a diagonal Sigma should break per-view calibration"
    )


# ---------------------------------------------------------------------------
# relative_ridge degenerate paths
# ---------------------------------------------------------------------------


def test_relative_ridge_scales_each_entry_by_its_own_variance() -> None:
    cov = np.diag([4e-4, 9e-4])
    np.testing.assert_allclose(relative_ridge(cov, 1e-6), [4e-10, 9e-10])


@pytest.mark.parametrize(
    ("diag", "expected_unusable_scale"),
    [
        ([4e-4, 0.0, 3e-4], (4e-4 + 0.0 + 3e-4) / 3.0),
        ([4e-4, np.nan, 3e-4], 1.0),
        ([4e-4, np.inf, 3e-4], 1.0),
        ([0.0, 0.0, 0.0], 1.0),
    ],
    ids=["one-zero", "nan", "inf", "all-zero"],
)
def test_relative_ridge_fallback_for_unusable_entries(
    diag: list[float], expected_unusable_scale: float
) -> None:
    """An unusable entry takes the mean of ALL absolute diagonal entries.

    When that mean is itself unusable (all-zero, or any NaN/inf present) the
    helper falls back to an absolute ridge. Pinning this exactly, because the
    README describes it and a plausible simplification of relative_ridge leaves
    every other test green while making an all-zero covariance raise.
    """
    cov = np.diag(np.asarray(diag, dtype=float))
    result = relative_ridge(cov, 1e-6)
    unusable = ~(np.isfinite(np.abs(np.diag(cov))) & (np.abs(np.diag(cov)) > 0.0))
    assert result[unusable] == pytest.approx(expected_unusable_scale * 1e-6)
    assert np.isfinite(result).all()
    assert (result > 0).all()


@pytest.mark.parametrize(
    "cov",
    [
        np.zeros((3, 3)),
        np.array([[4e-4, 4e-4, 0.0], [4e-4, 4e-4, 0.0], [0.0, 0.0, 3e-4]]),
        np.diag([4e-4, 0.0, 3e-4]),
    ],
    ids=["all-zero", "exactly-singular", "one-zero-variance"],
)
def test_long_only_weights_survive_degenerate_covariance(cov: np.ndarray) -> None:
    """The solver uses np.linalg.solve, so the ridge must keep cov invertible."""
    from portfolio_bl.models.mean_variance import long_only_markowitz_weights

    mu = pd.Series([1e-4, 5e-5, 2e-5], index=TICKERS)
    weights = long_only_markowitz_weights(
        mu, pd.DataFrame(cov, index=TICKERS, columns=TICKERS)
    )
    assert np.isfinite(weights.to_numpy()).all()
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()


def test_mean_diagonal_fallback() -> None:
    assert mean_diagonal(np.zeros((2, 2))) == 1.0
    assert mean_diagonal(np.zeros((2, 2)), fallback=7.0) == 7.0
    assert mean_diagonal(np.diag([2.0, 4.0])) == pytest.approx(3.0)
    assert mean_diagonal(np.empty((0, 0))) == 1.0


def test_realised_confidence_is_exact_without_the_ridge() -> None:
    """With the ridge disabled the calibration identity holds exactly."""
    cov = np.diag([1e-8, 1e-2]).astype(float)
    pi = np.array([0.001, 0.004])
    p = np.array([[1.0, 0.0]])
    for c in (0.25, 0.75):
        base = float((p @ pi)[0])
        q = np.array([base + 0.01])
        omega = diagonal_omega_from_confidence(cov, p, tau=0.05, confidence=c)
        mu, _ = black_litterman_posterior(pi, cov, p, q, tau=0.05, omega=omega, ridge=0.0)
        realised = (float((p @ mu)[0]) - base) / (q[0] - base)
        assert realised == pytest.approx(c, abs=1e-12)


def test_posterior_is_invariant_to_return_frequency() -> None:
    """Rescaling the covariance must not change the weights it implies."""
    from portfolio_bl.models.mean_variance import long_only_markowitz_weights

    base = np.array([[4e-4, 1e-4, 0.0], [1e-4, 5e-4, 1e-4], [0.0, 1e-4, 3e-4]])
    weights = []
    for factor in (1.0, 21.0, 252.0):
        cov = base * factor
        pi = implied_equilibrium_returns(
            pd.DataFrame(cov, index=TICKERS, columns=TICKERS),
            pd.Series([0.5, 0.3, 0.2], index=TICKERS),
            risk_aversion=2.5,
        )
        p = np.eye(3)
        q = pi + np.array([1e-4, 0.0, -1e-4]) * factor
        omega = diagonal_omega_from_confidence(cov, p, tau=0.05, confidence=0.65)
        mu, post = black_litterman_posterior(pi, cov, p, q, tau=0.05, omega=omega)
        weights.append(
            long_only_markowitz_weights(
                pd.Series(mu, index=TICKERS),
                pd.DataFrame(post, index=TICKERS, columns=TICKERS),
            ).to_numpy()
        )
    np.testing.assert_allclose(weights[1], weights[0], rtol=1e-9)
    np.testing.assert_allclose(weights[2], weights[0], rtol=1e-9)


@pytest.mark.parametrize(
    "cov",
    [
        np.array([[4e-4, 4e-4, 0.0], [4e-4, 4e-4, 0.0], [0.0, 0.0, 3e-4]]),
        np.zeros((3, 3)),
    ],
    ids=["exactly-singular", "all-zero"],
)
def test_posterior_survives_degenerate_covariance(cov: np.ndarray) -> None:
    """The relative ridge must still keep a singular covariance invertible."""
    pi = np.array([1e-4, 5e-5, 2e-5])
    p = np.eye(3)
    omega = diagonal_omega_from_confidence(cov, p, tau=0.05, confidence=0.65)
    mu, post = black_litterman_posterior(pi, cov, p, pi * 1.5, tau=0.05, omega=omega)
    assert np.isfinite(mu).all()
    assert np.isfinite(post).all()


def test_omega_stays_invertible_at_full_confidence() -> None:
    """(1-c)/c is exactly zero at c=1, so Omega needs a relative floor."""
    cov = np.diag([1e-8, 1e-2]).astype(float)
    omega = diagonal_omega_from_confidence(cov, np.eye(2), tau=0.05, confidence=1.0)
    assert (np.diag(omega) > 0).all()
    # The floor is proportional to each view's own projected variance, so the
    # ratio between the two entries tracks the covariance, not a constant.
    assert np.diag(omega)[1] / np.diag(omega)[0] == pytest.approx(1e6, rel=1e-6)


def test_posterior_relative_view_moves_spread_toward_view() -> None:
    pi = implied_equilibrium_returns(COV_DF, W_MKT, risk_aversion=2.5)
    p = np.array([[1.0, -1.0, 0.0]])
    q = np.array([0.03])
    omega = diagonal_omega_from_confidence(COV, p, tau=0.05, confidence=0.7)

    mu, cov = black_litterman_posterior(pi, COV, p, q, tau=0.05, omega=omega)

    prior_spread = pi[0] - pi[1]
    posterior_spread = mu[0] - mu[1]
    assert prior_spread < posterior_spread < 0.03
    assert cov.shape == (3, 3)
    assert np.isfinite(mu).all()


def test_posterior_rejects_mismatched_pick_matrix() -> None:
    pi = np.array([0.05, 0.04, 0.03])
    with pytest.raises(ValueError, match="p_matrix must have shape"):
        black_litterman_posterior(pi, COV, np.eye(2), np.zeros(2), tau=0.05)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path, backtest: dict, case: dict) -> Path:
    config = {
        "data": {
            "disclosures_path": str(tmp_path / "d.csv"),
            "prices_path": str(tmp_path / "p.csv"),
        },
        "backtest": backtest,
        "case_studies": {"buffett": {"person_label": "Buffett", **case}},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_load_config_parses_views(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        backtest={},
        case={
            "disclosure_aliases": ["buffett"],
            "views": [
                {"label": "AAPL bull", "assets": {"aapl": 1}, "annual_return": 0.08, "confidence": 0.6},
                {"assets": {"CVX": 1.0, "OXY": -1.0}, "annual_return": 0.02},
            ],
        },
    )
    cfg = load_config(path)

    views = cfg.case_studies["buffett"].views
    assert len(views) == 2
    assert views[0].label == "AAPL bull"
    assert views[0].assets == {"AAPL": 1.0}
    assert views[0].confidence == 0.6
    assert views[1].confidence is None
    assert views[1].is_relative
    assert cfg.backtest.use_sample_mean_views is True


def test_load_config_without_views_is_empty(tmp_path: Path) -> None:
    path = _write_config(tmp_path, backtest={}, case={"disclosure_aliases": ["buffett"]})
    assert load_config(path).case_studies["buffett"].views == ()


@pytest.mark.parametrize(
    ("views", "match"),
    [
        ({"assets": {"AAPL": 1.0}}, "'views' must be a list"),
        (["not a mapping"], "view 1: each view must be a mapping"),
        ([{"assets": {"AAPL": 1.0}}], r"view 1: missing required key\(s\): annual_return"),
        (
            [{"assets": {"AAPL": 1.0}, "annual_return": 0.1}, {"assets": {}, "annual_return": 0.1}],
            "Case study 'buffett' view 2: .*at least one asset",
        ),
    ],
)
def test_load_config_view_errors_name_case_and_position(
    tmp_path: Path, views: object, match: str
) -> None:
    path = _write_config(tmp_path, backtest={}, case={"disclosure_aliases": ["buffett"], "views": views})
    with pytest.raises(ValueError, match=match):
        load_config(path)


def test_load_config_rejects_unknown_view_key(tmp_path: Path) -> None:
    """A misspelled 'confidence' must not silently fall back to the global default."""
    path = _write_config(
        tmp_path,
        backtest={},
        case={
            "disclosure_aliases": ["buffett"],
            "views": [{"assets": {"AAPL": 1.0}, "annual_return": 0.08, "confidnece": 0.99}],
        },
    )
    with pytest.raises(ValueError, match=r"view 1: unknown key\(s\): confidnece"):
        load_config(path)


def test_load_config_accepts_non_string_case_study_key(tmp_path: Path) -> None:
    """A YAML key such as 2024: parses as an int and must not crash the loader."""
    config = {
        "data": {"disclosures_path": str(tmp_path / "d.csv"), "prices_path": str(tmp_path / "p.csv")},
        "backtest": {},
        "case_studies": {2024: {"disclosure_aliases": ["x"]}},
    }
    path = tmp_path / "int_key.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    cfg = load_config(path)
    assert cfg.case_studies["2024"].person_label == "2024"

    bad = {**config, "case_studies": {2024: {"views": [{"assets": {}, "annual_return": 0.1}]}}}
    bad_path = tmp_path / "int_key_bad.yaml"
    bad_path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="Case study '2024' view 1"):
        load_config(bad_path)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(False, False), (True, True), ("no", False), ("yes", True), ("off", False)],
)
def test_load_config_use_sample_mean_views_flag(tmp_path: Path, raw: object, expected: bool) -> None:
    path = _write_config(
        tmp_path, backtest={"use_sample_mean_views": raw}, case={"disclosure_aliases": ["buffett"]}
    )
    assert load_config(path).backtest.use_sample_mean_views is expected


def test_load_config_rejects_non_boolean_flag(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path, backtest={"use_sample_mean_views": "maybe"}, case={"disclosure_aliases": ["buffett"]}
    )
    with pytest.raises(ValueError, match="must be a boolean"):
        load_config(path)


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def _write_pipeline_config(
    tmp_path: Path,
    *,
    views: list[dict] | None = None,
    use_sample_mean_views: bool = True,
    lookback: int = 6,
    xom_first_month: int = 0,
    seed: int = 42,
) -> Path:
    """Write a seeded three-asset monthly fixture and return the config path.

    ``xom_first_month`` delays the first XOM price to simulate a ticker with
    no history early in the sample.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-31", periods=36, freq="ME")
    rows = []
    for ticker in TICKERS:
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.008, 0.06, len(dates))))
        start = xom_first_month if ticker == "XOM" else 0
        rows.extend(
            {"date": date, "ticker": ticker, "close": price}
            for date, price in list(zip(dates, prices))[start:]
        )
    pd.DataFrame(rows).to_csv(tmp_path / "p.csv", index=False)

    pd.DataFrame(
        {
            "person": ["Buffett"] * 3,
            "as_of_date": ["2023-12-31"] * 3,
            "ticker": TICKERS,
            "value_usd": [50.0, 30.0, 20.0],
        }
    ).to_csv(tmp_path / "d.csv", index=False)

    case: dict = {"disclosure_aliases": ["buffett"]}
    if views is not None:
        case["views"] = views
    return _write_config(
        tmp_path,
        backtest={
            "lookback_periods": lookback,
            "rebalance_frequency": "ME",
            "risk_aversion": 2.5,
            "tau": 0.05,
            "view_confidence": 0.65,
            "use_sample_mean_views": use_sample_mean_views,
        },
        case=case,
    )


def _bl_weights(config_path: Path) -> pd.DataFrame:
    result = run_case_study(load_config(config_path), person_key="buffett")
    return result.strategy_results["black_litterman"].weight_history


def test_pipeline_without_any_views_tracks_disclosed_weights(tmp_path: Path) -> None:
    """With no views the posterior is the prior, so BL reproduces the disclosed weights."""
    config_path = _write_pipeline_config(tmp_path, use_sample_mean_views=False, lookback=12)
    result = run_case_study(load_config(config_path), person_key="buffett")

    bl = result.strategy_results["black_litterman"].weight_history
    disclosed = result.strategy_results["disclosed"].weight_history
    # Two relative ridge terms perturb the solve slightly on short windows.
    np.testing.assert_allclose(bl.to_numpy(), disclosed.to_numpy(), atol=1e-2)

    # Sanity check that the tolerance is discriminating: sample-mean views move BL away.
    with_sample = _bl_weights(_write_pipeline_config(tmp_path, use_sample_mean_views=True, lookback=12))
    assert (with_sample - disclosed).abs().max().max() > 0.05


@pytest.mark.parametrize("seed", [11, 42, 40])
def test_pipeline_absolute_view_raises_target_weight(tmp_path: Path, seed: int) -> None:
    """A bullish view must lift its target's weight, whatever the sample path.

    The size of the lift depends on the fixture: when the sample-mean views are
    also active they may already favour XOM, leaving little headroom. Only the
    explicit-views-only branch gets a magnitude threshold; the stacked branch
    asserts the direction, which is what the feature actually guarantees. The
    seeds include two (11, 40) whose stacked lift is under 10 points.
    """
    bullish = [{"label": "XOM bull", "assets": {"XOM": 1.0}, "annual_return": 0.30, "confidence": 0.95}]
    for use_sample_mean_views in (True, False):
        base = _bl_weights(
            _write_pipeline_config(tmp_path, use_sample_mean_views=use_sample_mean_views, seed=seed)
        )
        with_view = _bl_weights(
            _write_pipeline_config(
                tmp_path, views=bullish, use_sample_mean_views=use_sample_mean_views, seed=seed
            )
        )
        lift = with_view["XOM"].mean() - base["XOM"].mean()
        if use_sample_mean_views:
            assert lift > 0.0, f"stacked view did not lift XOM at seed {seed} (lift {lift:.4f})"
        else:
            assert lift > 0.10, f"explicit-only view lifted XOM by only {lift:.4f} at seed {seed}"
        assert np.allclose(with_view.sum(axis=1), 1.0)


def test_pipeline_view_does_not_touch_other_strategies(tmp_path: Path) -> None:
    bullish = [{"assets": {"XOM": 1.0}, "annual_return": 0.30, "confidence": 0.95}]
    base = run_case_study(load_config(_write_pipeline_config(tmp_path)), person_key="buffett")
    with_view = run_case_study(
        load_config(_write_pipeline_config(tmp_path, views=bullish)), person_key="buffett"
    )
    for name in ("disclosed", "mean_variance"):
        pd.testing.assert_frame_equal(
            base.strategy_results[name].weight_history,
            with_view.strategy_results[name].weight_history,
        )


def test_pipeline_relative_view_widens_spread(tmp_path: Path) -> None:
    relative = [{"assets": {"AAPL": 1.0, "MSFT": -1.0}, "annual_return": 0.40, "confidence": 0.95}]
    base = _bl_weights(_write_pipeline_config(tmp_path, use_sample_mean_views=False))
    with_view = _bl_weights(
        _write_pipeline_config(tmp_path, views=relative, use_sample_mean_views=False)
    )
    base_spread = (base["AAPL"] - base["MSFT"]).mean()
    view_spread = (with_view["AAPL"] - with_view["MSFT"]).mean()
    assert view_spread > base_spread + 0.10


def test_pipeline_view_outside_universe_warns_once_and_runs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    views = [{"label": "NVDA bull", "assets": {"NVDA": 1.0}, "annual_return": 0.10}]
    config_path = _write_pipeline_config(tmp_path, views=views)
    with caplog.at_level(logging.WARNING, logger="portfolio_bl.pipeline"):
        result = run_case_study(load_config(config_path), person_key="buffett")

    messages = [r.getMessage() for r in caplog.records if "NVDA bull" in r.getMessage()]
    assert len(messages) == 1
    assert "not in the universe" in messages[0]
    assert not result.summary.empty


def test_pipeline_view_on_late_starting_ticker_is_skipped_then_applied(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A view waits for a complete lookback window, not merely for the first trade."""
    bullish = [{"label": "XOM bull", "assets": {"XOM": 1.0}, "annual_return": 0.30, "confidence": 0.95}]
    config_path = _write_pipeline_config(
        tmp_path, views=bullish, use_sample_mean_views=False, xom_first_month=12
    )
    with caplog.at_level(logging.WARNING, logger="portfolio_bl.pipeline"):
        bl = _bl_weights(config_path)

    skipped = [r.getMessage() for r in caplog.records if "XOM bull" in r.getMessage()]
    assert len(skipped) == 1
    assert "not applied from" in skipped[0]
    assert "complete lookback window" in skipped[0]

    # No XOM data in the early windows -> zero weight; once a full lookback window
    # exists the bullish view dominates because it is the only view.
    assert bl["XOM"].iloc[0] == 0.0
    assert bl["XOM"].iloc[-1] > 0.3

    # The view stays inactive for further rebalances after XOM's first return, because
    # the estimator drops tickers whose lookback window is incomplete.
    returns = to_return_matrix(load_prices_csv(load_config(config_path).prices_path))
    first_return = returns["XOM"].first_valid_index()
    first_active = bl.index[(bl["XOM"] > 0).argmax()]
    assert first_active > first_return
