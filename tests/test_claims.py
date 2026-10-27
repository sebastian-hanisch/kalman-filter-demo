"""Jede Zahl aus dem README, nachgerechnet über dieselben Auswertungsfunktionen wie die App (Median bzw. Mittel über SWEEP_SEEDS).

Plattformrobust: Bänder statt exakter Werte (CI installiert neueste numpy/scipy)."""

import numpy as np
import pytest

import kal_constants as C
import kal_evaluation as ev


@pytest.fixture(scope="module")
def default_analysis():
    return ev.analyse(ev.Settings())


@pytest.fixture(scope="module")
def sigma_rows():
    return {r["sigma"]: r for r in ev.sweep_sigma(ev.Settings())}


@pytest.fixture(scope="module")
def assumption_rows():
    return ev.sweep_assumptions(ev.Settings())


@pytest.fixture(scope="module")
def tunnel_rows():
    return {r["tunnel"]: r for r in ev.sweep_tunnel(ev.Settings())}


def test_standard_track_numbers(default_analysis):
    a = default_analysis
    assert a.rmse_raw == pytest.approx(8.5, abs=0.2)
    assert a.rmse_kf == pytest.approx(1.1, abs=0.15)
    assert a.rmse_vel == pytest.approx(0.24, abs=0.05)
    err_raw, err_kf = ev.eta_errors(a)
    assert err_raw == pytest.approx(2.1, abs=0.1) and err_kf == pytest.approx(0.02, abs=0.01)
    assert a.nees_mean < 2.0      # die Standardspur wirkt „zu vorsichtig“ (Zufall dieser Spur, im README so benannt)


def test_stationary_gain_standard_setting(default_analysis):
    assert default_analysis.k_steady[0] == pytest.approx(0.0315, abs=0.001)
    assert default_analysis.k_steady[1] == pytest.approx(0.0015, abs=0.0001)


def test_filter_vs_raw_at_ten_meters(sigma_rows):
    r = sigma_rows[10.0]
    assert r["raw"] == pytest.approx(9.7, abs=0.4)
    assert r["kf_aus"] == pytest.approx(4.9, abs=0.4)
    assert r["kf_an"] == pytest.approx(1.6, abs=0.3)
    assert r["raw"] / r["kf_aus"] > 1.7 and r["raw"] / r["kf_an"] > 4.5


def test_filter_without_wheel_beats_raw_at_every_noise_level_but_by_a_smaller_factor_than_with(sigma_rows):
    for r in sigma_rows.values():
        assert r["kf_aus"] < r["raw"] and r["kf_an"] < r["kf_aus"]


def test_wrong_assumptions_make_the_filter_overconfident(assumption_rows):
    q = {r["factor"]: r for r in assumption_rows["q"]}
    rr = {r["factor"]: r for r in assumption_rows["r"]}
    assert q[0.1]["nees"] > 30 and q[0.1]["coverage"] < 0.9
    assert rr[0.1]["nees"] > 4 and rr[0.1]["coverage"] < 0.7
    assert q[1.0]["nees"] == pytest.approx(rr[1.0]["nees"])
    assert q[1.0]["nees"] == pytest.approx(1.8, abs=0.3)          # 40 feste Fahrten; das Orakel über 150 Fahrten liefert 2,0
    assert q[0.1]["nees"] == pytest.approx(61, abs=8) and q[0.1]["coverage"] == pytest.approx(0.80, abs=0.05)
    assert rr[0.1]["nees"] == pytest.approx(6.5, abs=1.0) and rr[0.1]["coverage"] == pytest.approx(0.59, abs=0.06)
    assert all(r["nees"] < 2.5 for r in assumption_rows["q"] if r["factor"] >= 1.0)


def test_too_large_assumptions_cost_accuracy_not_honesty(assumption_rows):
    q = {r["factor"]: r for r in assumption_rows["q"]}
    assert q[10.0]["rmse"] > 1.2 * q[1.0]["rmse"]
    assert q[1.0]["rmse"] == pytest.approx(1.6, abs=0.25) and q[10.0]["rmse"] == pytest.approx(2.3, abs=0.3)
    assert q[10.0]["nees"] == pytest.approx(1.5, abs=0.3)
    assert q[10.0]["coverage"] >= q[1.0]["coverage"] - 0.02


def test_tunnel_without_wheel_error_grows_with_the_band(tunnel_rows):
    errs = [tunnel_rows[L]["err_aus"] for L in (10, 20, 30, 40)]
    sds = [tunnel_rows[L]["sd_aus"] for L in (10, 20, 30, 40)]
    assert errs == sorted(errs) and sds == sorted(sds)
    assert tunnel_rows[40]["err_aus"] == pytest.approx(49, abs=8)
    assert tunnel_rows[40]["sd_aus"] == pytest.approx(57, abs=4)
    assert tunnel_rows[40]["err_an"] == pytest.approx(1.4, abs=0.5) and tunnel_rows[40]["sd_an"] == pytest.approx(2.7, abs=0.3)


def test_local_level_gain_values_in_the_table():
    assert [round(ev.local_level_gain(q), 4) for q in (0.01, 0.1, 1.0)] == [0.0951, 0.2702, 0.618]
    assert ev.local_level_gain(1.0) == pytest.approx((np.sqrt(5) - 1) / 2)


def test_sweep_uses_the_fixed_instances():
    assert len(C.SWEEP_SEEDS) == 40 and C.SWEEP_SEEDS[0] == 100000
