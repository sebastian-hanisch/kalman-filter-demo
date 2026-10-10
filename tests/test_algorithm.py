"""Einheiten des Kalman-Kerns gegen Handrechnung auf der Mini-Instanz (conftest.mini_state) und Randfälle."""

import numpy as np
import pytest

import kal_algorithm as A
import kal_constants as C
import kal_scenario as S


def test_cv_model_matches_hand_values():
    F, Q = A.cv_model(1.0, 1.0)
    assert F.tolist() == [[1.0, 1.0], [0.0, 1.0]]
    assert Q == pytest.approx(np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]]))
    _, Q2 = A.cv_model(2.0, 0.5)
    assert Q2 == pytest.approx(0.25 * np.array([[8 / 3, 2.0], [2.0, 2.0]]))


def test_predict_moves_state_and_grows_uncertainty(mini_state):
    x, P = mini_state
    F, Q = A.cv_model(1.0, 1.0)
    x1, P1 = A.predict(x, P, F, Q)
    assert x1.tolist() == [2.0, 2.0]
    assert P1 == pytest.approx(np.array([[7 / 3, 3 / 2], [3 / 2, 2.0]]))


def test_update_scalar_hand_example(mini_state):
    x, P = mini_state
    x1, P1, K, nu, S_ = A.update_scalar(x, P, 2.0, A.H_GPS, 1.0)
    assert S_ == pytest.approx(2.0) and nu == pytest.approx(2.0)
    assert K == pytest.approx([0.5, 0.0])
    assert x1 == pytest.approx([1.0, 2.0])
    assert P1 == pytest.approx(np.diag([0.5, 1.0]))


def test_update_never_increases_uncertainty():
    rng = np.random.default_rng(0)
    for _ in range(40):
        M = rng.standard_normal((2, 2))
        P = M @ M.T + 0.1 * np.eye(2)
        x1, P1, *_ = A.update_scalar(np.zeros(2), P, 1.0, A.H_GPS, 2.0)
        assert np.all(np.linalg.eigvalsh(P - P1) >= -1e-12)
        assert np.allclose(P1, P1.T)


def test_nees_hand_example():
    assert A.nees(np.array([2.0, 1.0]), np.diag([4.0, 1.0])) == pytest.approx(2.0)


def test_filter_without_measurements_only_predicts():
    z = np.array([5.0] + [np.nan] * 9)
    w = np.zeros(10)
    run = A.run_filter(z, w, 1.0, 0.3, 10.0, 0.3, False, 10.0, 5.0)
    assert not run.gps_used[1:].any() and np.isnan(run.k_gps[1:]).all()
    tr = [np.trace(run.P[k]) for k in range(10)]
    assert all(b > a for a, b in zip(tr, tr[1:]))


def test_wheel_sensor_removes_most_velocity_uncertainty_at_start():
    z = np.array([0.0, 10.0])
    w = np.array([10.0, 10.0])
    with_wheel = A.run_filter(z, w, 1.0, 0.3, 10.0, 0.3, True, 10.0, 5.0)
    without = A.run_filter(z, w, 1.0, 0.3, 10.0, 0.3, False, 10.0, 5.0)
    assert with_wheel.P[0, 1, 1] < 0.1 < without.P[0, 1, 1]


def test_steady_state_is_a_fixed_point():
    F, Q = A.cv_model(1.0, 0.3)
    P_prior, gains = A.steady_state(F, Q, [(A.H_GPS, 100.0)])
    x, P, K, _, _ = A.update_scalar(np.zeros(2), P_prior, 0.0, A.H_GPS, 100.0)
    P_next = F @ P @ F.T + Q
    assert P_next == pytest.approx(P_prior, rel=1e-8)
    assert K == pytest.approx(gains[0], rel=1e-8)


def test_simulation_is_reproducible_and_regler_scale_the_same_randomness():
    a = S.simulate(7, 10.0, 0.3, 20)
    b = S.simulate(7, 10.0, 0.3, 20)
    assert np.array_equal(a.s, b.s) and np.array_equal(a.z, b.z, equal_nan=True)
    c = S.simulate(7, 20.0, 0.3, 20)
    assert np.array_equal(a.s, c.s)                      # GPS-Rauschen ändert die Wahrheit nicht
    have = ~a.tunnel
    assert np.allclose((c.z - c.s)[have], 2 * (a.z - a.s)[have])


def test_tunnel_mask_and_nan_positions():
    for length in (0, 1, 20, 40):
        tr = S.simulate(1, 10.0, 0.3, length)
        assert tr.tunnel.sum() == length
        assert np.isnan(tr.z[tr.tunnel]).all() and not np.isnan(tr.z[~tr.tunnel]).any()
        assert not np.isnan(tr.w).any()
    assert S.simulate(1, 10.0, 0.3, 20).tunnel[C.TUNNEL_START]


def test_constants_are_consistent():
    assert C.TUNNEL_START + C.TUNNEL_MAX < C.ETA_STEP < C.T_STEPS
    assert C.N_BURN < C.TUNNEL_START
    assert set(C.FACTOR_CHOICES) >= {1.0}
