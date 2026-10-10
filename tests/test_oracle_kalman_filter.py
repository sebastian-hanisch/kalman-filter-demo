"""Orakel-Test kalman-filter-demo: der Kern gegen ANDERE Rechenwege.

1. stationäre Verstärkung/Kovarianz gegen scipy.linalg.solve_discrete_are (Riccati),
2. das Lokales-Niveau-Modell gegen die geschlossene Formel und gegen scipy,
3. der gefilterte Endzustand und seine Kovarianz gegen die Lösung des GANZEN Problems als gewichtetes Kleinste-Quadrate-System,
4. die Vorhersage-Kovarianz gegen die explizite Summe (matrix_power),
5. die Joseph-Form gegen die einfache Form,
6. Konsistenz: mittlere NEES über viele Fahrten gleich der Zustandsdimension,
7. die in der App angezeigten Kennzahlen gegen eine unabhängige Neuberechnung.
"""

import numpy as np
import pytest

import kal_algorithm as A
import kal_constants as C
import kal_evaluation as ev
import kal_scenario as S

scipy_linalg = pytest.importorskip("scipy.linalg")


def test_steady_state_matches_scipy_riccati_gps_only():
    for q in (0.05, 0.3, 0.8):
        for r in (4.0, 100.0, 900.0):
            F, Q = A.cv_model(1.0, q)
            H = A.H_GPS.reshape(1, 2)
            P_ref = scipy_linalg.solve_discrete_are(F.T, H.T, Q, np.array([[r]]))
            K_ref = (P_ref @ H.T / (H @ P_ref @ H.T + r)).ravel()
            P_prior, gains = A.steady_state(F, Q, [(A.H_GPS, r)])
            assert P_prior == pytest.approx(P_ref, rel=1e-6, abs=1e-9)
            assert gains[0] == pytest.approx(K_ref, rel=1e-6, abs=1e-12)


def test_steady_state_with_both_sensors_matches_scipy_joint_update():
    F, Q = A.cv_model(1.0, 0.3)
    H = np.eye(2)
    R = np.diag([100.0, 0.09])
    P_ref = scipy_linalg.solve_discrete_are(F.T, H.T, Q, R)
    P_prior, _ = A.steady_state(F, Q, [(A.H_GPS, 100.0), (A.H_WHEEL, 0.09)])
    assert P_prior == pytest.approx(P_ref, rel=1e-6, abs=1e-9)


def test_local_level_gain_matches_closed_form_and_scipy():
    for q in (0.001, 0.01, 0.1, 1.0, 10.0):
        P_ref = scipy_linalg.solve_discrete_are(np.array([[1.0]]), np.array([[1.0]]), np.array([[q]]), np.array([[1.0]]))
        alpha_ref = float(P_ref[0, 0] / (P_ref[0, 0] + 1.0))
        _, gains = A.steady_state(np.array([[1.0]]), np.array([[q]]), [(np.array([1.0]), 1.0)])
        assert gains[0][0] == pytest.approx(alpha_ref, rel=1e-6)
        assert ev.local_level_gain(q) == pytest.approx(alpha_ref, rel=1e-9)


def _batch_solution(z, w, q, sigma_z, sigma_w, use_wheel, v_mean, v_std):
    """Gleichung 'ganzes Problem': Unbekannte x_0..x_{T-1}, gewichtete kleinste Quadrate (Whitening), letzte Zustandsschätzung und Kovarianz."""
    T = len(z)
    F, Q = A.cv_model(1.0, q)
    Lq = np.linalg.cholesky(np.linalg.inv(Q))     # Q^-1 = Lq Lq^T
    rows, rhs = [], []

    def add(row_blocks, b, weight):
        row = np.zeros(2 * T)
        for t, blk in row_blocks:
            row[2 * t:2 * t + 2] += blk
        rows.append(weight * row)
        rhs.append(weight * b)

    add([(0, np.array([1.0, 0.0]))], z[0], 1 / sigma_z)
    add([(0, np.array([0.0, 1.0]))], v_mean, 1 / v_std)
    for t in range(T - 1):
        for j in range(2):
            wj = Lq[:, j]
            # r = x_{t+1} - F x_t ; gewichtet mit Lq^T
            coeff_next = wj
            coeff_cur = -(F.T @ wj)
            add([(t + 1, coeff_next), (t, coeff_cur)], 0.0, 1.0)
    for t in range(1, T):
        if not np.isnan(z[t]):
            add([(t, A.H_GPS)], z[t], 1 / sigma_z)
    if use_wheel:
        for t in range(T):
            add([(t, A.H_WHEEL)], w[t], 1 / sigma_w)
    M, b = np.array(rows), np.array(rhs)
    x_hat = np.linalg.lstsq(M, b, rcond=None)[0]
    cov = np.linalg.inv(M.T @ M)
    return x_hat[-2:], cov[-2:, -2:]


def test_filter_equals_batch_least_squares():
    for seed in range(6):
        for wheel in (False, True):
            tr = S.simulate(seed, 10.0, 0.3, 8)
            z, w = tr.z[:40].copy(), tr.w[:40]
            z[15:23] = np.nan
            run = A.run_filter(z, w, 1.0, 0.3, 10.0, C.SIGMA_WHEEL, wheel, C.V_START_MEAN, C.V_PRIOR_STD)
            x_b, P_b = _batch_solution(z, w, 0.3, 10.0, C.SIGMA_WHEEL, wheel, C.V_START_MEAN, C.V_PRIOR_STD)
            assert run.x[-1] == pytest.approx(x_b, rel=1e-6, abs=1e-6), (seed, wheel)
            assert run.P[-1] == pytest.approx(P_b, rel=1e-6, abs=1e-9), (seed, wheel)


def test_prediction_covariance_matches_explicit_sum():
    F, Q = A.cv_model(1.0, 0.4)
    P0 = np.diag([25.0, 9.0])
    P = P0
    for k in range(1, 12):
        _, P = A.predict(np.zeros(2), P, F, Q)
        Fk = np.linalg.matrix_power(F, k)
        explicit = Fk @ P0 @ Fk.T + sum(np.linalg.matrix_power(F, j) @ Q @ np.linalg.matrix_power(F, j).T for j in range(k))
        assert P == pytest.approx(explicit, rel=1e-12)


def test_joseph_form_equals_standard_form_for_optimal_gain():
    rng = np.random.default_rng(3)
    for _ in range(30):
        M = rng.standard_normal((2, 2))
        P = M @ M.T + 0.2 * np.eye(2)
        r = float(rng.uniform(0.1, 5))
        _, P_j, K, _, _ = A.update_scalar(np.zeros(2), P, 0.0, A.H_GPS, r)
        P_std = (np.eye(2) - np.outer(K, A.H_GPS)) @ P
        assert P_j == pytest.approx(P_std, rel=1e-9, abs=1e-12)


def test_mean_nees_equals_state_dimension_over_many_tracks():
    for wheel in (False, True):
        vals = []
        for seed in range(150):
            tr = S.simulate(500 + seed, 10.0, 0.3, 20)
            run = A.run_filter(tr.z, tr.w, 1.0, 0.3, 10.0, C.SIGMA_WHEEL, wheel, C.V_START_MEAN, C.V_PRIOR_STD)
            e = np.c_[run.x[:, 0] - tr.s, run.x[:, 1] - tr.v]
            vals.append(np.mean([A.nees(e[k], run.P[k]) for k in range(C.N_BURN, C.T_STEPS)]))
        assert np.mean(vals) == pytest.approx(2.0, abs=0.25), wheel


def test_displayed_numbers_match_independent_recomputation():
    for seed in (3, 11, 40):
        s = ev.Settings(seed=seed, tunnel=20)
        a = ev.analyse(s)
        tr = S.simulate(seed, s.sigma_z, s.q, s.tunnel)
        have = ~np.isnan(tr.z)
        have[:C.N_BURN] = False
        assert a.rmse_raw == pytest.approx(np.sqrt(np.mean((tr.z[have] - tr.s[have]) ** 2)))
        assert a.rmse_kf == pytest.approx(np.sqrt(np.mean((a.run.x[have, 0] - tr.s[have]) ** 2)))
        end = C.TUNNEL_START + s.tunnel - 1
        assert a.tunnel_sd_end == pytest.approx(np.sqrt(a.run.P[end, 0, 0]))
        assert a.tunnel_err_end == pytest.approx(abs(a.run.x[end, 0] - tr.s[end]))
        inside = np.abs(a.run.x[C.N_BURN:, 0] - tr.s[C.N_BURN:]) <= 1.96 * np.sqrt(a.run.P[C.N_BURN:, 0, 0])
        assert a.coverage == pytest.approx(np.mean(inside))
        k = C.ETA_STEP
        assert a.eta_true[k] == pytest.approx((C.ROUTE_LENGTH - tr.s[k]) / tr.v[k])
        assert a.eta_kf[k] == pytest.approx((C.ROUTE_LENGTH - a.run.x[k, 0]) / a.run.x[k, 1])
        v_raw = (tr.z[k] - tr.z[k - C.RAW_SPEED_WINDOW]) / C.RAW_SPEED_WINDOW
        assert a.eta_raw[k] == pytest.approx((C.ROUTE_LENGTH - tr.z[k]) / v_raw)


def test_wrong_assumptions_are_applied_to_the_filter_not_to_the_truth():
    base = ev.analyse(ev.Settings(seed=5))
    over = ev.analyse(ev.Settings(seed=5, q_factor=0.1, r_factor=0.25))
    assert np.array_equal(base.track.s, over.track.s)
    assert np.array_equal(base.track.z, over.track.z, equal_nan=True)
    assert over.sd_pos[60] < base.sd_pos[60]          # glaubt weniger Rauschen: engeres Band


def test_process_noise_matrix_matches_numerical_integral_for_other_time_steps():
    """Q ist das Integral q^2 * g(s) g(s)^T ds mit g(s) = (dt - s, 1)^T: unabhängig von der Formel im Kern per Quadratur."""
    for dt in (0.25, 0.5, 1.0, 2.0, 3.5):
        for q in (0.05, 0.3, 1.0):
            s = np.linspace(0.0, dt, 20001)
            g = np.stack([dt - s, np.ones_like(s)])
            Q_ref = q ** 2 * getattr(np, "trapezoid", getattr(np, "trapz", None))(g[:, None, :] * g[None, :, :], s, axis=2)
            F, Q = A.cv_model(dt, q)
            assert Q == pytest.approx(Q_ref, rel=1e-6, abs=1e-12), (dt, q)
            assert F == pytest.approx(np.array([[1.0, dt], [0.0, 1.0]]))


def test_steady_state_matches_scipy_for_other_time_steps():
    for dt in (0.5, 2.0):
        F, Q = A.cv_model(dt, 0.3)
        H = A.H_GPS.reshape(1, 2)
        P_ref = scipy_linalg.solve_discrete_are(F.T, H.T, Q, np.array([[25.0]]))
        P_prior, _ = A.steady_state(F, Q, [(A.H_GPS, 25.0)])
        assert P_prior == pytest.approx(P_ref, rel=1e-6, abs=1e-9), dt
