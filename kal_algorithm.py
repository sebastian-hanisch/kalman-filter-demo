"""Kalman-Filter-Kern, numpy-only. Jede Funktion ist eine Einheit mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK 1b).

Modell: Zustand x = (Position s, Geschwindigkeit v), konstante Geschwindigkeit mit zufälliger Beschleunigung (Rauschen q in m/s^2).
Messungen: GPS misst s, Radsensor misst v. Beide werden nacheinander als skalare Messungen eingearbeitet.
"""

from dataclasses import dataclass

import numpy as np

import kal_constants as C

H_GPS = np.array([1.0, 0.0])
H_WHEEL = np.array([0.0, 1.0])


def cv_model(dt, q):
    """Übergangsmatrix F und Prozessrauschen Q des Modells mit konstanter Geschwindigkeit und weißem Beschleunigungsrauschen q."""
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = q ** 2 * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]])
    return F, Q


def predict(x, P, F, Q):
    """Vorhersage: Zustand mit dem Modell fortschreiben, Unsicherheit wächst um Q."""
    return F @ x, F @ P @ F.T + Q


def update_scalar(x, P, z, h, r):
    """Korrektur mit einer skalaren Messung z = h·x + Rauschen (Varianz r). Joseph-Form hält P symmetrisch und positiv.

    Gibt (x, P, K, nu, S) zurück: neuer Zustand, neue Kovarianz, Verstärkung, Innovation, Innovationsvarianz.
    """
    S = float(h @ P @ h + r)
    K = (P @ h) / S
    nu = float(z - h @ x)
    x_new = x + K * nu
    I_KH = np.eye(len(x)) - np.outer(K, h)
    P_new = I_KH @ P @ I_KH.T + np.outer(K, K) * r
    return x_new, P_new, K, nu, S


def nees(err, P):
    """Normiertes Fehlerquadrat (Normalized Estimation Error Squared): err' P^-1 err. Erwartung bei ehrlichem Filter: Zustandsdimension."""
    return float(err @ np.linalg.solve(P, err))


def steady_state(F, Q, updates, max_iter=5000, tol=1e-12):
    """Stationärer Zustand der Kovarianz-Rekursion (Riccati) durch Iteration.

    `updates`: Liste von (h, r), in dieser Reihenfolge eingearbeitet. Gibt (P_vorher, [K je Update]) zurück: die
    Kovarianz direkt nach der Vorhersage und die Verstärkungen des stationären Schritts.
    """
    P = np.eye(F.shape[0]) * 100.0
    P_prior = P
    for _ in range(max_iter):
        P_prior = F @ P @ F.T + Q
        Pn = P_prior
        gains = []
        for h, r in updates:
            S = float(h @ Pn @ h + r)
            K = (Pn @ h) / S
            I_KH = np.eye(F.shape[0]) - np.outer(K, h)
            Pn = I_KH @ Pn @ I_KH.T + np.outer(K, K) * r
            gains.append(K)
        if np.max(np.abs(Pn - P)) < tol:
            P = Pn
            break
        P = Pn
    return F @ P @ F.T + Q, gains


@dataclass(frozen=True)
class FilterRun:
    x: np.ndarray        # (T, 2) gefilterter Zustand nach der Korrektur
    P: np.ndarray        # (T, 2, 2) Kovarianz
    k_gps: np.ndarray    # (T, 2) Verstärkung des GPS-Updates, NaN ohne GPS-Messung
    gps_used: np.ndarray  # (T,) bool


def run_filter(z, w, dt, q, sigma_z, sigma_w, use_wheel, v_prior_mean, v_prior_std):
    """Kalman-Filter über eine ganze Spur. `q` und `sigma_z` sind die ANGENOMMENEN Werte des Filters.

    z: GPS-Positionen (NaN = Ausfall), w: Radsensor-Geschwindigkeiten (nur genutzt, wenn use_wheel).
    """
    T = len(z)
    F, Q = cv_model(dt, q)
    r_gps, r_wheel = sigma_z ** 2, sigma_w ** 2
    x = np.array([z[0], v_prior_mean], float)
    P = np.diag([r_gps, v_prior_std ** 2])
    xs, Ps = np.zeros((T, 2)), np.zeros((T, 2, 2))
    ks, used = np.full((T, 2), np.nan), np.zeros(T, bool)
    for t in range(T):
        if t > 0:
            x, P = predict(x, P, F, Q)
            if not np.isnan(z[t]):
                x, P, K, _, _ = update_scalar(x, P, z[t], H_GPS, r_gps)
                ks[t], used[t] = K, True
        else:
            ks[t], used[t] = np.nan, True
        if use_wheel:
            x, P, _, _, _ = update_scalar(x, P, w[t], H_WHEEL, r_wheel)
        xs[t], Ps[t] = x, P
    return FilterRun(xs, Ps, ks, used)
