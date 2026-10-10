"""Kennzahlen, Verdikte und Sweeps über die festen Instanzen (SWEEP_SEEDS)."""

from dataclasses import dataclass

import numpy as np

import kal_algorithm as A
import kal_constants as C
import kal_scenario as S


@dataclass(frozen=True)
class Settings:
    sigma_z: float = C.DEFAULT_SIGMA_Z      # wahres GPS-Rauschen in m
    q: float = C.DEFAULT_Q                  # wahres Beschleunigungsrauschen in m/s^2
    q_factor: float = C.DEFAULT_Q_FACTOR    # Annahme des Filters zu q als Vielfaches des wahren Werts
    r_factor: float = C.DEFAULT_R_FACTOR    # Annahme des Filters zur Messvarianz als Vielfaches des wahren Werts
    tunnel: int = C.DEFAULT_TUNNEL
    wheel: str = C.DEFAULT_WHEEL
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    track: S.Track
    run: A.FilterRun
    err_raw: np.ndarray        # GPS-Fehler (NaN im Tunnel)
    err_pos: np.ndarray
    err_vel: np.ndarray
    sd_pos: np.ndarray         # Standardabweichung der Position laut Filter
    sd_vel: np.ndarray
    nees: np.ndarray
    rmse_raw: float
    rmse_kf: float             # nur auf den Schritten mit GPS-Messung, damit der Vergleich fair ist
    rmse_vel: float
    nees_mean: float
    coverage: float            # Anteil der Schritte, in denen der wahre Wert im 95-%-Band des Filters liegt
    tunnel_err_end: float      # Positionsfehler am letzten Tunnelschritt (NaN ohne Tunnel)
    tunnel_sd_end: float
    eta_true: np.ndarray
    eta_raw: np.ndarray
    eta_kf: np.ndarray
    k_steady: np.ndarray       # stationäre GPS-Verstärkung (Position, Geschwindigkeit)
    k_end: np.ndarray          # letzte tatsächlich genutzte GPS-Verstärkung


def _assumed(settings):
    """Die Werte, mit denen der Filter rechnet."""
    return settings.q * settings.q_factor, settings.sigma_z * np.sqrt(settings.r_factor)


def _eta(pos, vel):
    out = np.full(len(pos), np.nan)
    ok = vel > C.ETA_MIN_SPEED
    out[ok] = (C.ROUTE_LENGTH - pos[ok]) / vel[ok]
    return out


def _raw_speed(z):
    v = np.full(len(z), np.nan)
    w = C.RAW_SPEED_WINDOW
    v[w:] = (z[w:] - z[:-w]) / (w * C.DT)
    return v


def run_settings(settings):
    """Spur erzeugen und den Filter mit den angenommenen Werten laufen lassen."""
    track = S.simulate(settings.seed, settings.sigma_z, settings.q, settings.tunnel)
    q_a, sz_a = _assumed(settings)
    run = A.run_filter(track.z, track.w, C.DT, q_a, sz_a, C.SIGMA_WHEEL, settings.wheel == "an", C.V_START_MEAN, C.V_PRIOR_STD)
    return track, run


def analyse(settings, steady=True):
    """Vollständige Auswertung einer Spur. `steady=False` überspringt die stationäre Verstärkung (für die Sweeps, spart Zeit)."""
    track, run = run_settings(settings)
    err_pos = run.x[:, 0] - track.s
    err_vel = run.x[:, 1] - track.v
    err_raw = track.z - track.s
    sd_pos = np.sqrt(run.P[:, 0, 0])
    sd_vel = np.sqrt(run.P[:, 1, 1])
    n = np.array([A.nees(np.array([err_pos[k], err_vel[k]]), run.P[k]) for k in range(C.T_STEPS)])
    sl = slice(C.N_BURN, None)
    have = ~np.isnan(track.z)
    have[:C.N_BURN] = False
    rmse_raw = float(np.sqrt(np.mean(err_raw[have] ** 2)))
    rmse_kf = float(np.sqrt(np.mean(err_pos[have] ** 2)))
    inside = np.abs(err_pos[sl]) <= 1.96 * sd_pos[sl]
    q_a, sz_a = _assumed(settings)
    F, Q = A.cv_model(C.DT, q_a)
    updates = [(A.H_GPS, sz_a ** 2)] + ([(A.H_WHEEL, C.SIGMA_WHEEL ** 2)] if settings.wheel == "an" else [])
    if steady:
        _, gains = A.steady_state(F, Q, updates)
    else:
        gains = [np.array([np.nan, np.nan])]
    last = np.where(run.gps_used & ~np.isnan(run.k_gps[:, 0]))[0]
    t_end = (C.TUNNEL_START + settings.tunnel - 1) if settings.tunnel > 0 else None
    eta_true = _eta(track.s, track.v)
    return Analysis(
        settings, track, run, err_raw, err_pos, err_vel, sd_pos, sd_vel, n,
        rmse_raw, rmse_kf, float(np.sqrt(np.mean(err_vel[sl] ** 2))), float(np.mean(n[sl])), float(np.mean(inside)),
        float(abs(err_pos[t_end])) if t_end is not None else float("nan"),
        float(sd_pos[t_end]) if t_end is not None else float("nan"),
        eta_true, _eta(track.z, _raw_speed(track.z)), _eta(run.x[:, 0], run.x[:, 1]),
        gains[0], run.k_gps[last[-1]] if len(last) else np.array([np.nan, np.nan]),
    )


def eta_errors(a, step=C.ETA_STEP):
    """Betrag des Ankunftszeitfehlers in s zum Zeitpunkt `step`: (Rohwert, Filter); NaN, wenn nicht definiert."""
    t = a.eta_true[step]
    return abs(a.eta_raw[step] - t), abs(a.eta_kf[step] - t)


# --- Sweeps ---------------------------------------------------------------------------------------------------------------------------------

def _over_seeds(settings, fn):
    from dataclasses import replace
    return [fn(analyse(replace(settings, seed=s), steady=False)) for s in C.SWEEP_SEEDS]


def sweep_sigma(settings):
    """GPS-Rauschen variieren: Positionsfehler Rohwert, Filter nur mit GPS und Filter mit GPS plus Radsensor (Median über die festen Spuren).

    Die Trennung ist wichtig: der Gewinn des reinen Filterns (Glätten über die Zeit) ist etwas anderes als der Gewinn der Sensorfusion.
    """
    from dataclasses import replace
    rows = []
    for sg in C.SIGMA_SWEEP:
        row = {"sigma": sg}
        for wheel in ("aus", "an"):
            res = _over_seeds(replace(settings, sigma_z=sg, wheel=wheel), lambda a: (a.rmse_raw, a.rmse_kf))
            if wheel == "aus":
                row["raw"] = float(np.median([r[0] for r in res]))
            row[f"kf_{wheel}"] = float(np.median([r[1] for r in res]))
            row[f"kf_{wheel}_high"] = float(np.max([r[1] for r in res]))
        rows.append(row)
    return rows


def sweep_assumptions(settings):
    """Konsistenz: mittlere NEES (Soll = Zustandsdimension) und 95-%-Abdeckung bei falsch angenommenem Q bzw. R.

    Mittelwert über die Spuren, nicht Median: die NEES ist schief verteilt, nur ihr Erwartungswert ist die Zustandsdimension.
    """
    from dataclasses import replace
    out = {"q": [], "r": []}
    for f in C.FACTOR_CHOICES:
        for key, st in (("q", replace(settings, q_factor=f, r_factor=1.0)), ("r", replace(settings, q_factor=1.0, r_factor=f))):
            res = _over_seeds(st, lambda a: (a.nees_mean, a.coverage, a.rmse_kf))
            out[key].append({"factor": f, "nees": float(np.mean([r[0] for r in res])), "coverage": float(np.mean([r[1] for r in res])),
                            "rmse": float(np.median([r[2] for r in res]))})
    return out


def sweep_tunnel(settings):
    """Tunnellänge variieren: Positionsfehler und Filterband (1 Sigma) am Tunnelende, mit und ohne Radsensor."""
    from dataclasses import replace
    rows = []
    for L in C.TUNNEL_SWEEP:
        row = {"tunnel": L}
        for wheel in ("an", "aus"):
            res = _over_seeds(replace(settings, tunnel=L, wheel=wheel), lambda a: (a.tunnel_err_end, a.tunnel_sd_end))
            row[f"err_{wheel}"] = float(np.median([r[0] for r in res])) if L > 0 else float("nan")
            row[f"sd_{wheel}"] = float(np.median([r[1] for r in res])) if L > 0 else float("nan")
        rows.append(row)
    return rows


def settle_time(a, tol=0.05):
    """Erster Schritt, ab dem die GPS-Verstärkung (Position) dauerhaft innerhalb `tol` (relativ) ihres stationären Werts liegt; None, wenn nie."""
    k = a.run.k_gps[:, 0]
    ref = float(a.k_steady[0])
    ok = np.abs(k - ref) <= tol * ref
    idx = np.where(~np.isnan(k))[0]
    if len(idx) == 0 or not ok[idx[-1]]:
        return None
    bad = [i for i in idx if not ok[i]]
    return int(bad[-1]) + 1 if bad else int(idx[0])


def local_level_gain(q_ratio):
    """Stationäre Verstärkung des Lokales-Niveau-Modells (einfache exponentielle Glättung): (sqrt(q^2 + 4q) - q) / 2 mit q = Niveau-/Messvarianz."""
    return float((np.sqrt(q_ratio ** 2 + 4 * q_ratio) - q_ratio) / 2.0)
