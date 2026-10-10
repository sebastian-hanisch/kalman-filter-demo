"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# Zeitachse: ein Schritt = eine Sekunde, 120 Schritte
DT = 1.0
T_STEPS = 120
N_BURN = 10                       # Einschwingzeit: Kennzahlen zählen erst ab diesem Schritt
ROUTE_LENGTH = 1500.0             # Strecke bis zum Ziel in m (für die Ankunftszeit)
ETA_STEP = 100                    # Zeitpunkt der Ankunftszeit-Auswertung
ETA_MIN_SPEED = 1.0               # unter dieser Geschwindigkeit (m/s) wird keine Ankunftszeit gerechnet
RAW_SPEED_WINDOW = 5              # Rohgeschwindigkeit aus der GPS-Differenz über so viele Sekunden

# feste Annahmen
V_START_MEAN = 10.0               # mittlere Startgeschwindigkeit in m/s (36 km/h)
V_START_STD = 1.0                 # Streuung der wahren Startgeschwindigkeit
V_PRIOR_STD = 5.0                 # das Vorwissen des Filters über die Startgeschwindigkeit ist weit gefasst
SIGMA_WHEEL = 0.3                 # Rauschen des Radsensors in m/s
TUNNEL_START = 50                 # Beginn des GPS-Ausfalls (Schritt)

# Reglergrenzen
SIGMA_Z_MIN, SIGMA_Z_MAX, DEFAULT_SIGMA_Z = 3.0, 30.0, 10.0
Q_MIN, Q_MAX, DEFAULT_Q = 0.05, 0.8, 0.3
FACTOR_CHOICES = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0)
DEFAULT_Q_FACTOR, DEFAULT_R_FACTOR = 1.0, 1.0
TUNNEL_MIN, TUNNEL_MAX, DEFAULT_TUNNEL = 0, 40, 20
WHEEL_CHOICES = ("an", "aus")
DEFAULT_WHEEL = "an"
SEED_MAX, DEFAULT_SEED = 999999, 35

# feste Instanzen für alle Sweeps: nie ein einzelner Seed als Befund
SWEEP_SEEDS = tuple(range(100000, 100040))
SIGMA_SWEEP = (3.0, 6.0, 10.0, 15.0, 20.0, 30.0)
TUNNEL_SWEEP = (0, 10, 20, 30, 40)

STATE_DIM = 2                     # Zustand: Position und Geschwindigkeit; Soll-Wert der mittleren NEES

COLOR_TRUTH = "#2F6B65"
COLOR_RAW = "rgba(150,150,150,0.8)"
COLOR_FILTER = "#4c78a8"
COLOR_BAND = "rgba(76,120,168,0.18)"
COLOR_WHEEL = "#c97b2a"
COLOR_TUNNEL = "rgba(120,120,120,0.12)"

PRESETS = {
    "Standardfahrt (Voreinstellung)": {"sz": DEFAULT_SIGMA_Z, "q": DEFAULT_Q, "qf": 1.0, "rf": 1.0, "tun": DEFAULT_TUNNEL, "rad": "an", "seed": DEFAULT_SEED},
    "Schlechtes GPS": {"sz": 25.0, "q": DEFAULT_Q, "qf": 1.0, "rf": 1.0, "tun": DEFAULT_TUNNEL, "rad": "an", "seed": DEFAULT_SEED},
    "Filter zu sicher eingestellt": {"sz": DEFAULT_SIGMA_Z, "q": DEFAULT_Q, "qf": 0.1, "rf": 0.25, "tun": DEFAULT_TUNNEL, "rad": "an", "seed": DEFAULT_SEED},
    "Langer Tunnel, ohne Radsensor": {"sz": DEFAULT_SIGMA_Z, "q": DEFAULT_Q, "qf": 1.0, "rf": 1.0, "tun": 40, "rad": "aus", "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfahrt (Voreinstellung)": "GPS mit 10 m Rauschen, gleichmäßige Fahrt mit leichten Geschwindigkeitsschwankungen, 20 s Tunnel, Radsensor an; der Filter kennt die wahren Rauschwerte.",
    "Schlechtes GPS": "Das GPS rauscht mit 25 m statt 10 m. Der Filter kennt das und gewichtet die Messung entsprechend weniger.",
    "Filter zu sicher eingestellt": "Der Filter nimmt das Prozessrauschen zehnmal und das Messrauschen viermal zu klein an: seine Unsicherheitsbänder sind zu schmal, die Konsistenzprüfung schlägt an.",
    "Langer Tunnel, ohne Radsensor": "40 s ohne GPS und ohne Radsensor: nur die Vorhersage trägt, die Unsicherheit wächst sichtbar.",
}
