"""Kalman-Filter: Zustand schätzen aus verrauschten Messungen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein Lieferfahrzeug fährt eine Strecke; GPS misst die Position (verrauscht, im Tunnel gar nicht), ein Radsensor die Geschwindigkeit.
Der Kalman-Filter schätzt daraus Position und Geschwindigkeit samt Unsicherheit und rechnet die Ankunftszeit.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import kal_algorithm as A
import kal_constants as C
import kal_evaluation as ev
from kal_evaluation import Settings, analyse, eta_errors, local_level_gain, settle_time
from kal_pdf_export import build_pdf
from kal_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from kal_visualization import (
    build_assumption_sweep,
    build_error_chart,
    build_eta_chart,
    build_gain_chart,
    build_sigma_sweep,
    build_speed_chart,
    build_tunnel_sweep,
)

st.set_page_config(page_title="Kalman-Filter – Sebastian Hanisch", layout="wide")


def de(x, nd=1):
    """Dezimalkomma statt Punkt."""
    return f"{x:.{nd}f}".replace(".", ",")


def factor_label(f):
    return "×" + f"{f:g}".replace(".", ",")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep_sigma(settings):
    return ev.sweep_sigma(settings)


@st.cache_data(show_spinner=False)
def _sweep_assumptions(settings):
    return ev.sweep_assumptions(settings)


@st.cache_data(show_spinner=False)
def _sweep_tunnel(settings):
    return ev.sweep_tunnel(settings)


def run_experiment(key, fn, settings, label):
    """Teure Sweeps nur auf Knopfdruck: das Ergebnis bleibt sichtbar, solange die Einstellungen gleich bleiben.

    Ohne diese Bremse würde jede Reglerbewegung mehrere Sekunden rechnen. Gerechnet wird über `st.cache_data`.
    """
    if st.button(label, key=f"run_{key}"):
        st.session_state[f"done_{key}"] = settings
    if st.session_state.get(f"done_{key}") != settings:
        st.info("Noch nicht gerechnet für diese Einstellungen. Der Knopf startet die Rechnung (wenige Sekunden); danach bleibt das Ergebnis stehen.")
        return None
    with st.spinner("Rechne … (je nach Rechner wenige Sekunden)"):
        return fn(settings)


st.title("📡 Kalman-Filter: Zustand schätzen aus verrauschten Messungen")
st.markdown(
    """
Ein Lieferfahrzeug fährt eine Strecke. **GPS** liefert seine Position, aber verrauscht und im Tunnel gar nicht; ein **Radsensor** liefert die
Geschwindigkeit. Der **Kalman-Filter** macht daraus eine Schätzung von Position und Geschwindigkeit **samt ihrer Unsicherheit** und rechnet
daraus die Ankunftszeit. Die Demo ist das erste Stück der Linie **Zustandsschätzung und Filter**; im Unterschied zu den Fall-Demos zeigt sie ein einziges
Verfahren an einem wachsenden Beispiel.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/kalman-filter-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Ein Lieferwagen fährt eine Strecke. Das GPS zeigt seine Position, aber verrauscht: Der Punkt springt hin und her, im Tunnel fehlt er ganz. Ein Radsensor liefert zusätzlich die Geschwindigkeit.

Der **Kalman-Filter** wiederholt zwei Schritte, Sekunde für Sekunde. Zuerst sagt er voraus, wo der Wagen jetzt sein müsste, und seine Unsicherheit wächst, weil der Wagen beschleunigen oder bremsen kann. Dann kommt eine Messung, und der Filter mischt Vorhersage und Messung. Ist die Vorhersage unsicher und die Messung genau, zählt die Messung mehr, sonst die Vorhersage. Danach schrumpft die Unsicherheit wieder.

Der Filter liefert also eine Schätzung und eine Angabe, wie sehr man ihr glauben darf (das blaue Band in den Abbildungen). Die Demo fragt deshalb nicht nur, wie genau er ist, sondern ob diese Angabe ehrlich ist: Passt der wahre Fehler zur Unsicherheit, die der Filter selbst nennt? Mit den Reglern „Was der Filter glaubt“ können Sie ihn absichtlich falsch einstellen und beobachten, was dann passiert.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/kalman-filter-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert der Kalman-Filter", expanded=True):
    st.markdown(
        """
Der Filter wiederholt zwei Schritte, Sekunde für Sekunde:

1. **Vorhersage:** Aus Position und Geschwindigkeit der letzten Sekunde folgt, wo das Fahrzeug jetzt sein sollte. Die Unsicherheit **wächst**, weil das Fahrzeug
   beschleunigen oder bremsen kann.
2. **Korrektur:** Kommt eine Messung, wird die Vorhersage mit ihr **gewichtet gemittelt**. Das Gewicht heißt Kalman-Verstärkung: ist die Vorhersage unsicher und die Messung
   genau, zählt die Messung viel; ist es umgekehrt, zählt sie wenig. Die Unsicherheit **schrumpft**.

**Warum das funktioniert:** Wenn Modell und Rauschen stimmen, gibt es kein besseres lineares Verfahren; der Filter ist unter dieser Annahme optimal und
liefert zugleich ein Maß, wie sehr man ihm glauben darf (das blaue Band unten).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Die Wirklichkeit**")
    sigma_z = st.slider("GPS-Rauschen (m)", *bounds("sigma_slider"), key="sigma_slider", step=0.5,
                        help="Streuung der GPS-Position um die wahre Position (Standardabweichung, in Metern).")
    q = st.slider("Fahrunruhe (m/s²)", *bounds("q_slider"), key="q_slider", step=0.05,
                  help="Wie stark die Beschleunigung zufällig schwankt. Höher heißt: unruhigere Fahrt, die der Filter schlechter vorhersagt.")
    tunnel = st.slider("Tunnellänge (s)", *bounds("tunnel_slider"), key="tunnel_slider", step=5,
                       help="So lange fällt das GPS ab Sekunde 50 aus. Null heißt: kein Tunnel.")
    wheel = st.radio("Radsensor", C.WHEEL_CHOICES, key="wheel_select", horizontal=True,
                     help="Der Radsensor misst die Geschwindigkeit (0,3 m/s Rauschen) und läuft auch im Tunnel weiter.")
    st.markdown("**Was der Filter glaubt**")
    qf = st.select_slider("Annahme zur Fahrunruhe", C.FACTOR_CHOICES, key="qf_select", format_func=factor_label,
                          help="Verhältnis der vom Filter angenommenen zur wahren Fahrunruhe. ×1 heißt: der Filter kennt die Wahrheit.")
    rf = st.select_slider("Annahme zur Messvarianz", C.FACTOR_CHOICES, key="rf_select", format_func=factor_label,
                          help="Verhältnis der vom Filter angenommenen zur wahren GPS-Varianz. ×1 heißt: der Filter kennt die Wahrheit.")
    seed = st.number_input("Zufalls-Seed der Fahrt", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Fahrt generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"sigma_slider": float(sigma_z), "q_slider": float(q), "qf_select": float(qf), "rf_select": float(rf),
                   "tunnel_slider": int(tunnel), "wheel_select": wheel, "seed_input": int(seed)})

settings = Settings(float(sigma_z), float(q), float(qf), float(rf), int(tunnel), wheel, int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
# Delta immer als „mein Wert minus Referenz“; die Farbe regelt delta_color="inverse" (Vorzeichen-Falle, DEMO-PLAYBOOK §2).
m1.metric("Positionsfehler Filter", f"{de(a.rmse_kf)} m", delta=f"{a.rmse_kf - a.rmse_raw:+.1f} m gegen Rohwert".replace(".", ","), delta_color="inverse",
          help="Wurzel des mittleren quadratischen Fehlers (RMSE) auf den Sekunden mit GPS-Messung, ab Sekunde 10.")
m2.metric("Positionsfehler GPS-Rohwert", f"{de(a.rmse_raw)} m", delta="Referenz", delta_color="off")
m3.metric("Geschwindigkeitsfehler Filter", f"{de(a.rmse_vel, 2)} m/s", delta="Radsensor-Rauschen 0,3 m/s" if wheel == "an" else "ohne Radsensor", delta_color="off")
m4.metric("Mittlere NEES (Soll 2)", de(a.nees_mean, 2), delta=f"{100 * a.coverage:.0f} % im 95-%-Band".replace(".", ","), delta_color="off",
          help="Normiertes Fehlerquadrat: misst, ob der wahre Fehler zur Unsicherheit passt, die der Filter selbst angibt. Soll: die Zustandsdimension, hier 2.")

st.markdown("**Positionsfehler über die Zeit** – grau: GPS-Rohwerte, blau: Filter, Band: das 95-%-Intervall, das der Filter sich selbst zuschreibt.")
st.plotly_chart(build_error_chart(a), width="stretch", key="error_chart")
st.markdown("**Geschwindigkeit** – der Filter schätzt sie, obwohl GPS nur die Position misst.")
st.plotly_chart(build_speed_chart(a), width="stretch", key="speed_chart")

st.download_button("📄 Auswertung als PDF", data=build_pdf(a), file_name="kalman_filter_auswertung.pdf", mime="application/pdf", key="pdf_button")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Ist der Filter ehrlich?")
ratio = a.nees_mean / C.STATE_DIM
if ratio > 1.6:
    st.warning(
        f"**Überselbstsicher.** Die mittlere NEES liegt bei {de(a.nees_mean, 2)} statt 2 (Faktor {de(ratio)}): der wahre Fehler ist größer, als das Band verspricht; "
        f"nur {100 * a.coverage:.0f} % der Zeit liegt die Wahrheit im 95-%-Band. Der Filter glaubt weniger Rauschen, als es gibt."
    )
elif ratio < 0.6:
    st.info(f"**Zu vorsichtig.** Die mittlere NEES liegt bei {de(a.nees_mean, 2)} statt 2: das Band ist breiter als nötig. Das ist sicher, kostet aber Genauigkeit.")
else:
    st.success(f"**Konsistent.** Die mittlere NEES liegt bei {de(a.nees_mean, 2)} (Soll 2), die Wahrheit liegt zu {100 * a.coverage:.0f} % im 95-%-Band.")
st.caption("Eine einzelne Fahrt streut stark; die Experimente unten mitteln über 40 feste Fahrten.")

e_raw, e_kf = eta_errors(a)
st.markdown("### Ankunftszeit")
st.plotly_chart(build_eta_chart(a), width="stretch", key="eta_chart")
if np.isnan(e_raw) or np.isnan(e_kf):
    st.info("Zum Zeitpunkt 100 s ist die Geschwindigkeit zu klein für eine Ankunftszeit.")
else:
    st.info(
        f"Restfahrzeit bis zum Ziel in {C.ROUTE_LENGTH:.0f} m Entfernung, bewertet bei {C.ETA_STEP} s: "
        f"aus den GPS-Rohwerten (Geschwindigkeit aus {C.RAW_SPEED_WINDOW} s Positionsdifferenz) um **{de(e_raw)} s** daneben, "
        f"aus dem Filter um **{de(e_kf)} s**."
    )

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Was bringt der Filter gegenüber der Rohmessung?")
st.caption(f"Median über {len(C.SWEEP_SEEDS)} feste Fahrten; Fahrunruhe, Tunnel und Annahmen wie in der Seitenleiste, das GPS-Rauschen läuft von 3 bis 30 m.")
rows = run_experiment("sigma", _sweep_sigma, settings, f"▶ Experiment 1 rechnen ({len(C.SIGMA_SWEEP) * 2 * len(C.SWEEP_SEEDS)} Filterläufe)")
if rows is not None:
    st.plotly_chart(build_sigma_sweep(rows), width="stretch", key="sigma_chart")
    r10 = min(rows, key=lambda r: abs(r["sigma"] - settings.sigma_z))
    st.info(
        f"Bei {r10['sigma']:.0f} m GPS-Rauschen: Rohwert {de(r10['raw'])} m, Filter nur mit GPS {de(r10['kf_aus'])} m "
        f"(Faktor {de(r10['raw'] / r10['kf_aus'])}), Filter mit GPS und Radsensor {de(r10['kf_an'])} m (Faktor {de(r10['raw'] / r10['kf_an'])}). "
        "Zwei Dinge sind hier vermischt, deshalb die zwei Linien: **Glätten über die Zeit** (nur GPS) und **Sensorfusion** (der Radsensor liefert Information, die GPS nicht hat)."
    )

st.subheader("🔬 Experiment 2: Was kostet eine falsche Annahme?")
st.caption(
    f"Mittlere NEES über {len(C.SWEEP_SEEDS)} feste Fahrten, wenn der Filter Prozess- bzw. Messrauschen um den Faktor auf der Achse falsch annimmt "
    "(Faktor 1 = richtig). Soll: 2."
)
sw = run_experiment("assumptions", _sweep_assumptions, settings, f"▶ Experiment 2 rechnen ({2 * len(C.FACTOR_CHOICES) * len(C.SWEEP_SEEDS)} Filterläufe)")
if sw is not None:
    st.plotly_chart(build_assumption_sweep(sw), width="stretch", key="assumption_chart")
    worst_q = sw["q"][0]
    worst_r = sw["r"][0]
    q_hi = sw["q"][-1]
    q_one = next(r for r in sw["q"] if r["factor"] == 1.0)
    st.info(
        f"Nimmt der Filter das Prozessrauschen **zehnmal zu klein** an, steigt die NEES auf {de(worst_q['nees'])} und nur {100 * worst_q['coverage']:.0f} % der Zeit liegt "
        f"die Wahrheit im Band. Beim zehnmal zu kleinen Messrauschen sind es {de(worst_r['nees'])} und {100 * worst_r['coverage']:.0f} %. "
        f"Zu große Annahmen schaden der Ehrlichkeit nicht (zehnmal zu großes Q: NEES {de(q_hi['nees'], 2)}), kosten aber Genauigkeit: "
        f"der Positionsfehler steigt von {de(q_one['rmse'])} m auf {de(q_hi['rmse'])} m."
    )

st.subheader("🔬 Experiment 3: Was passiert im Tunnel?")
st.caption(f"Positionsfehler (Median) und Unsicherheit des Filters (1 Sigma) am letzten Tunnelschritt, über {len(C.SWEEP_SEEDS)} feste Fahrten, mit und ohne Radsensor.")
trows = run_experiment("tunnel", _sweep_tunnel, settings, f"▶ Experiment 3 rechnen ({len(C.TUNNEL_SWEEP) * 2 * len(C.SWEEP_SEEDS)} Filterläufe)")
if trows is not None:
    st.plotly_chart(build_tunnel_sweep(trows), width="stretch", key="tunnel_chart")
    t_long = trows[-1]
    st.info(
        f"Bei {t_long['tunnel']} s ohne GPS wächst die Position ohne Radsensor auf {de(t_long['err_aus'])} m Fehler bei {de(t_long['sd_aus'])} m angegebener Unsicherheit; "
        f"mit Radsensor bleiben es {de(t_long['err_an'])} m bei {de(t_long['sd_an'])} m. In beiden Fällen **wächst das Band mit dem Fehler**; "
        "gemessen ist hier der Median, nicht wie oft die Wahrheit im Band liegt."
    )

st.subheader("🔬 Experiment 4: Die Verstärkung wird fest – die Brücke zur exponentiellen Glättung")
st.plotly_chart(build_gain_chart(a), width="stretch", key="gain_chart")
settle = settle_time(a)
st.info(
    f"Die Verstärkung auf die Position läuft gegen den stationären Wert {de(a.k_steady[0], 4)} (Geschwindigkeit: {de(a.k_steady[1], 4)}); "
    + (f"ab Sekunde {settle} liegt sie dauerhaft innerhalb von 5 % davon. " if settle is not None else "in dieser Fahrt erreicht sie ihn nicht dauerhaft (Tunnel). ")
    + "Mit festem Gewicht ist der Filter ein einfacher Glätter derselben Bauart wie ein α-β-Filter oder die Holt-Methode der exponentiellen Glättung; "
    "das Besondere am Kalman-Filter ist, dass er dieses Gewicht aus den Rauschwerten **herleitet**, statt es von Hand zu wählen, und dass er es **ändert**, wenn Messungen fehlen."
)
ratios = (0.01, 0.1, 1.0)
live = []
for qr in ratios:
    alpha_it = float(
        A.steady_state(np.array([[1.0]]), np.array([[qr]]), [(np.array([1.0]), 1.0)])[1][0][0]
    )
    live.append((qr, local_level_gain(qr), alpha_it))
st.markdown(
    "**Einfachster Fall (nur ein Niveau, keine Geschwindigkeit):** das Gewicht der neuen Messung ist $\\alpha = (\\sqrt{q^2+4q}-q)/2$ mit $q$ = Niveau-Varianz geteilt durch Messvarianz "
    "– genau die einfache exponentielle Glättung.\n\n"
    "| q | α nach Formel | α aus der Riccati-Iteration |\n|---|---|---|\n"
    + "\n".join(f"| {de(qr, 2)} | {de(x, 4)} | {de(y, 4)} |" for qr, x, y in live)
)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Bewegungs- und Messmodell sind linear** | Bei Kurvenfahrt oder wenn nicht die Position, sondern Entfernung und Peilung zu einer Bake gemessen werden, ist das Modell nur näherungsweise gültig; der Filter wird überselbstsicher. | Erweiterter Kalman-Filter, Unscented Kalman-Filter |
| **Rauschen ist gaußsch und seine Größe bekannt** | Ein einzelner GPS-Sprung von 80 m wird wie eine echte Messung eingearbeitet. Zu klein angenommenes Rauschen macht den Filter überselbstsicher (Experiment 2). | Robuste Filter mit Toren oder t-Verteilung |
| **Eine Hypothese, ein Gipfel** | Fährt das Fahrzeug auf einer von zwei parallelen Straßen, braucht es zwei Hypothesen; ein Gauß-Filter mittelt dazwischen. | Hidden-Markov-Modelle, Partikelfilter |
| **Nur die Vergangenheit zählt** | Der Filter kennt in Sekunde 60 nichts aus Sekunde 70, obwohl es die Schätzung für 60 verbessern würde, wenn man später auswertet. | Glätter (Rauch-Tung-Striebel) |
| **Ein Fahrzeug, klare Zuordnung** | Bei mehreren Fahrzeugen muss jede Messung erst einem Fahrzeug zugeordnet werden. | Mehrzielverfolgung mit Datenzuordnung |
| **Das Modell der Fahrt stimmt** | Bei einem Manöver (Bremsen, Abbiegen) ist die Annahme „fast konstante Geschwindigkeit“ falsch; ein hohes Q rettet den Filter, glättet aber schlechter. | Mehrere Bewegungsmodelle (IMM) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/kalman-filter-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind verrauschte Messungen der Position (GPS) und der Geschwindigkeit (Radsensor) eines Fahrzeugs. Gesucht sind Schätzungen von Position und Geschwindigkeit zu jedem Zeitpunkt samt Kovarianz, also der Unsicherheit der Schätzung.

| Symbol | Bedeutung |
|---|---|
| $x_k = (s_k, v_k)^\top$ | Zustand zum Zeitpunkt $k$: Position und Geschwindigkeit |
| $F$ | Übergangsmatrix: Position wächst um Geschwindigkeit mal $\Delta t$, Geschwindigkeit bleibt bis auf Zufall gleich |
| $Q$ | Kovarianz des Prozessrauschens (zufällige Beschleunigung, Stärke $q$) |
| $z$, $r$ | eine skalare Messung (GPS-Position oder Radsensor-Geschwindigkeit) und ihre Varianz |
| $h$ | Messvektor: wählt aus dem Zustand die gemessene Größe aus |
| $\hat x$, $P$ | Schätzung des Zustands und ihre Kovarianz (Unsicherheit) |
| $S$, $K$ | Varianz der Abweichung zwischen Messung und Vorhersage und Kalman-Verstärkung (das Gewicht der Messung) |
| $\mathrm{NEES}_k$ | normiertes Fehlerquadrat: wahrer Fehler im Verhältnis zur angegebenen Unsicherheit, Soll gleich der Zustandsdimension |
| $\alpha$ | Gewicht der neuen Messung im einfachsten Modell (nur ein Niveau) |
| $q$ | in der Dynamik die Stärke der zufälligen Beschleunigung ($Q \propto q^2$); im einfachsten Modell dagegen das Verhältnis von Niveau- zu Messvarianz (zwei verschiedene Größen mit demselben Buchstaben) |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/kalman-filter-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Zustand und Modell.** Zustand $x_k = (s_k, v_k)^\top$ (Position, Geschwindigkeit) im Abstand $\Delta t$:
$$x_{k+1} = F x_k + w_k,\quad F = \begin{pmatrix}1 & \Delta t\\ 0 & 1\end{pmatrix},\quad w_k \sim \mathcal N(0, Q),\quad
Q = q^2\begin{pmatrix}\Delta t^3/3 & \Delta t^2/2\\ \Delta t^2/2 & \Delta t\end{pmatrix}.$$
Messungen: GPS $z_k = s_k + \varepsilon_k$ mit $\varepsilon_k \sim \mathcal N(0, \sigma_z^2)$, Radsensor $y_k = v_k + \eta_k$ mit $\eta_k \sim \mathcal N(0, \sigma_w^2)$.

**Vorhersage.** $\hat x_{k|k-1} = F\hat x_{k-1}$, $\;P_{k|k-1} = F P_{k-1} F^\top + Q$.

**Korrektur** (je skalare Messung $z = h^\top x + \varepsilon$, Varianz $r$):
$$S = h^\top P h + r,\quad K = P h / S,\quad \hat x \leftarrow \hat x + K(z - h^\top\hat x),\quad P \leftarrow (I - Kh^\top)P(I - Kh^\top)^\top + K r K^\top.$$
Die letzte Form (Joseph-Form) hält $P$ symmetrisch und positiv.

**Gaußsche Optimalität.** Bei linearem Modell und gaußschem Rauschen ist $\hat x_k$ der Erwartungswert und $P_k$ die Kovarianz der Verteilung des Zustands
gegeben alle bisherigen Messungen; die Demo prüft das gegen die Lösung des gesamten Problems als Kleinste-Quadrate-System.

**Konsistenz.** $\mathrm{NEES}_k = (x_k - \hat x_k)^\top P_k^{-1}(x_k - \hat x_k)$ hat bei ehrlichem Filter den Erwartungswert 2 (Zustandsdimension).

**Stationäre Verstärkung.** Die Kovarianz-Rekursion (Riccati-Gleichung) konvergiert bei konstanten $F, Q, r$ gegen einen festen Wert; die zugehörige Verstärkung $K_\infty$ ist ein festes
Gewicht der Messung. Im Lokales-Niveau-Modell $x_{k+1} = x_k + w_k$, $z_k = x_k + \varepsilon_k$ mit $q = \operatorname{Var}(w)/\operatorname{Var}(\varepsilon)$ folgt
$\alpha = (\sqrt{q^2 + 4q} - q)/2$; das ist die einfache exponentielle Glättung mit diesem Glättungsparameter. Dieses $q$ (Verhältnis zweier Varianzen) ist nicht die Beschleunigungsstärke $q$ des Modells oben.

**Literatur.** R. E. Kalman (1960), *A New Approach to Linear Filtering and Prediction Problems*, Transactions of the ASME – Journal of Basic Engineering 82 (Series D), 35–45.
Y. Bar-Shalom, X. R. Li, T. Kirubarajan (2001), *Estimation with Applications to Tracking and Navigation*, Wiley (Standardwerk zur Zustandsschätzung).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
