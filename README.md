# Kalman-Filter: Zustand schätzen aus verrauschten Messungen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-kalman-filter-demo.streamlit.app/)**

Erstes Stück der Linie **Zustandsschätzung und Filter** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“.
Im Unterschied zu den Fall-Demos zeigt die Demo **ein** Verfahren an einem wachsenden Beispiel: ein Lieferfahrzeug fährt eine Strecke, GPS misst die Position
(verrauscht, im Tunnel gar nicht), ein Radsensor die Geschwindigkeit. Der Kalman-Filter schätzt daraus Position und Geschwindigkeit samt Unsicherheit und rechnet die Ankunftszeit.

## Warum dieses Problem

Eine Messung ist nie die Wahrheit. Wer Ankunftszeiten aus rohen GPS-Punkten rechnet, rechnet mit dem Rauschen. Der Kalman-Filter fasst alle bisherigen Messungen zu einer
Schätzung **und** zu einer Aussage zusammen, wie sehr man ihr glauben darf. Die Kernfrage der Demo ist deshalb nicht nur „wie genau?“, sondern **„ist der Filter ehrlich?“**:
passt der wahre Fehler zur Unsicherheit, die der Filter selbst angibt?

## Modell

Zustand (Position, Geschwindigkeit), konstante Geschwindigkeit mit zufälliger Beschleunigung; GPS misst die Position, der Radsensor die Geschwindigkeit; beide
Messungen werden nacheinander als skalare Messungen eingearbeitet (Joseph-Form). Formeln im 📐-Expander der App, Kern in `kal_algorithm.py`. Die Wahrheit wird mit
genau diesem Modell erzeugt: die Demo prüft das Verfahren unter **richtigen** Annahmen und zeigt dann, was passiert, wenn der Filter falsch eingestellt ist.

## Methodik

Vier Experimente; die ersten drei rechnen auf Knopfdruck über je 40 feste Fahrten (`SWEEP_SEEDS`), das vierte läuft sofort:

1. Positionsfehler gegen den GPS-Rohwert, getrennt nach **Glätten über die Zeit** (nur GPS) und **Sensorfusion** (GPS plus Radsensor).
2. Konsistenz (NEES) bei falsch angenommenem Prozess- bzw. Messrauschen.
3. Tunnel: Fehler und angegebene Unsicherheit am Tunnelende, mit und ohne Radsensor.
4. Die Verstärkung wird fest: Brücke zur exponentiellen Glättung.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Standardeinstellung (GPS 10 m, Fahrunruhe 0,3 m/s², Radsensor-Rauschen 0,3 m/s, Tunnel 20 s), über die 40 festen Fahrten; `tests/test_claims.py` rechnet jede nach. Positionsfehler und Tunnelwerte (am letzten Tunnelschritt) sind Mediane, NEES und Abdeckung Mittelwerte über diese Fahrten (`kal_evaluation.py`).

| Frage | Befund |
|---|---|
| Was bringt der Filter gegen den Rohwert? | Rohwert **9,7 m**, Filter nur mit GPS **4,9 m**, mit GPS und Radsensor **1,6 m** (Positionsfehler, RMSE). Das Glätten allein halbiert den Fehler ungefähr; die zusätzliche Information des Radsensors senkt ihn danach weiter. Welcher Teil der größere ist, hängt vom Maß (Meter oder Faktor) und von der Reihenfolge ab. |
| Ehrlichkeit bei richtigem Modell? | Mittlere NEES über 150 Fahrten (Orakel-Test) **2,0** (Soll 2). Einzelne Fahrten streuen stark; die 40 festen ergeben 1,8. |
| Filter glaubt zehnmal weniger Prozessrauschen? | NEES **61**, nur **80 %** der Zeit liegt die Wahrheit im 95-%-Band. |
| Filter glaubt zehnmal weniger Messrauschen? | NEES **6,5**, nur **59 %** im Band. |
| Und zu große Annahmen? | Ehrlich bleibt er (NEES 1,5), kostet aber Genauigkeit: Positionsfehler **2,3 m** statt 1,6 m bei zehnmal zu großem Q. |
| 40 s Tunnel ohne Radsensor | Position im Median **49 m** daneben, der Filter gibt **57 m** (1 Sigma, Median) an: das Band wächst mit dem Fehler. Wie oft die Wahrheit im Tunnel im Band liegt, ist nicht gemessen. |
| 40 s Tunnel mit Radsensor | **1,4 m** Fehler bei 2,7 m angegebener Unsicherheit: die Geschwindigkeit trägt die Position. |
| Stationäre Verstärkung (Standard) | Position **0,0315**, Geschwindigkeit **0,0015**; im Lokales-Niveau-Modell ist α = (√(q²+4q) − q)/2: q = 0,01 → 0,0951, 0,1 → 0,2702, 1 → 0,6180. |
| Ankunftszeit bei 100 s (Standardfahrt, Seed 35) | aus GPS-Rohwerten um 2,1 s daneben, aus dem Filter (mit GPS und Radsensor) um 0,02 s. Eine einzelne Fahrt, keine Aussage über die Streuung. |

## Ehrliche Grenzen

- **Die Wahrheit ist das Modell.** Die Fahrt wird mit dem Modell des Filters erzeugt; das zeigt, was das Verfahren unter richtigen Annahmen leistet, nicht, wie es auf echten GPS-Spuren abschneidet (kein echter Datensatz).
- **Zwei Effekte sind vermischt, wo der Radsensor mitläuft.** Der Gewinn von 9,7 auf 1,6 m stammt aus Glätten über die Zeit und Sensorfusion zugleich; die getrennte Linie „nur GPS“ zeigt den ersten Teil. Dass der Radsensor den größeren Teil beiträgt, gilt nur im Verhältnis (Faktor), nicht in Metern.
- **Eine einzelne Fahrt ist kein Befund.** NEES und Abdeckung einer Fahrt schwanken stark; belastbar sind nur die Mittel/Mediane über die 40 festen Fahrten.
- **Die Standardspur ist gutmütig:** ihre NEES liegt unter 2, sie wirkt „zu vorsichtig“. Das ist Zufall dieser Spur, kein Befund; die Orakel-Prüfung über 150 Fahrten ergibt 2,0.
- **Ankunftszeit** ist nur gerechnet, wo die Geschwindigkeit über 1 m/s liegt; das Ziel ist eine feste Strecke, kein Straßennetz.
- **Rauschen ist gaußsch, linear, bekannt.** Ausreißer, Kurven, mehrere Hypothesen und mehrere Fahrzeuge sind hier nicht abgedeckt (Tabelle „Wo die Annahmen enden“ in der App).

## Verifikation

- **Kern:** Einheiten (Vorhersage, Korrektur, NEES, Riccati-Iteration) gegen Handrechnung auf der Mini-Instanz (`tests/conftest.py`), Randfälle (kein GPS, Tunnel, Radsensor an/aus).
- **Orakel** (`tests/test_oracle_kalman_filter.py`, anderer Rechenweg): stationäre Kovarianz und Verstärkung gegen `scipy.linalg.solve_discrete_are`; Lokales-Niveau-Modell gegen die geschlossene Formel und scipy; der gefilterte Endzustand samt Kovarianz **gleich der Lösung des gesamten Problems als gewichtetes Kleinste-Quadrate-System** (auch mit GPS-Lücke); Vorhersage-Kovarianz gegen die explizite Summe; Joseph-Form gegen die einfache Form; mittlere NEES über 150 Fahrten gleich 2; die angezeigten Kennzahlen gegen unabhängige Neuberechnung.
- **Zahlen im README:** `tests/test_claims.py`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, Regler an Minimum und Maximum, jeder Auswahlwert, Experiment-Knöpfe) und Footer-Test.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App (Skelett nach DEMO-PLAYBOOK §2) |
| `kal_algorithm.py` | Kern: Modell, Vorhersage, Korrektur (Joseph-Form), NEES, stationäre Rekursion, Filterlauf |
| `kal_scenario.py` | Fahrt und Messungen (ein Generator, Regler skalieren denselben Zufall) |
| `kal_evaluation.py` | Kennzahlen, Verdikte, Sweeps über die festen Fahrten |
| `kal_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `kal_presets.py` | Permalink (Begrenzen/Einrasten), Presets, Zufalls-Seed-Button |
| `kal_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `kal_pdf_export.py` | PDF-Auswertung (fpdf2) |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit bzw. Laufzeit plus pytest/scipy |
| `tests/` | Einheiten, Orakel, Behauptungen, Presets, AppTest, Footer |

## Bewusst nicht umgesetzt

- **Kurvenfahrt und Entfernungs-/Peilmessung** (nichtlineare Modelle), **robuste Verfahren** gegen Ausreißer, **mehrere Hypothesen** (Straßenzuordnung), **Glätter** und **mehrere Fahrzeuge** sind Gegenstand eigener Stücke der Linie.
- **Echte GPS-Spuren** und ein Straßennetz: kein Datensatz eingebunden.
- **Automatisches Einstellen von Q und R** aus den Daten: die Demo zeigt gerade, was passiert, wenn man sie falsch setzt.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
