"""PDF-Export der aktuellen Auswertung (fpdf2, Kernschriften; Umlaute sind dort erlaubt, Gedankenstrich und Eurozeichen nicht)."""

from fpdf import FPDF


def _num(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",")


def build_pdf(a):
    """Einseitiger Bericht: Einstellungen, Kennzahlen, Verdikt. Gibt die Bytes der PDF-Datei zurück."""
    s = a.settings
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Kalman-Filter: Zustand schätzen aus verrauschten Messungen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Sebastian Hanisch, Operations Research und Machine Learning", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Einstellungen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    rows = [
        ("GPS-Rauschen (wahr)", f"{_num(s.sigma_z, 1)} m"),
        ("Beschleunigungsrauschen (wahr)", f"{_num(s.q, 2)} m/s²"),
        ("Annahme des Filters zu Q / R", f"x{_num(s.q_factor, 2)} / x{_num(s.r_factor, 2)}"),
        ("Tunnel", f"{s.tunnel} s"),
        ("Radsensor", s.wheel),
        ("Zufalls-Seed", str(s.seed)),
    ]
    for k, v in rows:
        pdf.cell(95, 6, k)
        pdf.cell(0, 6, v, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Kennzahlen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    metrics = [
        ("Positionsfehler GPS-Rohwert (RMSE)", f"{_num(a.rmse_raw)} m"),
        ("Positionsfehler Filter (RMSE)", f"{_num(a.rmse_kf)} m"),
        ("Geschwindigkeitsfehler Filter (RMSE)", f"{_num(a.rmse_vel)} m/s"),
        ("Mittlere NEES (Soll 2)", _num(a.nees_mean)),
        ("Anteil im 95-%-Band", f"{_num(100 * a.coverage, 1)} %"),
    ]
    for k, v in metrics:
        pdf.cell(95, 6, k)
        pdf.cell(0, 6, v, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, "Synthetische Fahrt mit bekanntem Modell; die Zahlen gelten für dieses Szenario und diese Einstellungen.")
    return bytes(pdf.output())
