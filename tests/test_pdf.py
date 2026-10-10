"""PDF-Export: erzeugt gültige PDF-Bytes mit Umlauten, ohne Gedankenstrich-Absturz."""

import kal_evaluation as ev
from kal_pdf_export import build_pdf


def test_pdf_is_a_pdf_for_default_and_extreme_settings():
    for s in (ev.Settings(), ev.Settings(sigma_z=30.0, tunnel=40, wheel="aus", q_factor=0.1, r_factor=0.25)):
        data = build_pdf(ev.analyse(s, steady=False))
        assert data[:5] == b"%PDF-" and len(data) > 1000
