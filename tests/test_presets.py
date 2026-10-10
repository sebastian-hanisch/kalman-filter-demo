"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink begrenzt und rastet ein."""

import kal_constants as C
import kal_evaluation as ev
import kal_presets as P


def _settings_from(preset):
    return ev.Settings(sigma_z=preset["sz"], q=preset["q"], q_factor=preset["qf"], r_factor=preset["rf"], tunnel=preset["tun"],
                       wheel=preset["rad"], seed=preset["seed"])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds():
    for name, p in C.PRESETS.items():
        assert C.SIGMA_Z_MIN <= p["sz"] <= C.SIGMA_Z_MAX, name
        assert C.Q_MIN <= p["q"] <= C.Q_MAX, name
        assert p["qf"] in C.FACTOR_CHOICES and p["rf"] in C.FACTOR_CHOICES, name
        assert C.TUNNEL_MIN <= p["tun"] <= C.TUNNEL_MAX and p["rad"] in C.WHEEL_CHOICES and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert _settings_from(C.PRESETS["Standardfahrt (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_permalink_clamps_numbers_and_snaps_choices():
    sz = P.SETTING_SPECS["sigma_slider"]
    assert P.clamp_value(sz, 1000.0) == C.SIGMA_Z_MAX and P.clamp_value(sz, -5.0) == C.SIGMA_Z_MIN
    qf = P.SETTING_SPECS["qf_select"]
    assert P.clamp_value(qf, 2.0) == 2.0 and P.clamp_value(qf, 3.0) is None
    rad = P.SETTING_SPECS["wheel_select"]
    assert P.clamp_value(rad, "an") == "an" and P.clamp_value(rad, "kaputt") is None


def test_the_presets_show_what_their_help_text_claims():
    """Kein Preset mit vertauschter Beschriftung: 'zu sicher' hat kleinere, nicht größere Annahmen; 'schlechtes GPS' mehr Rauschen."""
    base = C.PRESETS["Standardfahrt (Voreinstellung)"]
    assert C.PRESETS["Filter zu sicher eingestellt"]["qf"] < base["qf"] and C.PRESETS["Filter zu sicher eingestellt"]["rf"] < base["rf"]
    assert C.PRESETS["Schlechtes GPS"]["sz"] > base["sz"]
    long = C.PRESETS["Langer Tunnel, ohne Radsensor"]
    assert long["tun"] == C.TUNNEL_MAX and long["rad"] == "aus"
