"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math
import numpy as np
import plotly.graph_objects as go

import kal_constants as C


def _ref_line(fig, value, text, position, axis, log=True, **line_kwargs):
    """Referenzlinie mit Beschriftung auf einer logarithmischen Achse: die Linie liegt auf dem Datenwert, die Beschriftung braucht log10 des Werts
    (add_hline und add_vline setzen beide auf denselben Wert, die Beschriftung sitzt dann falsch und dehnt die Achse); log=False für lineare Achsen."""
    if log and not value > 0.0:
        return
    pos = math.log10(value) if log else value
    top, left = "top" in position, "left" in position
    if axis == "y":
        fig.add_hline(y=value, **line_kwargs)
        if text:
            fig.add_annotation(xref="paper", yref="y", x=0 if left else 1, y=pos, text=text, showarrow=False, xanchor="left" if left else "right", yanchor="bottom" if top else "top")
    else:
        fig.add_vline(x=value, **line_kwargs)
        if text:
            fig.add_annotation(xref="x", yref="paper", x=pos, y=1 if top else 0, text=text, showarrow=False, xanchor="right" if left else "left", yanchor="top" if top else "bottom")


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=70), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _tunnel_rect(fig, track):
    idx = np.where(track.tunnel)[0]
    if len(idx):
        fig.add_vrect(x0=track.t[idx[0]], x1=track.t[idx[-1]], fillcolor=C.COLOR_TUNNEL, line_width=0, annotation_text="Tunnel (kein GPS)",
                      annotation_position="top left")


def build_error_chart(a):
    """Positionsfehler gegen die Wahrheit: GPS-Rohwert, Filter und das 95-%-Band, das der Filter sich selbst zuschreibt."""
    t = a.track.t
    band = 1.96 * a.sd_pos
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[t, t[::-1]], y=np.r_[band, -band[::-1]], fill="toself", fillcolor=C.COLOR_BAND, line=dict(width=0),
                             hoverinfo="skip", name="95-%-Band des Filters"))
    fig.add_trace(go.Scatter(x=t, y=a.err_raw, mode="markers", marker=dict(color=C.COLOR_RAW, size=5), name="GPS-Rohwert",
                             hovertemplate="t = %{x:.0f} s: %{y:.1f} m<extra>GPS</extra>"))
    fig.add_trace(go.Scatter(x=t, y=a.err_pos, mode="lines", line=dict(color=C.COLOR_FILTER, width=3), name="Filter",
                             hovertemplate="t = %{x:.0f} s: %{y:.1f} m<extra>Filter</extra>"))
    _tunnel_rect(fig, a.track)
    fig.update_xaxes(title_text="Zeit (s)")
    fig.update_yaxes(title_text="Positionsfehler (m)")
    return _base(fig)


def build_speed_chart(a):
    t = a.track.t
    band = 1.96 * a.sd_vel
    est = a.run.x[:, 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[t, t[::-1]], y=np.r_[est + band, (est - band)[::-1]], fill="toself", fillcolor=C.COLOR_BAND,
                             line=dict(width=0), hoverinfo="skip", name="95-%-Band des Filters"))
    if a.settings.wheel == "an":
        fig.add_trace(go.Scatter(x=t, y=a.track.w, mode="markers", marker=dict(color=C.COLOR_WHEEL, size=4), name="Radsensor",
                                 hovertemplate="t = %{x:.0f} s: %{y:.2f} m/s<extra>Rad</extra>"))
    fig.add_trace(go.Scatter(x=t, y=a.track.v, mode="lines", line=dict(color=C.COLOR_TRUTH, width=2, dash="dot"), name="Wahrheit",
                             hovertemplate="t = %{x:.0f} s: %{y:.2f} m/s<extra>Wahrheit</extra>"))
    fig.add_trace(go.Scatter(x=t, y=est, mode="lines", line=dict(color=C.COLOR_FILTER, width=3), name="Filter",
                             hovertemplate="t = %{x:.0f} s: %{y:.2f} m/s<extra>Filter</extra>"))
    _tunnel_rect(fig, a.track)
    fig.update_xaxes(title_text="Zeit (s)")
    fig.update_yaxes(title_text="Geschwindigkeit (m/s)")
    return _base(fig)


def build_gain_chart(a):
    """Verstärkung des GPS-Updates über die Zeit und ihr stationärer Wert."""
    t = a.track.t
    k = a.run.k_gps
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=k[:, 0], mode="lines+markers", marker=dict(size=4), line=dict(color=C.COLOR_FILTER, width=2),
                             name="Verstärkung Position", connectgaps=False, hovertemplate="t = %{x:.0f} s: %{y:.4f}<extra>K Position</extra>"))
    fig.add_hline(y=float(a.k_steady[0]), line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text="stationärer Wert",
                  annotation_position="bottom right")
    _tunnel_rect(fig, a.track)
    fig.update_xaxes(title_text="Zeit (s)")
    fig.update_yaxes(title_text="Gewicht der GPS-Messung auf die Position")
    return _base(fig, 300)


def build_eta_chart(a):
    t = a.track.t
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=a.eta_raw, mode="markers", marker=dict(color=C.COLOR_RAW, size=5), name="aus GPS-Rohwerten",
                             hovertemplate="t = %{x:.0f} s: %{y:.0f} s<extra>Roh</extra>"))
    fig.add_trace(go.Scatter(x=t, y=a.eta_true, mode="lines", line=dict(color=C.COLOR_TRUTH, width=2, dash="dot"), name="Wahrheit"))
    fig.add_trace(go.Scatter(x=t, y=a.eta_kf, mode="lines", line=dict(color=C.COLOR_FILTER, width=3), name="aus dem Filter",
                             hovertemplate="t = %{x:.0f} s: %{y:.0f} s<extra>Filter</extra>"))
    _tunnel_rect(fig, a.track)
    top = float(np.nanmax(a.eta_true)) * 1.5 if np.any(~np.isnan(a.eta_true)) else 300.0
    fig.update_xaxes(title_text="Zeit (s)")
    fig.update_yaxes(title_text="Restfahrzeit bis zum Ziel (s)", range=[0, top])
    return _base(fig)


def build_sigma_sweep(rows):
    xs = [r["sigma"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["raw"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_RAW, width=3), name="GPS-Rohwert"))
    fig.add_trace(go.Scatter(x=xs, y=[r["kf_aus"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_FILTER, width=3, dash="dash"),
                             name="Filter nur mit GPS"))
    fig.add_trace(go.Scatter(x=xs, y=[r["kf_an"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_FILTER, width=3),
                             name="Filter mit GPS und Radsensor"))
    fig.update_xaxes(title_text="Wahres GPS-Rauschen (m)")
    fig.update_yaxes(title_text="Positionsfehler, RMSE (m)")
    return _base(fig)


def build_assumption_sweep(sw):
    fig = go.Figure()
    for key, label, color in (("q", "Prozessrauschen Q falsch angenommen", C.COLOR_FILTER), ("r", "Messrauschen R falsch angenommen", C.COLOR_WHEEL)):
        fig.add_trace(go.Scatter(x=[r["factor"] for r in sw[key]], y=[r["nees"] for r in sw[key]], mode="lines+markers",
                                 line=dict(color=color, width=3), name=label,
                                 hovertemplate="Faktor %{x}: NEES %{y:.2f}<extra></extra>"))
    _ref_line(fig, C.STATE_DIM, "Soll = 2", "bottom right", "y", line=dict(color=C.COLOR_TRUTH, dash="dash"))
    fig.update_xaxes(title_text="Angenommen / wahr (Faktor)", type="log")
    fig.update_yaxes(title_text="Mittlere NEES (logarithmisch)", type="log")
    return _base(fig)


def build_tunnel_sweep(rows):
    rows = [r for r in rows if r["tunnel"] > 0]
    xs = [r["tunnel"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["err_aus"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_FILTER, width=3, dash="dash"),
                             name="Fehler ohne Radsensor"))
    fig.add_trace(go.Scatter(x=xs, y=[r["sd_aus"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_RAW, width=2, dash="dash"),
                             name="Unsicherheit (1 Sigma) ohne Radsensor"))
    fig.add_trace(go.Scatter(x=xs, y=[r["err_an"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_FILTER, width=3),
                             name="Fehler mit Radsensor"))
    fig.add_trace(go.Scatter(x=xs, y=[r["sd_an"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_RAW, width=2),
                             name="Unsicherheit (1 Sigma) mit Radsensor"))
    fig.update_xaxes(title_text="Tunnellänge (s)")
    fig.update_yaxes(title_text="Position am Tunnelende (m)")
    return _base(fig)
