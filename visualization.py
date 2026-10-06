"""Matplotlib figures for the plume geometry and the density profile.

Functions return a Figure built from one validated PlumeResult; callers should
close figures (matplotlib.pyplot.close) after rendering them.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Arc, Polygon, Rectangle  # noqa: E402

from formatting import format_area, format_length, format_sig  # noqa: E402
from plume_model import PlumeResult  # noqa: E402

SCHEMATIC_LABEL = "Schematic, not to scale"
TO_SCALE_LABEL = "Drawn to scale (equal axes)"
COLLECTOR_LABEL = "Collector"

# A to-scale drawing stays readable for collectors up to this multiple of the beam radius.
COLLECTOR_MAX_SCALE_RATIO = 4.0

_FILL = "#dbe9f6"
_EDGE = "#1f4e79"
_ACCENT = "#b03a2e"
_MUTED = "#555555"
_COLLECTOR = "#c26a00"


def is_to_scale_readable(result: PlumeResult, collector_radius_m: float | None = None) -> bool:
    """True when an equal-axis drawing stays readable (no extreme aspect ratio)."""
    if collector_radius_m is not None and collector_radius_m / result.r2_m > COLLECTOR_MAX_SCALE_RATIO:
        return False
    if result.distance_m == 0.0:
        return True
    aspect = result.r2_m / result.distance_m
    return (1 / 20) <= aspect <= 20 and result.r0_m / result.r2_m >= 1 / 100


def _schematic_collector_radius(display_beam_radius: float, collector_radius_m: float, beam_radius_m: float) -> float:
    """Display half-height for the collector that keeps its size order relative to the beam."""
    ratio = collector_radius_m / beam_radius_m
    if ratio == 1.0:
        return display_beam_radius
    scaled = display_beam_radius * ratio
    if ratio < 1.0:
        return min(max(scaled, 0.12 * display_beam_radius), 0.88 * display_beam_radius)
    return min(max(scaled, 1.12 * display_beam_radius), 1.4 * display_beam_radius)


def _stamp(ax, text: str, schematic: bool) -> None:
    ax.text(
        0.01, 0.99, text, transform=ax.transAxes, ha="left", va="top", fontsize=9,
        fontweight="bold" if schematic else "normal", color=_ACCENT if schematic else _MUTED,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=_ACCENT if schematic else "#aaaaaa"),
    )


def _dim_arrow(ax, p0, p1, text, text_xy, ha="center", va="center", color=_EDGE):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="<->", color=color, lw=1.1))
    ax.text(*text_xy, text, ha=ha, va=va, fontsize=8.5, color=color)


def _collector_text(collector_radius_m: float) -> str:
    return (f"{COLLECTOR_LABEL} at x = l (centered,\nperpendicular to the axis)\n"
            f"r_coll = {format_length(collector_radius_m)}\n"
            f"A_coll = {format_area(math.pi * collector_radius_m ** 2)}")


def geometry_figure(result: PlumeResult, view: str = "auto", illustrative: bool = False,
                    collector_radius_m: float | None = None):
    """Side view of the conical frustum.

    Args:
        result: validated model result; every label shows its true value.
        view: "auto" draws to scale with equal axes when readable, otherwise a
            schematic; "schematic" always draws the schematic.
        illustrative: add an "Illustrative example values" note.
        collector_radius_m: optional centered collector radius at x = l, m. When
            given, the collector is drawn in place of the generic instrument marker.

    Returns:
        (figure, is_schematic)
    """
    figsize = (8.0, 4.4) if collector_radius_m is None else (9.6, 4.6)  # room for the collector label
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    if result.distance_m == 0.0:
        # A single shared plane; only an oversized collector forces a schematic.
        schematic = not is_to_scale_readable(result, collector_radius_m)
        _draw_zero_distance(ax, result, schematic, collector_radius_m)
    else:
        schematic = view == "schematic" or not is_to_scale_readable(result, collector_radius_m)
        _draw_frustum(ax, result, schematic, collector_radius_m)
    _stamp(ax, SCHEMATIC_LABEL if schematic else TO_SCALE_LABEL, schematic)
    if illustrative:
        ax.text(0.99, 0.99, "Illustrative example values", transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5, style="italic", color=_MUTED)
    ax.set_title("Plume geometry: side view of the conical frustum", fontsize=10.5)
    return fig, schematic


def _draw_frustum(ax, result: PlumeResult, schematic: bool, collector_radius_m: float | None) -> None:
    if schematic:
        # Display coordinates: fixed length, readable radii, finite inlet (never an apex).
        L = 1.0
        R2 = 0.32
        if result.r2_m == result.r0_m:
            R0 = R2
        else:
            R0 = min(max(R2 * result.r0_m / result.r2_m, 0.06), R2 - 0.08)
        RC = None if collector_radius_m is None else _schematic_collector_radius(R2, collector_radius_m, result.r2_m)
    else:
        L, R0, R2 = result.distance_m, result.r0_m, result.r2_m
        RC = collector_radius_m

    H = R2 if RC is None else max(R2, RC)  # vertical half-extent of the drawing
    span = max(L, 2 * H)
    ax.add_patch(Polygon([(0, -R0), (0, R0), (L, R2), (L, -R2)], closed=True,
                         fc=_FILL, ec=_EDGE, lw=1.6))
    # Cross-section planes A0 and A2 (full plume footprints).
    ax.plot([0, 0], [-R0, R0], color=_EDGE, lw=3)
    ax.plot([L, L], [-R2, R2], color=_EDGE, lw=3)

    # Centerline at y = h, with P0 and P2 on it.
    inst_x0, inst_w = L + 0.10 * span, 0.12 * span
    ax.plot([-0.08 * span, inst_x0], [0, 0], ls="--", color=_MUTED, lw=1)
    ax.text(0.5 * L, -0.06 * R2 - 0.02 * span, "centerline (y = h; h is not a model input)",
            ha="center", va="top", fontsize=8, color=_MUTED)
    ax.plot([0, L], [0, 0], "o", color=_EDGE, ms=5)
    ax.text(-0.02 * span, -0.03 * span, "P₀\nx = 0", ha="right", va="top", fontsize=9)
    if RC is None:
        ax.text(L + 0.015 * span, -0.03 * span, "P₂\nx = l", ha="left", va="top", fontsize=9)
    else:  # keep P2 clear of the collector drawn just downstream
        ax.text(L - 0.015 * span, -0.03 * span, "P₂\nx = l", ha="right", va="top", fontsize=9)

    # Radii.
    _dim_arrow(ax, (-0.035 * span, 0), (-0.035 * span, R0),
               f"r₀ = {format_length(result.r0_m)}", (-0.05 * span, R0 + 0.02 * span), ha="right", va="bottom")
    _dim_arrow(ax, (L - 0.03 * span, 0), (L - 0.03 * span, R2),
               f"r₂ = {format_length(result.r2_m)}", (L - 0.045 * span, 0.62 * R2), ha="right")

    # Plane areas.
    ax.text(0, -R0 - 0.16 * span, f"A₀ = {format_area(result.area0_m2)}", ha="center", va="top",
            fontsize=8.5, color=_EDGE)
    ax.text(L, -H - 0.05 * span, f"A₂ = {format_area(result.area2_m2)}\n(full plume footprint)",
            ha="center", va="top", fontsize=8.5, color=_EDGE)

    # Axial distance l.
    y_l = H + 0.12 * span
    _dim_arrow(ax, (0, y_l), (L, y_l), f"l = {format_length(result.distance_m)} (axial)",
               (0.5 * L, y_l + 0.02 * span), va="bottom")

    # Half-angle: construction line parallel to the axis from the inlet edge.
    ax.plot([0, 0.45 * L], [R0, R0], ls=":", color=_ACCENT, lw=1.3)
    display_angle = math.degrees(math.atan2(R2 - R0, L))
    if result.theta_rad == 0.0:
        ax.text(0.47 * L, R0 + 0.015 * span, "θ = 0° (cylindrical plume)", fontsize=9, color=_ACCENT,
                va="bottom")
    else:
        rho = 0.3 * L
        ax.add_patch(Arc((0, R0), 2 * rho, 2 * rho, theta1=0, theta2=display_angle, color=_ACCENT, lw=1.3))
        mid = math.radians(display_angle / 2)
        ax.text(rho * 1.06 * math.cos(mid), R0 + rho * 1.06 * math.sin(mid),
                f"θ = {format_sig(result.theta_deg, 3)}° (half-angle)", fontsize=9, color=_ACCENT,
                va="center", ha="left")

    if RC is None:
        # Generic instrument location, beyond the downstream plane; its aperture is not A2.
        h_inst = 0.5 * R2 + 0.05 * span
        ax.add_patch(Rectangle((inst_x0, -h_inst), inst_w, 2 * h_inst, fc="#eeeeee", ec=_MUTED, lw=1))
        ax.text(inst_x0 + inst_w / 2, h_inst + 0.02 * span, "Instrument\nlocation\n(aperture ≠ A₂)",
                ha="center", va="bottom", fontsize=8, color=_MUTED)
        x_right = inst_x0 + inst_w + 0.08 * span
    else:
        # Centered collector in the plane x = l, drawn just downstream of A2 so both stay visible.
        x_c = L + 0.025 * span
        ax.plot([x_c, x_c], [-RC, RC], color=_COLLECTOR, lw=4, solid_capstyle="butt")
        ax.plot([x_c], [0], marker="o", color=_COLLECTOR, ms=4)
        ax.annotate(_collector_text(collector_radius_m), xy=(x_c, RC), xytext=(x_c + 0.06 * span, 0.25 * H),
                    fontsize=8, color=_COLLECTOR, va="bottom", ha="left",
                    arrowprops=dict(arrowstyle="-", color=_COLLECTOR, lw=0.8))
        x_right = x_c + 0.62 * span

    ax.set_aspect("equal")
    ax.set_xlim(-0.35 * span if RC is None else -0.45 * span, x_right)
    ax.set_ylim(-H - 0.22 * span, y_l + 0.12 * span)
    if schematic:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y − h (m)")
        ax.ticklabel_format(style="sci", scilimits=(-3, 4))


def _draw_zero_distance(ax, result: PlumeResult, schematic: bool, collector_radius_m: float | None) -> None:
    """l = 0: one shared cross-section with P0 = P2; no separation is invented."""
    r = 1.0 if schematic else result.r0_m
    rc = None
    if collector_radius_m is not None:
        rc = _schematic_collector_radius(r, collector_radius_m, result.r0_m) if schematic else collector_radius_m
    H = r if rc is None else max(r, rc)
    ax.plot([0, 0], [-r, r], color=_EDGE, lw=3)
    ax.plot([-2.5 * H, 3.2 * H], [0, 0], ls="--", color=_MUTED, lw=1)
    ax.plot([0], [0], "o", color=_EDGE, ms=6)
    ax.text(0.08 * H, -0.08 * H, "P₀ = P₂ (x = 0 = l)", ha="left", va="top", fontsize=9)
    _dim_arrow(ax, (-0.25 * H, 0), (-0.25 * H, r), f"r₀ = r₂ = {format_length(result.r0_m)}",
               (-0.35 * H, 0.5 * r), ha="right")
    ax.text(0, -1.12 * H, f"A₀ = A₂ = {format_area(result.area0_m2)}", ha="center", va="top", fontsize=8.5,
            color=_EDGE)
    ax.text(0, 1.12 * H, f"l = 0: planes coincide, so n₂ = n₀  (θ = {format_sig(result.theta_deg, 3)}°)",
            ha="center", va="bottom", fontsize=9, color=_ACCENT)
    if rc is None:
        ax.add_patch(Rectangle((1.6 * r, -0.6 * r), 1.0 * r, 1.2 * r, fc="#eeeeee", ec=_MUTED, lw=1))
        ax.text(2.1 * r, 0.65 * r, "Instrument\nlocation", ha="center", va="bottom", fontsize=8, color=_MUTED)
    else:
        x_c = 0.06 * H
        ax.plot([x_c, x_c], [-rc, rc], color=_COLLECTOR, lw=4, solid_capstyle="butt")
        ax.annotate(_collector_text(collector_radius_m), xy=(x_c, 0.8 * rc), xytext=(1.0 * H, 0.35 * H),
                    fontsize=8, color=_COLLECTOR, va="bottom", ha="left",
                    arrowprops=dict(arrowstyle="-", color=_COLLECTOR, lw=0.8))
    ax.set_aspect("equal")
    ax.set_xlim(-3.2 * H, 3.4 * H)
    ax.set_ylim(-1.6 * H, 1.6 * H)
    if schematic:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y − h (m)")
        ax.ticklabel_format(style="sci", scilimits=(-3, 4))


def profile_figure(result: PlumeResult, x_m: np.ndarray, density: np.ndarray, log_y: bool,
                   illustrative: bool = False):
    """Density n(x) versus axial distance x (m) with P0 and P2 marked.

    Args:
        result: validated model result that produced (x_m, density).
        x_m: axial positions in m, from 0 to l inclusive.
        density: n(x) in particles/m^3 (all positive by model validation).
        log_y: logarithmic y-axis when True, linear otherwise. The x-axis is linear.
    """
    fig, ax = plt.subplots(figsize=(8.0, 4.0), layout="constrained")
    if result.distance_m == 0.0:
        ax.plot([0.0], [result.n0], "o", color=_EDGE, ms=7)
        ax.annotate(f"P₀ = P₂: n₀ = n₂ = {format_sig(result.n0)} particles/m³",
                    (0.0, result.n0), xytext=(10, 10), textcoords="offset points", fontsize=9)
        ax.text(0.5, 0.15, "l = 0: the reference and downstream planes coincide.",
                transform=ax.transAxes, ha="center", fontsize=9, color=_ACCENT)
        ax.set_xlim(-1.0, 1.0)
    else:
        ax.plot(x_m, density, color=_EDGE, lw=1.8, label="n(x), uniform model estimate")
        ax.plot([0.0], [result.n0], "o", color=_EDGE, ms=7)
        ax.plot([result.distance_m], [result.n2], "s", color=_ACCENT, ms=7)
        ax.annotate(f"P₀: n₀ = {format_sig(result.n0)}", (0.0, result.n0), xytext=(12, -4),
                    textcoords="offset points", fontsize=9, va="top")
        ax.annotate(f"P₂: n₂ = {format_sig(result.n2)}", (result.distance_m, result.n2),
                    xytext=(-12, 22), textcoords="offset points", fontsize=9, ha="right", color=_ACCENT)
        ax.legend(loc="upper right", fontsize=8.5)
    if log_y:
        ax.set_yscale("log")
    else:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))
    ax.set_xlabel("Axial distance x (m)")
    ax.set_ylabel("Number density n (particles/m³)")
    ax.set_title(f"Density vs axial distance ({'logarithmic' if log_y else 'linear'} y-axis)", fontsize=10.5)
    ax.grid(True, which="both", alpha=0.3)
    if illustrative:
        ax.text(0.01, 0.02, "Illustrative example values", transform=ax.transAxes, fontsize=8.5,
                style="italic", color=_MUTED)
    return fig
