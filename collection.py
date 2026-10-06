"""Ideal centered circular collector at x = l (pure SI, independent of mass and voltage).

    collector area       A_coll = pi r_coll^2
    sampled fraction     f_coll = min(1, A_coll/A2) = 1 if r_coll >= r2 else (r_coll/r2)^2
    collected current    I_coll = I0 * f_coll

Assumes a circular collector centered on the axis in the plane perpendicular to it
at x = l, a uniform axial current density across the full beam footprint A2, and
ideal collection of every particle crossing the overlap. A collector larger than the
beam collects no current outside the beam. f_coll is a separate quantity from the
density fraction F = A0/A2, even when the two happen to be equal. I_coll is an ideal
primary-current magnitude, not a calibrated detector reading or secondary current.

Offsets, tilt or radial profiles would replace collection_fraction without changing
source normalization or plume propagation.
"""

from __future__ import annotations

from dataclasses import dataclass

from plume_model import PlumeRangeError, named_area, require_normal, require_positive


@dataclass(frozen=True)
class CollectionResult:
    """Collector geometry and ideal collected current.

    collector_radius_m, collector_area_m2: r_coll (m) and A_coll (m^2).
    beam_radius_m: full downstream beam radius r2 at the collector plane, m.
    fraction: f_coll; uncollected_fraction: 1 - f_coll (cancellation-free).
    source_current_a: |I0|, A; collected_current_a: I_coll = I0 f_coll, A.
    """

    collector_radius_m: float
    collector_area_m2: float
    beam_radius_m: float
    fraction: float
    uncollected_fraction: float
    source_current_a: float
    collected_current_a: float

    @property
    def percent(self) -> float:
        """Collected percentage 100 f_coll."""
        return 100.0 * self.fraction

    @property
    def saturated(self) -> bool:
        """True when r_coll >= r2, so the collector covers the whole modeled beam."""
        return self.collector_radius_m >= self.beam_radius_m


def collection_fraction(beam_radius_m: float, collector_radius_m: float) -> float:
    """Sampled current fraction f_coll for a centered circular collector and uniform beam.

    The radii are compared before squaring, so an enormous collector returns exactly 1
    without forming an overflowing A_coll/A2 ratio.

    Args:
        beam_radius_m: full beam radius r2 at the collector plane, m (> 0).
        collector_radius_m: collector radius r_coll, m (> 0).
    """
    beam_radius_m = require_positive(beam_radius_m, "Beam radius r2", "r2")
    collector_radius_m = require_positive(collector_radius_m, "Collector radius r_coll", "collector")
    if collector_radius_m >= beam_radius_m:
        return 1.0
    ratio = collector_radius_m / beam_radius_m
    return require_normal(ratio * ratio, "f_coll", "Collected fraction f_coll = (r_coll/r2)^2")


def _uncollected_fraction(beam_radius_m: float, collector_radius_m: float) -> float:
    """1 - f_coll = ((r2 - r_coll)/r2) * (1 + r_coll/r2), free of cancellation near saturation."""
    if collector_radius_m >= beam_radius_m:
        return 0.0
    return ((beam_radius_m - collector_radius_m) / beam_radius_m) * (1.0 + collector_radius_m / beam_radius_m)


def calculate_collection(source_current_a: float, beam_radius_m: float, collector_radius_m: float) -> CollectionResult:
    """Ideal collected current for a centered circular collector at x = l.

    Args:
        source_current_a: total conserved beam current magnitude |I0|, A (> 0).
        beam_radius_m: full downstream beam radius r2, m (> 0).
        collector_radius_m: collector radius r_coll, m (> 0).

    Raises:
        PlumeModelError: for inputs outside the domain.
        PlumeRangeError: if A_coll, f_coll or I_coll is not representable.
    """
    source_current_a = require_positive(source_current_a, "Beam current magnitude |I0|", "current")
    beam_radius_m = require_positive(beam_radius_m, "Beam radius r2", "r2")
    collector_radius_m = require_positive(collector_radius_m, "Collector radius r_coll", "collector")
    try:
        collector_area = named_area(collector_radius_m, "area_coll", "Collector area A_coll")
    except PlumeRangeError as exc:
        raise PlumeRangeError(str(exc), exc.quantity, "collector") from exc
    fraction = collection_fraction(beam_radius_m, collector_radius_m)
    collected = require_normal(
        source_current_a * fraction, "I_coll", "Collected current I_coll = I0 f_coll"
    )
    return CollectionResult(
        collector_radius_m=collector_radius_m,
        collector_area_m2=collector_area,
        beam_radius_m=beam_radius_m,
        fraction=fraction,
        uncollected_fraction=_uncollected_fraction(beam_radius_m, collector_radius_m),
        source_current_a=source_current_a,
        collected_current_a=collected,
    )
