"""Current-derived mode: source normalization -> shared plume propagation -> collection.

Each stage is a separate pure function. The current only sets n0; downstream
geometry and density use the same plume_model.calculate_plume as direct-density
mode, and collection uses only the beam current and the two radii.
"""

from __future__ import annotations

from dataclasses import dataclass

from collection import CollectionResult, calculate_collection
from constants import MODELED_CHARGE_C
from plume_model import PlumeResult, calculate_plume, from_scaled, scaled_product
from source_model import SourceResult, calculate_source_from_current


@dataclass(frozen=True)
class CurrentDerivedResult:
    """Source, propagation and collection results of one validated calculation."""

    source: SourceResult
    plume: PlumeResult
    collection: CollectionResult

    @property
    def n2_from_current(self) -> float:
        """Independent check n2 = I0 / (q v0 A2), particles/m^3."""
        s = self.source
        return from_scaled(*scaled_product([s.current_a], [s.charge_c, s.velocity_m_s, self.plume.area2_m2]))

    @property
    def downstream_beam_current_a(self) -> float:
        """Full beam current n2 q v0 A2 at x = l, A; equals I0 (it is not reduced by F)."""
        s = self.source
        return from_scaled(*scaled_product([self.plume.n2, s.charge_c, s.velocity_m_s, self.plume.area2_m2]))


def calculate_current_derived(
    current_a: float,
    voltage_v: float,
    mass_kg: float,
    r0_m: float,
    distance_m: float,
    theta_rad: float,
    collector_radius_m: float,
    charge_c: float = MODELED_CHARGE_C,
) -> CurrentDerivedResult:
    """Run the full current-derived calculation with SI inputs.

    Args:
        current_a: |I0|, A (> 0).
        voltage_v: |V0|, V (> 0).
        mass_kg: individual modeled particle mass, kg (> 0).
        r0_m: source radius, m (> 0).
        distance_m: axial distance l, m (>= 0).
        theta_rad: half-angle, rad (0 <= theta < pi/2).
        collector_radius_m: centered collector radius at x = l, m (> 0).
        charge_c: modeled charge magnitude, C (fixed to e in the UI).
    """
    source = calculate_source_from_current(current_a, voltage_v, mass_kg, r0_m, charge_c)
    plume = calculate_plume(source.n0, source.r0_m, distance_m, theta_rad)
    collection = calculate_collection(source.current_a, plume.r2_m, collector_radius_m)
    return CurrentDerivedResult(source=source, plume=plume, collection=collection)
