"""Source normalization for current-derived mode: n0 from beam current (pure SI).

    ideal source energy        (1/2) m v0^2 = q V0
    characteristic velocity    v0 = sqrt(2 q V0 / m)
    planar source current      I0 = n0 q v0 A0,   A0 = pi r0^2
    source density             n0 = I0 / (q v0 A0) = [I0/(q pi r0^2)] sqrt(m/(2 q V0))

Inputs are positive SI magnitudes: current_a (A), voltage_v (V), mass_kg (kg, one
modeled particle), source_radius_m (m), charge_c (C). The UI fixes charge_c to
one elementary charge. A future charge state z only changes charge_c = z*e; the
plume propagation law in plume_model.py is unaffected.

v0 is an idealized characteristic velocity (negligible initial kinetic energy,
nonrelativistic), used as the constant axial flux velocity. It is not a measured
plume-average velocity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from constants import MODELED_CHARGE_C, SPEED_OF_LIGHT_M_S
from plume_model import (
    PlumeModelError,
    from_scaled,
    named_area,
    require_normal,
    require_positive,
    scaled_product,
)

# Speeds at or above this fraction of c get a notice that the nonrelativistic energy
# relation is approximate. This is a notice, not a validity boundary.
RELATIVISTIC_NOTICE_FRACTION = 0.01


@dataclass(frozen=True)
class SourceResult:
    """Normalized source inputs and the derived source density.

    current_a: |I0| assigned to the modeled population through A0, A.
    voltage_v: |V0| accelerating potential difference, V.
    mass_kg: mass of one modeled particle, kg.
    charge_c: modeled charge magnitude q, C.
    r0_m, area0_m2: source radius (m) and full circular source area (m^2).
    kinetic_energy_j: q V0, J.
    velocity_m_s: idealized characteristic velocity v0, m/s.
    n0: uniform source number density, particles/m^3.
    """

    current_a: float
    voltage_v: float
    mass_kg: float
    charge_c: float
    r0_m: float
    area0_m2: float
    kinetic_energy_j: float
    velocity_m_s: float
    n0: float

    @property
    def speed_fraction_of_c(self) -> float:
        """v0 / c (presentation only)."""
        return self.velocity_m_s / SPEED_OF_LIGHT_M_S

    @property
    def relativistic_notice(self) -> bool:
        """True when v0 >= 1 % of c, where the (1/2) m v^2 relation is noticeably approximate."""
        return self.speed_fraction_of_c >= RELATIVISTIC_NOTICE_FRACTION


def _require_positive_normal(value, name: str, field: str, quantity: str) -> float:
    v = require_positive(value, name, field)
    return require_normal(v, quantity, name)


def characteristic_velocity(voltage_v: float, mass_kg: float, charge_c: float = MODELED_CHARGE_C) -> float:
    """Idealized velocity v0 = sqrt(2 q V0 / m) in m/s.

    Args:
        voltage_v: accelerating potential difference magnitude, V (> 0).
        mass_kg: mass of one modeled particle, kg (> 0).
        charge_c: modeled charge magnitude, C (> 0).

    Raises:
        PlumeModelError: if an input is outside its domain, or if v0 reaches the
            speed of light (the nonrelativistic model does not apply).
        PlumeRangeError: if v0 is not representable.
    """
    voltage_v = _require_positive_normal(voltage_v, "Accelerating voltage magnitude |V0|", "voltage", "V0")
    mass_kg = _require_positive_normal(mass_kg, "Particle mass m_i", "mass", "m_i")
    charge_c = _require_positive_normal(charge_c, "Particle charge q", "charge", "q")
    # v0^2 = 2 q V0 / m evaluated in scaled form so intermediates cannot over/underflow;
    # an even exponent lets the square root be taken exactly as sqrt(mantissa) * 2**(e/2).
    mantissa, exponent = scaled_product([2.0, charge_c, voltage_v], [mass_kg])
    if exponent % 2:
        mantissa, exponent = mantissa * 2.0, exponent - 1
    velocity = require_normal(
        from_scaled(math.sqrt(mantissa), exponent // 2), "v0", "Characteristic velocity v0"
    )
    if velocity >= SPEED_OF_LIGHT_M_S:
        raise PlumeModelError(
            f"The idealized velocity v0 = sqrt(2qV0/m) = {velocity:.4g} m/s reaches or exceeds the speed of "
            "light, so the nonrelativistic source model does not apply. Use a lower voltage or a larger mass.",
            "velocity",
        )
    return velocity


def source_current(n0: float, velocity_m_s: float, area0_m2: float, charge_c: float = MODELED_CHARGE_C) -> float:
    """Planar current I = n q v A in A (used to reconstruct I0 or check conservation)."""
    n0 = require_positive(n0, "Number density", "n0")
    velocity_m_s = require_positive(velocity_m_s, "Velocity", "velocity")
    area0_m2 = require_positive(area0_m2, "Area", "r0")
    charge_c = require_positive(charge_c, "Particle charge q", "charge")
    return require_normal(
        from_scaled(*scaled_product([n0, charge_c, velocity_m_s, area0_m2])), "I", "Current n q v A"
    )


def calculate_source_from_current(
    current_a: float,
    voltage_v: float,
    mass_kg: float,
    source_radius_m: float,
    charge_c: float = MODELED_CHARGE_C,
) -> SourceResult:
    """Derive the uniform source density n0 = I0 / (q v0 A0).

    Args:
        current_a: total beam current magnitude |I0| through A0, A (> 0).
        voltage_v: accelerating potential difference magnitude |V0|, V (> 0).
        mass_kg: mass of one modeled ion/particle, kg (> 0); not the beam mass.
        source_radius_m: finite source radius r0, m (> 0).
        charge_c: modeled charge magnitude q, C (> 0); one elementary charge by default.

    Returns:
        SourceResult in SI units.

    Raises:
        PlumeModelError: for inputs outside the domain (including v0 >= c).
        PlumeRangeError: when a derived quantity is not representable.
    """
    current_a = _require_positive_normal(current_a, "Beam current magnitude |I0|", "current", "I0")
    voltage_v = _require_positive_normal(voltage_v, "Accelerating voltage magnitude |V0|", "voltage", "V0")
    mass_kg = _require_positive_normal(mass_kg, "Particle mass m_i", "mass", "m_i")
    charge_c = _require_positive_normal(charge_c, "Particle charge q", "charge", "q")
    source_radius_m = require_positive(source_radius_m, "Source radius r0", "r0")

    area0 = named_area(source_radius_m, "area0", "Source area A0")
    energy = require_normal(
        from_scaled(*scaled_product([charge_c, voltage_v])), "qV0", "Source kinetic energy qV0"
    )
    velocity = characteristic_velocity(voltage_v, mass_kg, charge_c)
    n0 = require_normal(
        from_scaled(*scaled_product([current_a], [charge_c, velocity, area0])),
        "n0",
        "Source density n0 = I0/(q v0 A0)",
    )
    return SourceResult(
        current_a=current_a,
        voltage_v=voltage_v,
        mass_kg=mass_kg,
        charge_c=charge_c,
        r0_m=source_radius_m,
        area0_m2=area0,
        kinetic_energy_j=energy,
        velocity_m_s=velocity,
        n0=n0,
    )
