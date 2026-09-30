"""Pure SI physics for the ion plume geometric expansion model.

The model conserves the particle flow of one tracked population through an
axisymmetric conical frustum with equal mean axial velocity at both planes:

    n1 * A1 * v_x = n2 * A2 * v_x   =>   n2 = n1 * (r1 / (r1 + l * tan(theta)))**2

Conventions (all SI, 64-bit floats):
    n1, n2      uniform number densities, particles/m^3
    r1_m, r2_m  circular plume radii, m
    area1_m2,
    area2_m2    full circular plume cross-sections, m^2 (A2 is not a detector aperture)
    distance_m  axial separation l between P1 (x = 0) and P2 (x = l), m
    theta_rad   plume half-angle from the axis, radians; 0 <= theta < pi/2

This module has no UI dependencies so it can be used from tests or batch scripts.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from numbers import Real

import numpy as np

HIGH_ANGLE_SENSITIVITY_DEG = 85.0
HIGH_ANGLE_SENSITIVITY_RAD = HIGH_ANGLE_SENSITIVITY_DEG * math.pi / 180.0
MIN_PROFILE_SAMPLES = 400

# Smallest positive float64 that keeps full precision. Results below this are
# subnormal (reduced precision) or zero and are treated as unrepresentable.
_MIN_NORMAL = sys.float_info.min


class PlumeModelError(ValueError):
    """Raised for inputs outside the supported physical domain.

    Attributes:
        field: input name the error refers to ("n1", "r1", "distance",
            "theta", "x"), or None when it is not tied to one input.
    """

    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.field = field


class PlumeRangeError(PlumeModelError):
    """Raised when a valid input or derived quantity cannot be represented.

    Attributes:
        quantity: symbol of the derived quantity that overflowed, underflowed
            or rounded to zero (for example "r2", "area2", "n2").
    """

    def __init__(self, message: str, quantity: str, field: str | None = None):
        super().__init__(message, field)
        self.quantity = quantity


@dataclass(frozen=True)
class PlumeResult:
    """Normalized SI inputs and all outputs of one validated calculation.

    Inputs: n1 (particles/m^3), r1_m (m), distance_m (m), theta_rad (rad).
    Outputs: r2_m and delta_r_m (m), area1_m2 and area2_m2 (m^2), n2
    (particles/m^3), expansion E = A2/A1, remaining_fraction F = A1/A2 = n2/n1,
    reduction_fraction 1 - F, and the corresponding percentages.
    """

    n1: float
    r1_m: float
    distance_m: float
    theta_rad: float
    delta_r_m: float
    r2_m: float
    area1_m2: float
    area2_m2: float
    n2: float
    expansion: float
    remaining_fraction: float
    reduction_fraction: float
    remaining_percent: float
    reduction_percent: float

    @property
    def theta_deg(self) -> float:
        """Half-angle in degrees (presentation only)."""
        return self.theta_rad * 180.0 / math.pi

    @property
    def high_angle_sensitivity(self) -> bool:
        """True for theta >= 85 deg, where tan(theta) makes n2 highly sensitive.

        This is a numerical-sensitivity note, not a physical validity boundary.
        """
        return self.theta_rad >= HIGH_ANGLE_SENSITIVITY_RAD


def _real(value, name: str, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise PlumeModelError(f"{name} must be a real number.", field)
    v = float(value)
    if not math.isfinite(v):
        raise PlumeModelError(f"{name} must be a finite number.", field)
    return v


def validate_domain(n1, r1_m, distance_m, theta_rad) -> tuple[float, float, float, float]:
    """Check the supported domain and return the inputs as floats.

    Args:
        n1: initial density at P1, particles/m^3; finite and > 0.
        r1_m: source radius, m; finite and > 0.
        distance_m: axial distance, m; finite and >= 0.
        theta_rad: half-angle, rad; finite and 0 <= theta < pi/2.

    Raises:
        PlumeModelError: if any value is non-real, NaN, infinite or out of range.
    """
    n1 = _real(n1, "Initial density n1", "n1")
    r1_m = _real(r1_m, "Source radius r1", "r1")
    distance_m = _real(distance_m, "Axial distance l", "distance")
    theta_rad = _real(theta_rad, "Half-angle theta", "theta")
    if n1 <= 0.0:
        raise PlumeModelError("Initial density n1 must be greater than zero.", "n1")
    if r1_m <= 0.0:
        raise PlumeModelError("Source radius r1 must be greater than zero.", "r1")
    if distance_m < 0.0:
        raise PlumeModelError("Axial distance must be zero or greater.", "distance")
    if not (0.0 <= theta_rad < math.pi / 2.0):
        raise PlumeModelError(
            "Half-angle must be at least 0 deg and less than 90 deg (0 <= theta < pi/2 rad).", "theta"
        )
    return n1, r1_m, distance_m, theta_rad


def _radial_growth(distance_m: float, theta_rad: float) -> float:
    """Delta r = l * tan(theta) in m; exactly 0 for l = 0 or theta = 0."""
    if distance_m == 0.0 or theta_rad == 0.0:
        return 0.0
    delta_r = distance_m * math.tan(theta_rad)
    if not math.isfinite(delta_r):
        raise PlumeRangeError(
            "Radial growth l*tan(theta) overflows 64-bit floating point; "
            "use a smaller distance or half-angle.",
            "delta_r",
        )
    return delta_r


def plume_radius(r1_m: float, distance_m: float, theta_rad: float) -> float:
    """Plume radius r(l) = r1 + l*tan(theta) in m.

    Args:
        r1_m: source radius at x = 0, m (> 0).
        distance_m: axial distance from P1, m (>= 0).
        theta_rad: half-angle, rad (0 <= theta < pi/2).
    """
    _, r1_m, distance_m, theta_rad = validate_domain(1.0, r1_m, distance_m, theta_rad)
    delta_r = _radial_growth(distance_m, theta_rad)
    if delta_r == 0.0:
        return r1_m
    r2 = r1_m + delta_r
    if not math.isfinite(r2):
        raise PlumeRangeError(
            "Downstream radius r2 overflows 64-bit floating point; use a less extreme range.", "r2"
        )
    return r2


def plume_area(radius_m: float) -> float:
    """Full circular cross-section pi*r^2 in m^2 for a radius in m (> 0)."""
    radius_m = _real(radius_m, "Radius", "r1")
    if radius_m <= 0.0:
        raise PlumeModelError("Radius must be greater than zero.", "r1")
    area = math.pi * radius_m * radius_m
    if not math.isfinite(area):
        raise PlumeRangeError(
            f"Area pi*r^2 for r = {radius_m:.6g} m overflows 64-bit floating point; "
            "use a smaller radius.",
            "area",
        )
    if area < _MIN_NORMAL:
        raise PlumeRangeError(
            f"Area pi*r^2 for r = {radius_m:.6g} m underflows 64-bit floating point; "
            "use a larger radius.",
            "area",
        )
    return area


def _named_area(radius_m: float, symbol: str, label: str) -> float:
    try:
        return plume_area(radius_m)
    except PlumeRangeError as exc:
        raise PlumeRangeError(f"{label} {symbol}: {exc}", symbol) from exc


def calculate_plume(n1: float, r1_m: float, distance_m: float, theta_rad: float) -> PlumeResult:
    """Compute the downstream uniform density and all derived outputs.

    Args:
        n1: initial uniform density at P1, particles/m^3 (> 0).
        r1_m: finite source radius at P1, m (> 0).
        distance_m: axial distance l from P1 to P2, m (>= 0).
        theta_rad: plume half-angle measured from the axis, rad (0 <= theta < pi/2).

    Returns:
        PlumeResult with every quantity in SI units.

    Raises:
        PlumeModelError: for inputs outside the domain.
        PlumeRangeError: when a derived quantity is not representable.
    """
    n1, r1_m, distance_m, theta_rad = validate_domain(n1, r1_m, distance_m, theta_rad)

    delta_r = _radial_growth(distance_m, theta_rad)
    r2 = plume_radius(r1_m, distance_m, theta_rad)

    area1 = _named_area(r1_m, "area1", "Source area A1")
    area2 = _named_area(r2, "area2", "Downstream area A2")

    if r2 == r1_m:
        # Identity branch (l = 0, theta = 0, or growth below float resolution).
        ratio = 1.0
    else:
        ratio = r1_m / r2
    remaining = ratio * ratio  # F = (r1/r2)^2, from the radius ratio
    if not (remaining >= _MIN_NORMAL):
        raise PlumeRangeError(
            "Remaining fraction F = (r1/r2)^2 underflows 64-bit floating point; "
            "use a smaller distance or half-angle.",
            "F",
        )
    n2 = n1 * remaining
    if not math.isfinite(n2) or n2 < _MIN_NORMAL:
        raise PlumeRangeError(
            "Downstream density n2 underflows 64-bit floating point; "
            "use a larger n1 or a less extreme dilution.",
            "n2",
        )

    growth = 1.0 if r2 == r1_m else r2 / r1_m
    expansion = growth * growth  # E = A2/A1 = (r2/r1)^2
    if not math.isfinite(expansion):
        raise PlumeRangeError(
            "Area expansion E = A2/A1 overflows 64-bit floating point; use a less extreme range.",
            "E",
        )

    # 1 - F = (r2^2 - r1^2)/r2^2 = (delta_r/r2)*(1 + r1/r2). Algebraically identical to
    # 1 - F but free of cancellation when F is close to 1 (tiny angles or distances).
    reduction = (delta_r / r2) * (1.0 + ratio) if delta_r != 0.0 else 0.0
    if delta_r != 0.0 and reduction == 0.0:
        raise PlumeRangeError(
            "Reduction 1 - F is below 64-bit floating-point resolution; "
            "use a larger distance or half-angle.",
            "reduction",
        )

    return PlumeResult(
        n1=n1,
        r1_m=r1_m,
        distance_m=distance_m,
        theta_rad=theta_rad,
        delta_r_m=delta_r,
        r2_m=r2,
        area1_m2=area1,
        area2_m2=area2,
        n2=n2,
        expansion=expansion,
        remaining_fraction=remaining,
        reduction_fraction=reduction,
        remaining_percent=100.0 * remaining,
        reduction_percent=100.0 * reduction,
    )


def density_at_distance(n1: float, r1_m: float, x_m, theta_rad: float):
    """Uniform-model density n(x) = n1*(r1/(r1 + x*tan(theta)))^2 in particles/m^3.

    Args:
        n1: density at x = 0, particles/m^3 (> 0).
        r1_m: source radius at x = 0, m (> 0).
        x_m: axial position(s) from P1, m; scalar or array, each finite and >= 0.
        theta_rad: half-angle, rad (0 <= theta < pi/2).

    Returns:
        float for scalar x, numpy.ndarray for array x. Evaluated with the same
        operation order as calculate_plume, so n(0) == n1 and n(l) == n2.
    """
    n1, r1_m, _, theta_rad = validate_domain(n1, r1_m, 0.0, theta_rad)
    scalar = np.ndim(x_m) == 0
    try:
        x = np.asarray(x_m, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PlumeModelError("Axial positions must be real numbers.", "x") from exc
    if not np.all(np.isfinite(x)):
        raise PlumeModelError("Axial positions must be finite.", "x")
    if np.any(x < 0.0):
        raise PlumeModelError("Axial positions must be zero or greater.", "x")

    tan_theta = 0.0 if theta_rad == 0.0 else math.tan(theta_rad)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        r = r1_m + x * tan_theta
        if not np.all(np.isfinite(r)):
            raise PlumeRangeError(
                "Plume radius r(x) overflows 64-bit floating point along the profile.", "r(x)"
            )
        ratio = np.where(r == r1_m, 1.0, r1_m / r)
        density = n1 * (ratio * ratio)
    if not np.all(np.isfinite(density)) or np.any(density < _MIN_NORMAL):
        raise PlumeRangeError(
            "Density n(x) underflows 64-bit floating point along the profile; "
            "use a larger n1 or a less extreme dilution.",
            "n(x)",
        )
    return float(density) if scalar else density


def profile_grid(
    r1_m: float, distance_m: float, theta_rad: float, n_log: int = 500, n_lin: int = 100
) -> np.ndarray:
    """Axial sample positions in m from 0 to l inclusive, strictly increasing.

    For theta > 0 the main grid is uniform in u = log(1 + x*tan(theta)/r1), which
    concentrates samples near x = 0 where dilution is fastest when r1/tan(theta)
    is much smaller than l; a uniform grid is merged in to sample the far tail.
    For theta = 0 (or negligible growth) the grid is uniform in x. For l = 0 the
    grid is the single point [0.0]. Endpoints are exactly 0 and l.
    """
    _, r1_m, distance_m, theta_rad = validate_domain(1.0, r1_m, distance_m, theta_rad)
    n_log = max(int(n_log), MIN_PROFILE_SAMPLES)
    if distance_m == 0.0:
        return np.array([0.0])

    r2 = plume_radius(r1_m, distance_m, theta_rad)
    # u_max = log(r2/r1) written as a difference so it cannot overflow.
    u_max = math.log(r2) - math.log(r1_m) if theta_rad > 0.0 else 0.0
    if u_max <= 1e-9:
        x = np.linspace(0.0, distance_m, n_log)
    else:
        tan_theta = math.tan(theta_rad)
        u = np.linspace(0.0, u_max, n_log)
        x_log = (np.expm1(u) * r1_m) / tan_theta
        x = np.concatenate([x_log, np.linspace(0.0, distance_m, max(int(n_lin), 2))])
    x = np.clip(x, 0.0, distance_m)
    x = np.unique(np.concatenate([[0.0], x, [distance_m]]))
    if not np.all(np.isfinite(x)):
        raise PlumeRangeError("Profile grid is not representable; use a less extreme range.", "x grid")
    return x


def plume_profile(result: PlumeResult, n_log: int = 500, n_lin: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_m, density) for a validated result, using its normalized inputs."""
    x = profile_grid(result.r1_m, result.distance_m, result.theta_rad, n_log, n_lin)
    density = density_at_distance(result.n1, result.r1_m, x, result.theta_rad)
    return x, density
