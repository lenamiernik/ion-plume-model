"""Pure SI physics for the ion plume geometric expansion model (shared by both modes).

The model conserves the particle flow of one tracked population through an
axisymmetric conical frustum with equal mean axial velocity at both planes:

    n0 * A0 * v_x = n2 * A2 * v_x   =>   n2 = n0 * (r0 / (r0 + l * tan(theta)))**2

Conventions (all SI, 64-bit floats):
    n0, n2      uniform number densities at P0 (x = 0) and P2 (x = l), particles/m^3
    r0_m, r2_m  circular plume radii, m
    area0_m2,
    area2_m2    full circular plume cross-sections, m^2 (A2 is not a detector aperture)
    distance_m  axial separation l between P0 and P2, m
    theta_rad   plume half-angle from the axis, radians; 0 <= theta < pi/2

How n0 is obtained (entered directly or derived from beam current in
source_model.py) does not change this propagation law. This module has no UI
dependencies so it can be used from tests or batch scripts.

V1 compatibility: the source quantities were renamed n1 -> n0, r1_m -> r0_m and
area1_m2 -> area0_m2 (P1 -> P0). The legacy keyword names n1= and r1_m= are still
accepted by the public functions, and PlumeResult keeps read-only n1, r1_m and
area1_m2 aliases. Both routes call the same implementation.
"""

from __future__ import annotations

import functools
import math
import sys
from dataclasses import dataclass
from numbers import Real

import numpy as np

HIGH_ANGLE_SENSITIVITY_DEG = 85.0
HIGH_ANGLE_SENSITIVITY_RAD = HIGH_ANGLE_SENSITIVITY_DEG * math.pi / 180.0
MIN_PROFILE_SAMPLES = 400

# Smallest positive float64 that keeps full precision. Positive results below this
# are subnormal (reduced precision) or zero and are treated as unrepresentable.
MIN_NORMAL = sys.float_info.min


class PlumeModelError(ValueError):
    """Raised for inputs outside the supported physical domain.

    Attributes:
        field: input name the error refers to (for example "n0", "r0", "distance",
            "theta", "x", "current", "voltage", "mass", "charge", "collector",
            "velocity"), or None when it is not tied to one input.
    """

    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.field = field


class PlumeRangeError(PlumeModelError):
    """Raised when a valid input or derived quantity cannot be represented.

    Attributes:
        quantity: symbol of the quantity that overflowed, underflowed or rounded to
            zero (for example "r2", "area2", "n2", "n0", "v0", "area_coll", "I_coll").
    """

    def __init__(self, message: str, quantity: str, field: str | None = None):
        super().__init__(message, field)
        self.quantity = quantity


def as_real(value, name: str, field: str) -> float:
    """Return value as a finite float; reject booleans, non-reals, NaN and infinity."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise PlumeModelError(f"{name} must be a real number.", field)
    v = float(value)
    if not math.isfinite(v):
        raise PlumeModelError(f"{name} must be a finite number.", field)
    return v


def require_positive(value, name: str, field: str) -> float:
    """Return value as a finite float > 0, else raise PlumeModelError."""
    v = as_real(value, name, field)
    if v <= 0.0:
        raise PlumeModelError(f"{name} must be greater than zero.", field)
    return v


def require_normal(value: float, quantity: str, description: str) -> float:
    """Check that a positive derived value is finite and at least MIN_NORMAL."""
    if not math.isfinite(value):
        raise PlumeRangeError(
            f"{description} overflows 64-bit floating point; use a less extreme range.", quantity
        )
    if value < MIN_NORMAL:
        raise PlumeRangeError(
            f"{description} underflows 64-bit floating point; use a less extreme range.", quantity
        )
    return value


def scaled_product(numerators, denominators=()) -> tuple[float, int]:
    """Return (mantissa, exponent) with prod(numerators)/prod(denominators) = mantissa * 2**exponent.

    The factors must be positive and finite. Each step works on a mantissa in
    [0.5, 1), so intermediate products cannot overflow or underflow even when the
    final value is representable. Scaling by powers of two is exact, so in the
    normal range the result is bitwise identical to direct evaluation in the same order.
    """
    mantissa, exponent = 1.0, 0
    for factor in numerators:
        m, e = math.frexp(factor)
        mantissa, e2 = math.frexp(mantissa * m)
        exponent += e + e2
    for factor in denominators:
        m, e = math.frexp(factor)
        mantissa, e2 = math.frexp(mantissa / m)
        exponent += e2 - e
    return mantissa, exponent


def from_scaled(mantissa: float, exponent: int) -> float:
    """mantissa * 2**exponent, returning inf on overflow (0 or subnormal on underflow)."""
    try:
        return math.ldexp(mantissa, exponent)
    except OverflowError:
        return math.inf


def _v1_keyword_aliases(func):
    """Accept the V1 keyword names n1= and r1_m= for n0= and r0_m=."""
    legacy = {"n1": "n0", "r1_m": "r0_m"}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for old, new in legacy.items():
            if old in kwargs:
                if new in kwargs:
                    raise TypeError(f"Pass either {new} or the legacy name {old}, not both.")
                kwargs[new] = kwargs.pop(old)
        return func(*args, **kwargs)

    return wrapper


@dataclass(frozen=True)
class PlumeResult:
    """Normalized SI inputs and all outputs of one validated propagation calculation.

    Inputs: n0 (particles/m^3), r0_m (m), distance_m (m), theta_rad (rad).
    Outputs: r2_m and delta_r_m (m), area0_m2 and area2_m2 (m^2), n2
    (particles/m^3), expansion E = A2/A0, remaining_fraction F = A0/A2 = n2/n0,
    reduction_fraction 1 - F, and the corresponding percentages.
    """

    n0: float
    r0_m: float
    distance_m: float
    theta_rad: float
    delta_r_m: float
    r2_m: float
    area0_m2: float
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

    # V1 compatibility aliases (read-only).
    @property
    def n1(self) -> float:
        """Deprecated V1 name for n0."""
        return self.n0

    @property
    def r1_m(self) -> float:
        """Deprecated V1 name for r0_m."""
        return self.r0_m

    @property
    def area1_m2(self) -> float:
        """Deprecated V1 name for area0_m2."""
        return self.area0_m2


@_v1_keyword_aliases
def validate_domain(n0, r0_m, distance_m, theta_rad) -> tuple[float, float, float, float]:
    """Check the supported propagation domain and return the inputs as floats.

    Args:
        n0: source density at P0, particles/m^3; finite and > 0.
        r0_m: source radius, m; finite and > 0.
        distance_m: axial distance, m; finite and >= 0.
        theta_rad: half-angle, rad; finite and 0 <= theta < pi/2.

    Raises:
        PlumeModelError: if any value is non-real, NaN, infinite or out of range.
    """
    n0 = as_real(n0, "Source density n0", "n0")
    r0_m = as_real(r0_m, "Source radius r0", "r0")
    distance_m = as_real(distance_m, "Axial distance l", "distance")
    theta_rad = as_real(theta_rad, "Half-angle theta", "theta")
    if n0 <= 0.0:
        raise PlumeModelError("Source density n0 must be greater than zero.", "n0")
    if r0_m <= 0.0:
        raise PlumeModelError("Source radius r0 must be greater than zero.", "r0")
    if distance_m < 0.0:
        raise PlumeModelError("Axial distance must be zero or greater.", "distance")
    if not (0.0 <= theta_rad < math.pi / 2.0):
        raise PlumeModelError(
            "Half-angle must be at least 0 deg and less than 90 deg (0 <= theta < pi/2 rad).", "theta"
        )
    return n0, r0_m, distance_m, theta_rad


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


@_v1_keyword_aliases
def plume_radius(r0_m: float, distance_m: float, theta_rad: float) -> float:
    """Plume radius r(l) = r0 + l*tan(theta) in m.

    Args:
        r0_m: source radius at x = 0, m (> 0).
        distance_m: axial distance from P0, m (>= 0).
        theta_rad: half-angle, rad (0 <= theta < pi/2).
    """
    _, r0_m, distance_m, theta_rad = validate_domain(1.0, r0_m, distance_m, theta_rad)
    delta_r = _radial_growth(distance_m, theta_rad)
    if delta_r == 0.0:
        return r0_m
    r2 = r0_m + delta_r
    if not math.isfinite(r2):
        raise PlumeRangeError(
            "Downstream radius r2 overflows 64-bit floating point; use a less extreme range.", "r2"
        )
    return r2


def plume_area(radius_m: float) -> float:
    """Full circular cross-section pi*r^2 in m^2 for a radius in m (> 0)."""
    radius_m = as_real(radius_m, "Radius", "r0")
    if radius_m <= 0.0:
        raise PlumeModelError("Radius must be greater than zero.", "r0")
    area = math.pi * radius_m * radius_m
    if not math.isfinite(area):
        raise PlumeRangeError(
            f"Area pi*r^2 for r = {radius_m:.6g} m overflows 64-bit floating point; "
            "use a smaller radius.",
            "area",
        )
    if area < MIN_NORMAL:
        raise PlumeRangeError(
            f"Area pi*r^2 for r = {radius_m:.6g} m underflows 64-bit floating point; "
            "use a larger radius.",
            "area",
        )
    return area


def named_area(radius_m: float, symbol: str, label: str) -> float:
    """plume_area with range errors renamed to the given quantity symbol."""
    try:
        return plume_area(radius_m)
    except PlumeRangeError as exc:
        raise PlumeRangeError(f"{label} {symbol}: {exc}", symbol) from exc


@_v1_keyword_aliases
def calculate_plume(n0: float, r0_m: float, distance_m: float, theta_rad: float) -> PlumeResult:
    """Compute the downstream uniform density and all derived geometric outputs.

    Args:
        n0: uniform source density at P0, particles/m^3 (> 0).
        r0_m: finite source radius at P0, m (> 0).
        distance_m: axial distance l from P0 to P2, m (>= 0).
        theta_rad: plume half-angle measured from the axis, rad (0 <= theta < pi/2).

    Returns:
        PlumeResult with every quantity in SI units.

    Raises:
        PlumeModelError: for inputs outside the domain.
        PlumeRangeError: when a derived quantity is not representable.
    """
    n0, r0_m, distance_m, theta_rad = validate_domain(n0, r0_m, distance_m, theta_rad)

    delta_r = _radial_growth(distance_m, theta_rad)
    r2 = plume_radius(r0_m, distance_m, theta_rad)

    area0 = named_area(r0_m, "area0", "Source area A0")
    area2 = named_area(r2, "area2", "Downstream area A2")

    if r2 == r0_m:
        # Identity branch (l = 0, theta = 0, or growth below float resolution).
        ratio = 1.0
    else:
        ratio = r0_m / r2
    remaining = ratio * ratio  # F = (r0/r2)^2, from the radius ratio
    if not (remaining >= MIN_NORMAL):
        raise PlumeRangeError(
            "Remaining fraction F = (r0/r2)^2 underflows 64-bit floating point; "
            "use a smaller distance or half-angle.",
            "F",
        )
    n2 = n0 * remaining
    if not math.isfinite(n2) or n2 < MIN_NORMAL:
        raise PlumeRangeError(
            "Downstream density n2 underflows 64-bit floating point; "
            "use a larger n0 or a less extreme dilution.",
            "n2",
        )

    growth = 1.0 if r2 == r0_m else r2 / r0_m
    expansion = growth * growth  # E = A2/A0 = (r2/r0)^2
    if not math.isfinite(expansion):
        raise PlumeRangeError(
            "Area expansion E = A2/A0 overflows 64-bit floating point; use a less extreme range.",
            "E",
        )

    # 1 - F = (r2^2 - r0^2)/r2^2 = (delta_r/r2)*(1 + r0/r2). Algebraically identical to
    # 1 - F but free of cancellation when F is close to 1 (tiny angles or distances).
    reduction = (delta_r / r2) * (1.0 + ratio) if delta_r != 0.0 else 0.0
    if delta_r != 0.0 and reduction == 0.0:
        raise PlumeRangeError(
            "Reduction 1 - F is below 64-bit floating-point resolution; "
            "use a larger distance or half-angle.",
            "reduction",
        )

    return PlumeResult(
        n0=n0,
        r0_m=r0_m,
        distance_m=distance_m,
        theta_rad=theta_rad,
        delta_r_m=delta_r,
        r2_m=r2,
        area0_m2=area0,
        area2_m2=area2,
        n2=n2,
        expansion=expansion,
        remaining_fraction=remaining,
        reduction_fraction=reduction,
        remaining_percent=100.0 * remaining,
        reduction_percent=100.0 * reduction,
    )


@_v1_keyword_aliases
def density_at_distance(n0: float, r0_m: float, x_m, theta_rad: float):
    """Uniform-model density n(x) = n0*(r0/(r0 + x*tan(theta)))^2 in particles/m^3.

    Args:
        n0: density at x = 0, particles/m^3 (> 0).
        r0_m: source radius at x = 0, m (> 0).
        x_m: axial position(s) from P0, m; scalar or array, each finite and >= 0.
        theta_rad: half-angle, rad (0 <= theta < pi/2).

    Returns:
        float for scalar x, numpy.ndarray for array x. Evaluated with the same
        operation order as calculate_plume, so n(0) == n0 and n(l) == n2.
    """
    n0, r0_m, _, theta_rad = validate_domain(n0, r0_m, 0.0, theta_rad)
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
        r = r0_m + x * tan_theta
        if not np.all(np.isfinite(r)):
            raise PlumeRangeError(
                "Plume radius r(x) overflows 64-bit floating point along the profile.", "r(x)"
            )
        ratio = np.where(r == r0_m, 1.0, r0_m / r)
        density = n0 * (ratio * ratio)
    if not np.all(np.isfinite(density)) or np.any(density < MIN_NORMAL):
        raise PlumeRangeError(
            "Density n(x) underflows 64-bit floating point along the profile; "
            "use a larger n0 or a less extreme dilution.",
            "n(x)",
        )
    return float(density) if scalar else density


@_v1_keyword_aliases
def profile_grid(
    r0_m: float, distance_m: float, theta_rad: float, n_log: int = 500, n_lin: int = 100
) -> np.ndarray:
    """Axial sample positions in m from 0 to l inclusive, strictly increasing.

    For theta > 0 the main grid is uniform in u = log(1 + x*tan(theta)/r0), which
    concentrates samples near x = 0 where dilution is fastest when r0/tan(theta)
    is much smaller than l; a uniform grid is merged in to sample the far tail.
    For theta = 0 (or negligible growth) the grid is uniform in x. For l = 0 the
    grid is the single point [0.0]. Endpoints are exactly 0 and l.
    """
    _, r0_m, distance_m, theta_rad = validate_domain(1.0, r0_m, distance_m, theta_rad)
    n_log = max(int(n_log), MIN_PROFILE_SAMPLES)
    if distance_m == 0.0:
        return np.array([0.0])

    r2 = plume_radius(r0_m, distance_m, theta_rad)
    # u_max = log(r2/r0) written as a difference so it cannot overflow.
    u_max = math.log(r2) - math.log(r0_m) if theta_rad > 0.0 else 0.0
    if u_max <= 1e-9:
        x = np.linspace(0.0, distance_m, n_log)
    else:
        tan_theta = math.tan(theta_rad)
        u = np.linspace(0.0, u_max, n_log)
        x_log = (np.expm1(u) * r0_m) / tan_theta
        x = np.concatenate([x_log, np.linspace(0.0, distance_m, max(int(n_lin), 2))])
    x = np.clip(x, 0.0, distance_m)
    x = np.unique(np.concatenate([[0.0], x, [distance_m]]))
    if not np.all(np.isfinite(x)):
        raise PlumeRangeError("Profile grid is not representable; use a less extreme range.", "x grid")
    return x


def plume_profile(result: PlumeResult, n_log: int = 500, n_lin: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_m, density) for a validated result, using its normalized inputs."""
    x = profile_grid(result.r0_m, result.distance_m, result.theta_rad, n_log, n_lin)
    density = density_at_distance(result.n0, result.r0_m, x, result.theta_rad)
    return x, density
