# Ion Plume Geometric Expansion Model

A locally run Streamlit app. It estimates how much the particle number density of an
axisymmetric ion (or electrospray) plume drops through geometric expansion between a finite
source cross-section at P₀ (x = 0) and a downstream plane at P₂ (x = l).

The app has two tabs, and both use the same plume propagation model:

- **Direct Density.** You enter the source density n₀ directly. This is the original (V1)
  workflow.
- **Current-Derived Density.** n₀ is calculated from the beam current, the accelerating
  voltage, the mass of one ion, and the source size. This tab also calculates the ideal current
  collected by a centered circular collector at x = l.

Results are **uniform-plume model estimates**. They are not measured densities, detector count
rates, or calibrated detector currents. The app is intended for quick geometric estimates,
such as plume diagnostics in the ASTRAlab research context, and has no experimental
calibration.

## Model

### Shared propagation (both tabs)

The app assumes steady conservation of the tracked particle flow, with **equal mean axial
velocity** at both planes:

```
N_dot = ∫ n v·x̂ dA  →  n A vₓ          n₀ A₀ vₓ = n₂ A₂ vₓ   (vₓ cancels)

r₀ = √(A₀/π)     Δr = l tan θ     r₂ = r₀ + Δr     A₂ = π r₂²

n₂ = n₀ A₀/A₂ = n₀ ( r₀ / (r₀ + l tan θ) )²

F = A₀/A₂ = n₂/n₀  (remaining fraction)      E = A₂/A₀ = 1/F  (area expansion)

Remaining % = 100 F        Reduction % = 100 (1 − F)

Profile: r(x) = r₀ + x tan θ,   n(x) = n₀ ( r₀ / (r₀ + x tan θ) )²,   0 ≤ x ≤ l
```

At l = 0 or θ = 0, the app gives n₂ = n₀ and F = E = 1, and the reduction is exactly zero.

### Current-derived source density

```
q = e = 1.602176634 × 10⁻¹⁹ C          1 amu = 1.66053906660 × 10⁻²⁷ kg (agreed fixed factor)

½ mᵢ v₀² = q V₀        v₀ = √(2 q V₀ / mᵢ)

I₀ = n₀ q v₀ A₀        n₀ = I₀ / (q v₀ A₀) = [I₀ / (q π r₀²)] √(mᵢ / (2 q V₀))

Check: n₂ = I₀ / (q v₀ A₂). The full beam current stays I₀ at every cross-section.
```

How to read these quantities:

- **Mass.** mᵢ is the mass of **one** modeled ion, entered in amu and converted to kg. It is
  not the total mass of the beam. The app assigns the whole measured current to a single
  representative population with that mass and with charge e. This is an explicit
  approximation: real electrospray plumes contain several species and a broad spread of
  energies.
- **Voltage.** V₀ is the potential difference used to estimate the particle energy qV₀.
- **Velocity.** v₀ is an **idealized characteristic velocity**, not a measured plume-average
  velocity. The energy relation assumes the particles start with negligible kinetic energy
  and stay nonrelativistic.
- **Axial velocity.** The axial velocity is taken to be v₀ and stays constant downstream.
  θ only controls how the footprint grows; velocity is not multiplied by cos θ. Changing I₀
  does not change v₀.

### Centered collector (Current-Derived tab)

```
A_coll = π r_coll²        f_coll = min(1, A_coll/A₂) = 1 if r_coll ≥ r₂ else (r_coll/r₂)²

I_coll = I₀ f_coll        Collected % = 100 f_coll
```

The collector is circular, centered on the axis, and perpendicular to it at x = l. The model
assumes the axial current density is uniform across the full footprint A₂, and that the
collector catches every particle that crosses the overlap.

- **Large collectors.** A collector with r_coll ≥ r₂ covers the whole beam, so f_coll = 1
  exactly and I_coll = I₀. A larger collector collects no extra current.
- **A_coll is not A₂.** A_coll is never substituted into the density dilution equation.
- **f_coll is not F.** They are separate quantities and are equal only when the collector and
  the source have the same area, as in the benchmark below.
- **What I_coll means.** I_coll is an ideal primary-current magnitude. It is not a calibrated
  detector reading, a count rate, or a secondary-ion current.

### Numerical details

- **Density ratio.** F is computed from the radius ratio (r₀/r₂)², never by dividing two
  rounded areas.
- **Reduction.** 1 − F is evaluated as (Δr/r₂)(1 + r₀/r₂). This is algebraically identical,
  and it stays accurate when F is very close to 1. The fraction of current that misses the
  collector, 1 − f_coll, is evaluated the same way.
- **Overflow and underflow.** v₀ and n₀ are evaluated in a mantissa/exponent-scaled form, so
  an intermediate product cannot overflow or underflow when the final value is
  representable. For values in the normal range, the result is bitwise identical to direct
  evaluation.
- **Collector fraction.** The collector fraction compares radii before squaring, so an
  enormous collector cannot overflow the ratio A_coll/A₂.
- **Unrepresentable values.** These are rejected with a message naming the quantity: any
  converted input or derived value that overflows, underflows, or falls below the smallest
  normal float64. The app never shows such a value as an exact result.
- **Far-field limit.** n₂ ≈ n₀r₀²/(l² tan²θ) is shown as explanation only.

### Additions beyond the V2 specification

None of these change the governing equations:

- **v₀ ≥ c is rejected.** If v₀ reaches or exceeds the speed of light, the nonrelativistic
  source model does not apply, so the calculation is rejected with an explanation. The
  specification states that this nonrelativistic assumption holds; the app enforces it
  rather than displaying an unphysical velocity.
- **Notice at v₀ ≥ 1 % of c.** The app shows a notice that the ½mv² relation becomes
  approximate. Like the θ ≥ 85° note, this is a notice, not a validity boundary.

### Assumptions

- Emission is steady, and tracked particles do not accumulate between the planes.
- The plume is axisymmetric: circular cross-sections, a constant half-angle, and straight
  frustum boundaries.
- Number density and axial current density are uniform across each cross-section.
- The tracked population is conserved. There are no sources, losses, neutralization,
  fragmentation, deposition, or escape across the boundary.
- The mean *axial* velocity is the same at both planes. Acceleration and changes in the
  velocity-direction distribution are excluded.
- P₀ is a finite reference cross-section, not a point source. The source radius is entered
  by the user and is never replaced by an emitter or extractor dimension.

### Limitations and excluded physics

The model leaves out:

- changes in velocity along the plume
- electric-field forces
- Coulomb interactions and space charge
- collisions
- fragmentation or neutralization
- evolution of velocity directions
- nonuniform (for example Gaussian) radial profiles
- multi-species mixtures
- particle creation or loss
- detector efficiency, secondary emission, count rates, and instrument response

None of these exclusions justify assuming zero loss or a uniform density for experimental data
without validation.

## Setup

Tested with **Python 3.14.6** on Windows 11, using the pinned versions in `requirements.txt`
(Streamlit 1.64.0, NumPy 2.5.3, Matplotlib 3.11.2) and pytest 9.1.1.

Create and activate a virtual environment in the project folder:

```
python -m venv .venv
```

| Platform | Activate |
|---|---|
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (cmd) | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

Install the dependencies and run the app:

```
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

Streamlit prints a local URL (normally http://127.0.0.1:8501). The server binds to localhost
only. `.streamlit/config.toml` also sets this and turns off Streamlit's usage statistics.

After the dependencies are installed, the app needs no account, API key, or network
connection.

Developer validation:

```
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Using the app

### Inputs common to both tabs

- **Axial distance l.** Choose m, cm, mm, or µm. Distance is measured **along the plume
  axis** from P₀ to P₂; a slant distance along the cone boundary is not supported. l = 0 is
  allowed.
- **Half-angle θ.** Enter it in degrees, with 0 ≤ θ < 90. θ is measured between the plume
  boundary and a line parallel to the axis. **If you have a full opening angle, divide it by
  two.**
- **Source geometry.** Choose **Radius r₀** (the default; m, cm, mm, µm) or **Area A₀** (m²,
  cm², mm², µm²; the app uses r₀ = √(A₀/π)). The field **starts empty**, and the app never
  assumes a source size. Switching between radius and area converts the value you entered.
  The exact radius is kept, so switching back and forth repeatedly does not accumulate
  rounding.
- **Scientific notation.** Every numeric field accepts it, for example `1e20` or `2.5e-3`.

### Direct Density tab

Enter n₀ (particles/m³), l, θ, and the source size. **Load example** fills in and calculates
the original benchmark.

### Current-Derived Density tab

| Input | Units | Rule |
|---|---|---|
| Beam current magnitude \|I₀\| | A, mA, µA, nA | Nonzero. A negative entry is read as its magnitude and does not set polarity. |
| Accelerating voltage magnitude \|V₀\| | V, kV | Nonzero. A negative entry is read as its magnitude and does not set polarity. |
| Individual ion mass mᵢ | amu | Positive (negative values are rejected, not flipped). |
| Source radius r₀ or area A₀ | length or area units | Required; starts empty. |
| Collector radius r_coll or area A_coll | length or area units | Required; starts empty. It has its own radius/area toggle, separate from the source toggle. |
| Charge q | none | Fixed at one elementary charge (shown in the calculation details). |

The results show:

- n₂ first, as in the Direct tab.
- The **calculated** n₀, labelled as derived rather than entered.
- |I₀|, |V₀|, mᵢ, and v₀.
- Every geometric output from the Direct tab.
- r_coll, A_coll, f_coll, the collected percentage, and I_coll.

"Calculation details" lists the mass in amu and kg, the fixed charge and amu factor, qV₀,
the reconstruction n₀qv₀A₀ = I₀, the check n₂ = I₀/(qv₀A₂), and the full beam current at x = l.

### Behavior in both tabs

- **Calculate** validates every field in its own tab together and shows each error under its
  field. It stays disabled until the required geometry is filled in: the source in Direct
  Density, and both the source and the collector in Current-Derived Density.
- **Load example** fills in values that are labelled **illustrative, not a measured lab
  configuration**.
- **Reset** affects only its own tab. It clears that tab's results, plots, errors, example
  state, and geometry fields (source, plus the collector in the Current-Derived tab). Other
  inputs are kept.
- **Separate state.** Each tab keeps its own inputs, results, and errors. A blank or invalid
  field in one tab never blocks the other.
- **Out-of-date results.** After any change to a physical input, the results and plots are
  marked **OUT OF DATE** until a successful recalculation. A failed calculation never relabels
  an old result as current. These changes do not invalidate a result: an equivalent unit
  (1000 nA instead of 1 µA), switching between radius and area, a sign change on current or
  voltage, or changing the plot scale or diagram view. Changing a unit selector while leaving
  the number unchanged *does* invalidate the result.

### Interpreting the results

n₂ is the uniform density the model predicts at the downstream plane. In this model the value
at P₂ equals the cross-sectional average. A real plume is not uniform, so n₂ is not the local
density at an instrument. The assumed source radius sets the normalization of the prediction.

The main display shows three significant figures, and "Calculation details" shows ten.
Percentages always keep enough digits to tell a small nonzero value from 0 %, and a value just
below 100 % from exact saturation.

## Benchmarks

Both benchmarks are reproduced by the test suite to a relative tolerance of 10⁻¹⁰.

**Direct Density** (n₀ = 1.0 × 10²⁰ particles/m³, r₀ = 1.0 mm, l = 0.10 m, θ = 10°):

| Quantity | Value |
|---|---|
| r₂ | 0.0186326980708465 m |
| A₀ | 3.141592653589793 × 10⁻⁶ m² |
| A₂ | 1.090690086825855 × 10⁻³ m² |
| n₂ | 2.880371511152640 × 10¹⁷ particles/m³ |
| E | 347.1774373993268 |
| F | 0.002880371511152640 |
| Remaining | 0.2880371511152640 % |
| Reduction | 99.71196284888474 % |

**Current-Derived** (I₀ = 1.0 µA, V₀ = 1.0 kV, mᵢ = 100 amu, r₀ = 1.0 mm, l = 0.10 m, θ = 10°,
r_coll = 1.0 mm). These are software verification inputs, not an experimental configuration.

| Quantity | Value |
|---|---|
| mᵢ | 1.66053906660 × 10⁻²⁵ kg |
| v₀ | 43928.42636759329 m/s |
| n₀ | 4.522661536932149 × 10¹³ particles/m³ |
| r₂, A₂, F, E | as in the Direct benchmark |
| n₂ | 1.302694544556517 × 10¹¹ particles/m³ |
| A_coll | 3.141592653589793 × 10⁻⁶ m² |
| f_coll | 0.002880371511152640 (equal to F only because r_coll = r₀) |
| I_coll | 2.880371511152640 × 10⁻⁹ A = 2.88 nA |

**Collector check** (θ = 0, r₀ = 2 mm, I₀ = 1 µA): r_coll = 1 mm gives f_coll = 0.25 and
I_coll = 0.25 µA. r_coll = 2 mm or 3 mm gives f_coll = 1 exactly and I_coll = 1 µA.

## Units

| Quantity | Units | Conversion to SI |
|---|---|---|
| Length (l, r₀, r_coll) | m, cm, mm, µm | × 1, 10⁻², 10⁻³, 10⁻⁶ |
| Area (A₀, A_coll) | m², cm², mm², µm² | × 1, 10⁻⁴, 10⁻⁶, 10⁻¹² |
| Current I₀ | A, mA, µA, nA | × 1, 10⁻³, 10⁻⁶, 10⁻⁹ |
| Voltage V₀ | V, kV | × 1, 10³ |
| Ion mass mᵢ | amu | × 1.66053906660 × 10⁻²⁷ kg |
| Half-angle θ | degrees (UI) | θ_rad = θ_deg × π/180 |
| Density n | particles/m³ | none |

`u` and `μ` (Greek mu) are accepted as aliases for the micro sign in unit names, for example
`um` and `uA`.

## Project structure

```
app.py               Streamlit tabs, controls, validation messages, per-tab state, results display
plume_model.py       Shared propagation: domain/range checks, PlumeResult, profile, sampling grid
source_model.py      Current-derived source normalization: v₀ and n₀ (SourceResult)
collection.py        Centered circular collector: f_coll, A_coll, I_coll (CollectionResult)
current_derived.py   Composes source -> propagation -> collection (CurrentDerivedResult)
constants.py         Elementary charge, amu factor, speed of light, modeled charge
units.py             Unit conversion, number parsing, input validation for both tabs (no UI)
formatting.py        Significant-figure, engineering-unit and percentage display strings
visualization.py     Matplotlib geometry diagram (with optional collector) and density profile
.streamlit/config.toml   Localhost binding; usage statistics disabled
requirements.txt     Pinned runtime dependencies
requirements-dev.txt Runtime dependencies plus pytest
pytest.ini           Test configuration
tests/               Physics, source, collection, units, profile, formatting, figure and UI checks
```

These boundaries are designed to support future extensions:

- A charge state z only changes `charge_c` (q = z·e) in `source_model`.
- Offsets, tilt, or radial profiles would replace `collection_fraction`.
- Velocity changing along the plume would change propagation.

None of these extensions is implemented yet.

### Python API

The physics modules are usable without the UI. All values are SI, and angles are in radians.

```python
from current_derived import calculate_current_derived
from plume_model import calculate_plume
from units import amu_to_kg, deg_to_rad, length_to_m

direct = calculate_plume(n0=1e20, r0_m=length_to_m(1.0, "mm"), distance_m=0.10,
                         theta_rad=deg_to_rad(10.0))
direct.n2  # 2.88037151115264e+17 particles/m^3

result = calculate_current_derived(current_a=1e-6, voltage_v=1000.0, mass_kg=amu_to_kg(100.0),
                                   r0_m=1e-3, distance_m=0.10, theta_rad=deg_to_rad(10.0),
                                   collector_radius_m=1e-3)
result.source.n0, result.plume.n2, result.collection.collected_current_a
```

Invalid inputs raise `PlumeModelError`, and its `field` attribute names the input. Values that
cannot be represented raise `PlumeRangeError`, and its `quantity` attribute names the quantity.

**V1 compatibility.** The source quantities were renamed n1 → n0, r1_m → r0_m, area1_m2 →
area0_m2 (P₁ → P₀). For older code, `calculate_plume`, `density_at_distance`, `plume_radius`,
`profile_grid`, and `validate_domain` still accept the keyword names `n1=` and `r1_m=`.
`PlumeResult` still has read-only `n1`, `r1_m`, and `area1_m2` aliases, and
`units.validate_inputs` remains an alias of `validate_direct_inputs`. These aliases call the
same implementation; there is no second geometry formula.
