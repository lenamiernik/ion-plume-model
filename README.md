# Ion Plume Geometric Expansion Model

A locally run Streamlit app. It estimates how much the particle number density of an
axisymmetric ion (or electrospray) plume drops through geometric expansion between a finite
source cross-section at P₁ (x = 0) and a downstream instrument plane at P₂ (x = l).

The result is a **uniform cross-sectional model estimate**. It is not a measured density or a
detector count rate. The app is intended for quick geometric estimates, such as plume
diagnostics in the ASTRAlab research context, and has no experimental calibration.

## Model

The app assumes steady conservation of the tracked particle flow, with **equal mean axial
velocity** at both planes:

```
N_dot = ∫ n v·x̂ dA  →  n A vₓ          n₁ A₁ vₓ,₁ = n₂ A₂ vₓ,₂,   vₓ,₁ = vₓ,₂ > 0

r₁ = √(A₁/π)     r₂ = r₁ + l tan θ     A₂ = π r₂²

n₂ = n₁ A₁/A₂ = n₁ ( r₁ / (r₁ + l tan θ) )²

F = A₁/A₂ = n₂/n₁  (remaining fraction)      E = A₂/A₁ = 1/F  (area expansion)

Remaining % = 100 F        Reduction % = 100 (1 − F)

Profile: r(x) = r₁ + x tan θ,   n(x) = n₁ ( r₁ / (r₁ + x tan θ) )²
```

Implementation details:

- The density ratio is computed from the radius ratio (r₁/r₂)², never by dividing two
  rounded areas.
- 1 − F is evaluated as (Δr/r₂)(1 + r₁/r₂). This is algebraically identical to 1 − F and
  stays accurate when F is very close to 1.
- The far-field limit n₂ ≈ n₁r₁²/(l² tan²θ), valid only for θ > 0 and l tan θ ≫ r₁, is given
  in the app as explanation only. Every calculation uses the full frustum equation.

### Assumptions

- Emission is steady, and tracked particles do not accumulate between the planes.
- The plume is axisymmetric: circular cross-sections, a constant half-angle, and straight
  frustum boundaries.
- Number density is uniform across each cross-section.
- The tracked population is conserved. There are no sources, losses, neutralization,
  deposition, or escape across the boundary.
- The mean *axial* velocity is equal at both planes. Acceleration and changes in the
  velocity-direction distribution are excluded.
- P₁ is a finite reference cross-section, not a point source.

### Limitations and excluded physics

The model leaves out:

- electric-field forces
- Coulomb interactions and space charge
- collisions and acceleration
- the evolution of velocity directions
- nonuniform radial profiles
- particle creation or loss
- instrument response

A₂ is the full plume footprint, **not** the instrument aperture. Predicting an instrument
signal also requires:

- the instrument's sampling geometry, acceptance, and efficiency
- particle properties
- a radial density or flux distribution

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

1. **Initial density n₁.** Enter it in particles/m³. Scientific notation such as `1e20` is
   accepted.
2. **Axial distance l.** Choose m, cm, mm, or µm. Distance is measured **along the plume
   axis** from P₁ to P₂; a slant distance along the cone boundary is not supported.
   l = 0 is allowed and gives n₂ = n₁.
3. **Half-angle θ.** Enter it in degrees, with 0 ≤ θ < 90. θ is measured between the plume
   boundary and a line parallel to the axis. **If you have a full opening angle, divide it by
   two.** For θ ≥ 85° the app adds a sensitivity note. This is not a physical validity limit.
4. **Source geometry.** Choose **Radius r₁** (the default; m, cm, mm, µm) or **Area A₁**
   (m², cm², mm², µm²; the app uses r₁ = √(A₁/π)). The field **starts empty** and Calculate
   stays unavailable until you fill it in. n₁, l, and θ alone do not determine n₂, and the
   app never assumes a source size. Switching modes converts the value you entered, so the
   physical size stays the same. There is only ever one source value.
5. **Calculate.** All fields are validated together, and each error appears under its
   field. If a value cannot be represented in 64-bit floating point (overflow, underflow, or
   rounding to zero during conversion), the app names the quantity instead of showing 0,
   NaN, or ∞.
6. **Load example.** This fills in and calculates the PRD benchmark: n₁ = 1.0 × 10²⁰
   particles/m³, r₁ = 1.0 mm, l = 0.10 m, θ = 10°. These values are **illustrative, not a
   measured lab configuration**, and the app labels them that way.
7. **Reset.** Clears the results, errors, and source geometry.

If you edit an input after calculating, the results and plots are marked **OUT OF DATE**
until you press Calculate again. Changing only a unit without changing the physical value,
or changing the plot axis scale, does not invalidate the result.

### Interpreting the result

n₂ is the uniform density the model predicts at the downstream plane. In this model the value
at the centerline point P₂ equals the cross-sectional average. A real plume is not uniform, so
n₂ is not the local density at an instrument, and it is not a detector signal. The assumed
source radius sets the normalization of the prediction.

The main display shows three significant figures, and "Calculation details" shows ten.
Percentages always keep enough digits to tell them apart from 0 % and 100 %.

Benchmark results:

| Quantity | Value |
|---|---|
| r₂ | 0.0186326980708465 m |
| A₁ | 3.141592653589793 × 10⁻⁶ m² |
| A₂ | 1.090690086825855 × 10⁻³ m² |
| n₂ | 2.880371511152640 × 10¹⁷ particles/m³ |
| E | 347.1774373993268 |
| F | 0.002880371511152640 |
| Remaining | 0.2880371511152640 % |
| Reduction | 99.71196284888474 % |

## Units

| Quantity | Units | Conversion to SI |
|---|---|---|
| Length (l, r₁) | m, cm, mm, µm | × 1, 10⁻², 10⁻³, 10⁻⁶ |
| Area (A₁) | m², cm², mm², µm² | × 1, 10⁻⁴, 10⁻⁶, 10⁻¹² |
| Half-angle θ | degrees (UI) | θ_rad = θ_deg × π/180 |
| Density n | particles/m³ | none |

## Project structure

```
app.py               Streamlit layout, controls, validation messages, state, results display
plume_model.py       Pure SI physics: domain/range checks, PlumeResult, profile and sampling grid
units.py             Length/area/degree conversion, number parsing, input validation (no UI)
formatting.py        Significant-figure, engineering-unit and percentage display strings
visualization.py     Matplotlib geometry diagram and density-profile figures
.streamlit/config.toml   Localhost binding; usage statistics disabled
requirements.txt     Pinned runtime dependencies
requirements-dev.txt Runtime dependencies plus pytest
pytest.ini           Test configuration
tests/               Physics, units, profile, formatting, figure and UI (AppTest) checks
```

The physics API in `plume_model.py` is usable without the UI. All units are SI and angles are
in radians:

```python
from plume_model import calculate_plume, density_at_distance, plume_radius, plume_area
from units import deg_to_rad, length_to_m

result = calculate_plume(n1=1e20, r1_m=length_to_m(1.0, "mm"), distance_m=0.10,
                         theta_rad=deg_to_rad(10.0))
result.n2  # 2.88037151115264e+17 particles/m^3
```

Invalid inputs raise `PlumeModelError`. Values that cannot be represented raise
`PlumeRangeError`, and its `quantity` attribute names the quantity.
