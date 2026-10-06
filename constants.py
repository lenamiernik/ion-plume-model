"""Centralized physical constants and fixed model parameters (SI units).

These values are used by every calculation and regression benchmark; change them
only deliberately, never between calculations.
"""

# Elementary charge, C (exact SI value).
ELEMENTARY_CHARGE_C = 1.602176634e-19

# Agreed fixed amu -> kg conversion for this implementation (V2 specification,
# section 3.1). This is not an exact SI-defined constant.
AMU_KG = 1.66053906660e-27

# Speed of light in vacuum, m/s (exact SI value). Used only to flag speeds where the
# nonrelativistic energy relation is a poor approximation.
SPEED_OF_LIGHT_M_S = 299_792_458.0

# Modeled charge magnitude in the V2 UI: one elementary charge. Future charge-state
# support (q = z*e) passes a different charge_c to the source model; the plume
# propagation law does not depend on charge.
MODELED_CHARGE_C = ELEMENTARY_CHARGE_C
