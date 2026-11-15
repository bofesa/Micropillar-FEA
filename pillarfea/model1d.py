"""1D progressive-yield model of a tapered pillar (small strain, uniaxial stress per slice).

sigma(s) = F / A(s), A(s) = pi (R_top + s tan(theta))^2, s = distance from the top;
local strain from the material's uniaxial small-strain law; displacement = integral of strain.
Rigid base, no friction, no lateral stress interaction. Used as an independent check of the
FE trends, not as a reference solution.
"""
from __future__ import annotations

import numpy as np

from .geometry import Geometry
from .materials import Material


def tapered_curve(geo: Geometry, mat: Material, eps_max: float, n: int = 1500,
                  nz: int = 4001):
    s = np.linspace(0.0, geo.h, nz)
    ratio = (geo.R_top / geo.outer_radius(geo.h - s)) ** 2      # A_top / A(s), incl. fillet
    S_hi = float(mat.uniaxial_small_strain(eps_max)) / ratio.min() * 1.05
    S = np.linspace(0.0, S_hi, n)
    sig = S[:, None] * ratio[None, :]
    e = sig / mat.E + mat.plastic_strain(sig)
    eps = np.trapezoid(e, s, axis=1) / geo.h
    keep = eps <= eps_max
    return eps[keep], S[keep]
