"""Yield-stress extraction from nominal stress-strain curves.

All functions take monotonic-loading curves (eps, sig) starting at the origin.
Methods
  offset   : intersection of the curve with a line of slope E_s shifted by `offset_strain`
  bilinear : least-squares fit of a continuous two-segment line through the origin
             (elastic + linear hardening); yield = stress at the break point
The slope E_s is either fitted over a stress window (`slope: fit`) or the input E
(`slope: material`).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def elastic_fit(eps, sig, lo: float, hi: float):
    """Linear fit sig = E*eps + c over lo <= sig <= hi on the rising part of the curve."""
    eps, sig = np.asarray(eps), np.asarray(sig)
    kmax = int(np.argmax(sig))
    m = np.zeros(len(sig), bool)
    m[:kmax + 1] = (sig[:kmax + 1] >= lo) & (sig[:kmax + 1] <= hi)
    if m.sum() < 3:
        raise ValueError("Too few points in the elastic-fit window; increase n_increments "
                         "or widen slope_window")
    E, c = np.polyfit(eps[m], sig[m], 1)
    return float(E), float(c)


def offset_intersection(eps, sig, E, c, offset):
    """First crossing of the curve with sig = E*(eps - offset) + c (linear interpolation)."""
    eps, sig = np.asarray(eps), np.asarray(sig)
    g = sig - (E * (eps - offset) + c)
    neg = np.nonzero(g[1:] <= 0)[0]
    if len(neg) == 0:
        return np.nan, np.nan
    k = neg[0] + 1
    t = g[k - 1] / (g[k - 1] - g[k])
    return (float(eps[k - 1] + t * (eps[k] - eps[k - 1])),
            float(sig[k - 1] + t * (sig[k] - sig[k - 1])))


def _bilinear_sse(ek, e, s):
    X = np.stack([e, np.maximum(0.0, e - ek)], 1)
    coef, *_ = np.linalg.lstsq(X, s, rcond=None)
    r = s - X @ coef
    return float(r @ r), coef


def bilinear_fit(eps, sig, eps_max):
    """Two-segment continuous fit through the origin on eps <= eps_max."""
    eps, sig = np.asarray(eps), np.asarray(sig)
    m = eps <= eps_max
    e, s = eps[m], sig[m]
    if len(e) < 8:
        raise ValueError("Too few points for the bilinear fit")
    cands = e[2:-2]
    sse = [_bilinear_sse(x, e, s)[0] for x in cands]
    i = int(np.argmin(sse))
    lo, hi = cands[max(i - 1, 0)], cands[min(i + 1, len(cands) - 1)]
    res = minimize_scalar(lambda x: _bilinear_sse(x, e, s)[0], bounds=(lo, hi),
                          method="bounded", options={"xatol": 1e-10})
    ek = float(res.x)
    _, (E1, dH) = _bilinear_sse(ek, e, s)
    return {"eps_y": ek, "sigma_y": float(E1 * ek), "E": float(E1), "H": float(E1 + dH)}


def extract(eps, sig, an: dict, sigma_y_in: float, E_in: float) -> dict:
    """Apply the configured yield-extraction method to one curve."""
    eps, sig = np.asarray(eps, float), np.asarray(sig, float)
    w = an["slope_window"]
    if an["slope"] == "fit":
        E, c = elastic_fit(eps, sig, w[0] * sigma_y_in, w[1] * sigma_y_in)
    elif an["slope"] == "material":
        E, c = E_in, 0.0
    else:
        raise ValueError(f"Unknown slope mode '{an['slope']}'")
    out = {"E_slope": E, "intercept": c}
    # proportional limit: first point deviating from the elastic line by > tol
    lin = E * eps + c
    dev = (np.abs(sig - lin) > an["prop_limit_tol"] * np.abs(lin)) & (sig > w[0] * sigma_y_in)
    out["sigma_prop"] = float(sig[int(np.argmax(dev))]) if dev.any() else np.nan
    if an["method"] == "offset":
        ey, sy = offset_intersection(eps, sig, E, c, float(an["offset_strain"]))
        out.update(eps_yield=ey, sigma_yield=sy, offset=float(an["offset_strain"]))
    elif an["method"] == "bilinear":
        b = bilinear_fit(eps, sig, float(an["bilinear_max_strain"]))
        out.update(eps_yield=b["eps_y"], sigma_yield=b["sigma_y"], E_bilinear=b["E"],
                   H_bilinear=b["H"])
    else:
        raise ValueError(f"Unknown method '{an['method']}'")
    return out
