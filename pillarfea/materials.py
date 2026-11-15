"""Isotropic J2 material with isotropic hardening (true stress vs. true plastic strain)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS_P_TABLE_END = 10.0  # the hardening table is extended to this plastic strain


@dataclass
class Material:
    E: float
    nu: float
    sigma_y: float
    eps_p: np.ndarray       # plastic-strain table (starts at 0)
    sig: np.ndarray         # flow-stress table (starts at sigma_y)
    plastic: bool = True

    def flow_stress(self, ep):
        return np.interp(ep, self.eps_p, self.sig)

    def plastic_strain(self, s):
        """Inverse of the flow curve (0 below yield)."""
        s = np.asarray(s, float)
        if not self.plastic:
            return np.zeros_like(s)
        return np.where(s <= self.sigma_y, 0.0, np.interp(s, self.sig, self.eps_p))

    def offset_stress(self, offset: float) -> float:
        """Flow stress at a given plastic strain, i.e. the input-law offset yield stress."""
        return float(self.flow_stress(offset))

    def uniaxial_small_strain(self, eps):
        """Small-strain uniaxial stress for monotonic loading, sigma(eps)."""
        smax = float(self.flow_stress(EPS_P_TABLE_END))
        s = np.linspace(0.0, smax, 200001)
        e = s / self.E + self.plastic_strain(s)
        return np.interp(eps, e, s)

    def ccx_lines(self, name: str) -> list[str]:
        lines = [f"*MATERIAL, NAME={name}", "*ELASTIC", f"{self.E:.10g}, {self.nu:.10g}"]
        if self.plastic:
            lines.append("*PLASTIC")
            lines += [f"{s:.10g}, {e:.10g}" for s, e in zip(self.sig, self.eps_p)]
        return lines


def material_from_cfg(m: dict, plastic: bool = True) -> Material:
    E = float(m["E"])
    nu = float(m["nu"])
    sy = float(m["sigma_y_over_E"]) * E
    h = m["hardening"]
    if h["type"] == "linear":
        Et = float(h["Et_over_E"]) * E
        if not 0.0 <= Et < E:
            raise ValueError("Et_over_E must be in [0, 1)")
        H = E * Et / (E - Et)  # plastic hardening modulus d(sigma)/d(eps_p)
        ep = np.array([0.0, EPS_P_TABLE_END])
        sg = np.array([sy, sy + H * EPS_P_TABLE_END])
    elif h["type"] == "table":
        ep = np.asarray(h["plastic_strain"], float)
        sg = np.asarray(h["stress_over_sigma_y"], float) * sy
        if ep[0] != 0.0 or abs(sg[0] - sy) > 1e-9 * sy:
            raise ValueError("Hardening table must start at (plastic strain 0, stress/sigma_y 1)")
        if np.any(np.diff(ep) <= 0) or np.any(np.diff(sg) < 0):
            raise ValueError("Hardening table must have increasing strain, non-decreasing stress")
        if ep[-1] < EPS_P_TABLE_END:  # extend with the slope of the last segment (stated assumption)
            slope = (sg[-1] - sg[-2]) / (ep[-1] - ep[-2]) if len(ep) > 1 else 0.0
            ep = np.append(ep, EPS_P_TABLE_END)
            sg = np.append(sg, sg[-1] + slope * (EPS_P_TABLE_END - ep[-2]))
    else:
        raise ValueError(f"Unknown hardening type '{h['type']}'")
    return Material(E, nu, sy, ep, sg, plastic)


def substrate_material(cfg: dict) -> Material | None:
    s = cfg["substrate"]
    if s["type"] == "none":
        return None
    # same flow curve (true stress vs. plastic strain) as the pillar, own elastic constants
    p = material_from_cfg(cfg["material"])
    return Material(float(s["E"]), float(s["nu"]), p.sigma_y, p.eps_p, p.sig,
                    plastic=(s["type"] == "elastoplastic"))
