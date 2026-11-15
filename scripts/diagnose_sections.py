"""Section-averaged stresses and plastic strain along the pillar at the offset-yield frame,
compared with the 1D progressive-yield model assumptions (sigma = F/A(z), uniaxial).

    python scripts/diagnose_sections.py [config.yaml] [--set geometry.taper_deg=3 ...]

Explains differences between C (FE) and C_1d: lateral confinement of the yielding zone shows
up as compressive <s_rr>, <s_tt> and an axial stress above the flow stress.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from pillarfea import plotting as P  # noqa: E402
from pillarfea.ccx_io import von_mises  # noqa: E402
from pillarfea.config import load_config  # noqa: E402
from pillarfea.geometry import geometry_from_cfg  # noqa: E402
from pillarfea.materials import material_from_cfg  # noqa: E402
from pillarfea.simulate import load_frames, run_case  # noqa: E402
from pillarfea.platform_info import portable  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    ov = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in a.set)}
    r = run_case(load_config(a.config, ov))
    mesh, fr = load_frames(r["workdir"])
    mat, geo = material_from_cfg(r["config"]["material"]), geometry_from_cfg(r["config"])
    t_y = np.interp(r["eps_y_raw"], r["curves"]["eps"]["raw"], r["curves"]["time"])
    k = int(np.argmin(np.abs(fr["time"] - t_y)))
    F = np.interp(fr["time"][k], r["curves"]["time"], r["curves"]["F"])
    X = mesh.nodes
    pn = np.unique(mesh.elements[mesh.elsets["PILLAR"]])
    out = []
    for z in np.unique(np.round(X[pn, 1], 9)):
        ids = pn[np.abs(X[pn, 1] - z) < 1e-9]
        ids = ids[np.argsort(X[ids, 0])]
        rr = X[ids, 0]
        A = np.trapezoid(rr, rr)
        m = lambda v: np.trapezoid(v * rr, rr) / A  # noqa: E731
        S = fr["S"][k][ids].astype(float)
        pe = np.nan_to_num(fr["PEEQ"][k][ids].astype(float))
        out.append([z / geo.h, -m(S[:, 1]), -m(S[:, 0]), -m(S[:, 2]), m(von_mises(S)), m(pe)])
    d = np.array(out)
    s1 = F / (np.pi * geo.outer_radius(d[:, 0] * geo.h) ** 2)
    e1 = mat.plastic_strain(s1)
    print(f"C = {r['C_raw']:.4f}, C_1d = {r['C_1d']:.4f}; frame t = {fr['time'][k]:.3f} "
          f"(offset yield at t = {t_y:.3f}), F/A_top = {F / geo.A_top:.1f} MPa")
    print("  z/h   -<s_zz>  F/A(z)  -<s_rr>  -<s_tt>  <mises>  flow(<PEEQ>)  <PEEQ>   PEEQ_1D")
    for row, a1, b1 in list(zip(d, s1, e1))[::max(1, len(d) // 15)]:
        print(f"{row[0]:6.3f} {row[1]:8.1f} {a1:7.1f} {row[2]:8.1f} {row[3]:8.1f} {row[4]:8.1f} "
              f"{mat.flow_stress(row[5]):10.1f}  {row[5]:.5f}  {b1:.5f}")
    print(f"plastic shortening / h:  FE {np.trapezoid(d[:, 5], d[:, 0]):.5f}   "
          f"1D (same force) {np.trapezoid(e1, d[:, 0]):.5f}")
    P.style()
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.4), sharey=True)
    ax[0].plot(d[:, 1], d[:, 0], color=P.SERIES[0], label="FE $-\\langle\\sigma_{zz}\\rangle$")
    ax[0].plot(s1, d[:, 0], "--", color=P.INK2, label="1D $F/A(z)$")
    ax[0].plot(-d[:, 2] + d[:, 1], d[:, 0], color=P.SERIES[1], lw=1,
               label="FE $\\langle\\sigma_{rr}\\rangle - \\langle\\sigma_{zz}\\rangle$")
    ax[0].set_xlabel("stress (MPa)"); ax[0].set_ylabel("z / h"); ax[0].legend(loc="lower left")
    ax[1].plot(d[:, 5], d[:, 0], color=P.SERIES[0], label="FE section mean")
    ax[1].plot(e1, d[:, 0], "--", color=P.INK2, label="1D")
    ax[1].set_xlabel("equivalent plastic strain"); ax[1].legend()
    fig.suptitle(f"at offset yield, nominal {F / geo.A_top:.0f} MPa")
    outp = ROOT / "results" / "diagnostics" / f"sections_{Path(r['workdir']).name}.png"
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp)
    print(f"figure: {portable(outp)}")


if __name__ == "__main__":
    main()
