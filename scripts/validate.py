"""Validation suite. Writes CSVs, figures and a pass/fail summary to results/validation/.

    python scripts/validate.py [--jobs 3] [--only uniaxial taper mesh increments substrate]

Groups
  uniaxial    straight cylinder, rigid frictionless base: FE pillar vs ideal-geometry curve,
              ideal-geometry curve vs an independent analytical finite-strain estimate
  taper       taper 0-5 deg x {rigid base, elastic substrate} x {frictionless, bonded}, + 1D model
  mesh        mesh convergence at 3 deg (frictionless and bonded)
  increments  load-increment convergence at 3 deg
  substrate   substrate-domain-size sensitivity at 3 deg
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pillarfea import plotting as P  # noqa: E402
from pillarfea.analysis import extract  # noqa: E402
from pillarfea.config import load_config, set_dotted  # noqa: E402
from pillarfea.materials import material_from_cfg  # noqa: E402
from pillarfea.simulate import load_frames  # noqa: E402
from pillarfea.sweep import result_row, run_cases, write_csv  # noqa: E402

OUT = ROOT / "results" / "validation"
BASE = load_config(ROOT / "configs" / "validation_base.yaml")
TAPERS = [0.0, 1.0, 2.0, 3.0, 5.0]
MUS = [0.0, 0.1, 0.3, 5.0]
FILLETS = [0.0, 0.05, 0.1, 0.2]   # µm, with D_top = 1 µm


def cfg(**ov):
    c = copy.deepcopy(BASE)
    for k, v in ov.items():
        set_dotted(c, k.replace("__", "."), v)
    return c


def build_cases(groups):
    cases = {}
    if "uniaxial" in groups or "taper" in groups:
        for t in TAPERS:
            for sub in ("none", "elastic"):
                for ct in ("frictionless", "bonded"):
                    cases[("taper", t, sub, ct)] = cfg(geometry__taper_deg=t, substrate__type=sub,
                                                       contact__top=ct)
    if "mesh" in groups:
        for n in (8, 12, 16, 24):
            for ct in ("frictionless", "bonded"):
                cases[("mesh", n, ct)] = cfg(geometry__taper_deg=3.0, contact__top=ct,
                                             mesh__n_radial=n)
    if "increments" in groups:
        for n in (50, 100, 200, 400):
            cases[("inc", n)] = cfg(geometry__taper_deg=3.0, loading__n_increments=n)
    if "substrate" in groups:
        for R in (20.0, 40.0, 80.0):
            cases[("sub", R)] = cfg(geometry__taper_deg=3.0, substrate__radius_over_D=R)
    if "contact" in groups:
        for ct in ("frictionless", "bonded"):   # exact limits
            cases[("taper", 3.0, "elastic", ct)] = cfg(geometry__taper_deg=3.0,
                                                       substrate__type="elastic", contact__top=ct)
        for mu in MUS:
            cases[("coulomb", mu)] = cfg(geometry__taper_deg=3.0, contact__top="coulomb",
                                         contact__mu=mu)
        for pf in (10.0, 1000.0):
            cases[("penalty", pf)] = cfg(geometry__taper_deg=3.0, contact__top="coulomb",
                                         contact__mu=0.0, contact__penalty_factor=pf)
        cases[("stick", 0.01)] = cfg(geometry__taper_deg=3.0, contact__top="coulomb",
                                     contact__mu=0.1, contact__stick_ratio=0.01)
    if "fillet" in groups:
        for fr in FILLETS:
            cases[("fillet", fr)] = cfg(geometry__taper_deg=3.0, geometry__fillet_radius=fr)
        for n in (12, 24):
            cases[("filletmesh", n)] = cfg(geometry__taper_deg=3.0, geometry__fillet_radius=0.1,
                                           mesh__n_radial=n)
    return cases


def analytic_finite_strain(c):
    """Homogeneous uniaxial compression, additive log-strain hypoelastic-plastic estimate.
    Independent of CalculiX; differs from ccx's hyperelastic-based model only at O(eps_e^2)."""
    mat = material_from_cfg(c["material"])
    e_eng = np.linspace(0, c["loading"]["max_strain"], 4001)
    e_log = -np.log(1 - e_eng)
    s_true = mat.uniaxial_small_strain(e_log)
    ep = mat.plastic_strain(s_true)
    lat = mat.nu * s_true / mat.E + 0.5 * ep
    return e_eng, s_true * np.exp(2 * lat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", nargs="*",
                    default=["uniaxial", "taper", "mesh", "increments", "substrate", "contact",
                             "fillet"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    P.style()
    cases = build_cases(a.only)
    keys = list(cases)
    print(f"{len(keys)} validation cases")

    def prog(k, n, i, r):
        msg = f"FAILED {r}" if isinstance(r, Exception) else f"{r['runtime_s']:.0f} s"
        print(f"[{k}/{n}] {keys[i]}: {msg}", flush=True)

    res_list = run_cases([cases[k] for k in keys], a.jobs, a.threads, prog)
    R = dict(zip(keys, res_list))
    failed = {k: v for k, v in R.items() if isinstance(v, Exception)}
    for k, e in failed.items():
        print("FAILED", k, getattr(e, "trace", e))
    R = {k: v for k, v in R.items() if not isinstance(v, Exception)}
    rows = [dict(case=str(k), **result_row(v)) for k, v in R.items()]
    write_csv(rows, OUT / "validation_cases.csv")
    lines = []

    def check(name, ok, detail):
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    # ---------------- uniaxial ----------------
    k = ("taper", 0.0, "none", "frictionless")
    if k in R:
        r = R[k]
        check("V1 cylinder/rigid base reproduces ideal geometry",
              abs(r["C_raw"] - 1) < 1e-3, f"C = {r['C_raw']:.5f}")
        check("V1 force equilibrium", r["equilibrium_err"] < 1e-6,
              f"max |F_punch - F_support| / F = {r['equilibrium_err']:.2e}")
        c = r["config"]
        ea, sa = analytic_finite_strain(c)
        xa = extract(ea, sa, c["analysis"], r["sigma_y_input"], r["E_input"])
        d = r["sigma_ref"] / xa["sigma_yield"] - 1
        check("V1 ideal-geometry FE vs analytical finite-strain estimate", abs(d) < 3e-3,
              f"sigma_0.2: FE {r['sigma_ref']:.1f} MPa, analytic {xa['sigma_yield']:.1f} MPa "
              f"({d * 100:+.2f} %)")
        lines.append(f"     input-law sigma at 0.2% plastic strain = {r['sigma_offset_input_law']:.1f} MPa;"
                     f" ideal-geometry engineering sigma_0.2 = {r['sigma_ref']:.1f} MPa "
                     f"(compression: F/A0 > true stress)")
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        cv = r["curves"]
        ax.plot(cv["eps_law"] * 100, cv["sigma_law"], color=P.SERIES[5], ls=":", label="input law (true stress, small strain)")
        ax.plot(ea * 100, sa, color=P.SERIES[1], lw=1.2, label="analytical finite-strain, engineering")
        ax.plot(cv["eps_ref"] * 100, cv["sigma_ref"], color=P.INK2, ls="--", label="ideal geometry (1 element FE)")
        ax.plot(cv["eps"]["raw"] * 100, cv["sigma"], color=P.SERIES[0], lw=1.0, label="cylinder, rigid base (FE)")
        ax.set_xlabel("nominal strain (%)"); ax.set_ylabel("nominal stress (MPa)")
        ax.set_xlim(0, None); ax.set_ylim(0, None); ax.legend(loc="lower right")
        fig.savefig(OUT / "V1_uniaxial.png"); plt.close(fig)
    k = ("taper", 0.0, "elastic", "frictionless")
    if k in R:
        r = R[k]
        check("V2 cylinder on elastic substrate", abs(r["C_raw"] - 1) < 5e-3,
              f"C_raw = {r['C_raw']:.5f}, C_sneddon = {r['C_sneddon']:.5f}, C_fe_base = {r['C_fe_base']:.5f}")
        check("V2 force equilibrium", r["equilibrium_err"] < 1e-5, f"{r['equilibrium_err']:.2e}")
        lines.append(f"     apparent modulus: raw {r['E_slope_raw']:.0f}, Sneddon-corrected "
                     f"{r['E_slope_sneddon']:.0f}, FE-foot-corrected {r['E_slope_fe_base']:.0f}, "
                     f"ideal {r['E_ref']:.0f} MPa")

    # ---------------- taper ----------------
    if any(k[0] == "taper" for k in R):
        fig, ax = plt.subplots(figsize=(4.4, 3.3))
        i = 0
        for sub in ("none", "elastic"):
            for ct in ("frictionless", "bonded"):
                pts = [(t, R[("taper", t, sub, ct)]["C_raw"]) for t in TAPERS
                       if ("taper", t, sub, ct) in R]
                if pts:
                    x, y = zip(*pts)
                    lab = f"{'rigid base' if sub == 'none' else 'elastic substrate'}, {ct}"
                    ax.plot(x, y, "o-", color=P.SERIES[i], ms=4, mec="white", label=lab)
                i += 1
        pts = [(t, R[("taper", t, "none", "frictionless")]["C_1d"]) for t in TAPERS
               if ("taper", t, "none", "frictionless") in R]
        if pts:
            x, y = zip(*pts)
            ax.plot(x, y, "s--", color=P.INK2, ms=3.2, lw=1.0, label="1D progressive-yield model")
        P.plot_analytical(ax, [dict(t=t, **R[("taper", t, "none", "frictionless")])
                               for t in TAPERS if ("taper", t, "none", "frictionless") in R], "t")
        ax.set_xlabel("taper angle (°)")
        ax.set_ylabel("$C_{0.2} = \\sigma_{0.2,apparent} / \\sigma_{0.2,ideal}$")
        ax.legend(loc="upper left", fontsize=7)
        fig.savefig(OUT / "V3_C_vs_taper.png"); plt.close(fig)
        lines.append("     taper table (C_raw; apparent overestimate = C - 1):")
        lines.append("     taper  rigid/fricless  rigid/bonded  subst/fricless  subst/bonded  subst/fricless C_fe_base   1D")
        for t in TAPERS:
            v = [R.get(("taper", t, s, c)) for s in ("none", "elastic") for c in ("frictionless", "bonded")]
            if all(v):
                lines.append("     " + f"{t:4.1f}  " + "  ".join(f"{x['C_raw']:.4f} ({(x['C_raw'] - 1) * 100:+5.1f}%)" for x in v)
                             + f"  {v[2]['C_fe_base']:.4f}   {v[0]['C_1d']:.4f} ({(v[0]['C_1d'] - 1) * 100:+5.1f}%)")
        for t in TAPERS:
            r = R.get(("taper", t, "none", "frictionless"))
            if r:
                check(f"V3 equilibrium {t} deg", r["equilibrium_err"] < 1e-5, f"{r['equilibrium_err']:.1e}")
        # curves + fields at 3 deg
        for ct in ("frictionless", "bonded"):
            r = R.get(("taper", 3.0, "elastic", ct))
            if not r:
                continue
            fig, ax = plt.subplots(figsize=(4.6, 3.4))
            P.plot_stress_strain(ax, r, "raw")
            ax.set_title(f"3° taper, h/D = 2, elastic substrate, {ct} top")
            fig.savefig(OUT / f"V3_curve_3deg_{ct}.png"); plt.close(fig)
            mesh, fr = load_frames(r["workdir"])
            tf = fr["time"]
            t_y = np.interp(r["eps_y_raw"], r["curves"]["eps"]["raw"], r["curves"]["time"])
            k_y = int(np.argmin(np.abs(tf - t_y)))
            fig, axs = plt.subplots(1, 3, figsize=(11, 4.0), layout="constrained")
            for ax, (kk, fld, title) in zip(axs, [(k_y, "PEEQ", f"PEEQ at offset yield (t={tf[k_y]:.2f})"),
                                                   (len(tf) - 1, "PEEQ", "PEEQ at end of loading"),
                                                   (k_y, "mises", "von Mises at offset yield")]):
                P.plot_field(ax, mesh, fr, kk, fld)
                ax.set_title(title)
            fig.suptitle(f"3° taper, {ct} top, elastic substrate (deformation ×1)")
            fig.savefig(OUT / f"V3_fields_3deg_{ct}.png"); plt.close(fig)

    # ---------------- mesh ----------------
    if any(k[0] == "mesh" for k in R):
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        for i, ct in enumerate(("frictionless", "bonded")):
            pts = [(n, R[("mesh", n, ct)]["C_raw"]) for n in (8, 12, 16, 24) if ("mesh", n, ct) in R]
            if pts:
                x, y = zip(*pts)
                ax.plot(x, y, "o-", color=P.SERIES[i], ms=4, mec="white", label=ct)
                lines.append(f"     mesh {ct}: " + ", ".join(f"n={a}: C={b:.5f}" for a, b in pts))
                if len(pts) >= 2:
                    check(f"V4 mesh convergence ({ct}), n=16 vs n=24", abs(pts[-2][1] - pts[-1][1]) < 2e-3,
                          f"|dC| = {abs(pts[-2][1] - pts[-1][1]):.1e}")
        ax.set_xlabel("elements across top radius"); ax.set_ylabel("$C_{0.2}$")
        ax.set_title("3° taper, elastic substrate"); ax.legend()
        fig.savefig(OUT / "V4_mesh_convergence.png"); plt.close(fig)

    # ---------------- increments / substrate ----------------
    for grp, vals, label, fname in (("inc", (50, 100, 200, 400), "load increments", "V5_increments.png"),
                                    ("sub", (20.0, 40.0, 80.0), "substrate radius / $D_{top}$", "V6_substrate_size.png")):
        pts = [(v, R[(grp, v)]) for v in vals if (grp, v) in R]
        if not pts:
            continue
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        ax.plot([p[0] for p in pts], [p[1]["C_raw"] for p in pts], "o-", color=P.SERIES[0], ms=4, mec="white", label="C (fitted slope)")
        ax.set_xlabel(label); ax.set_ylabel("$C_{0.2}$"); ax.set_title("3° taper, frictionless, elastic substrate")
        fig.savefig(OUT / fname); plt.close(fig)
        lines.append(f"     {grp}: " + ", ".join(f"{p[0]:g}: C={p[1]['C_raw']:.5f}, E_app={p[1]['E_slope_raw']:.0f}" for p in pts))
        check(f"V5/6 {grp} sensitivity (last two)", abs(pts[-1][1]["C_raw"] - pts[-2][1]["C_raw"]) < 1e-3,
              f"|dC| = {abs(pts[-1][1]['C_raw'] - pts[-2][1]['C_raw']):.1e}")

    # ---------------- Coulomb contact ----------------
    if any(k[0] == "coulomb" for k in R):
        fr = R.get(("taper", 3.0, "elastic", "frictionless"))
        bo = R.get(("taper", 3.0, "elastic", "bonded"))
        mus = [(mu, R[("coulomb", mu)]) for mu in MUS if ("coulomb", mu) in R]
        lines.append("     Coulomb contact (3°, elastic substrate, penalty factor 100): " +
                     ", ".join(f"μ={m:g}: C={r['C_raw']:.5f}" for m, r in mus))
        if fr and ("coulomb", 0.0) in R:
            d = R[("coulomb", 0.0)]["C_raw"] - fr["C_raw"]
            check("V7 Coulomb μ=0 vs exact frictionless", abs(d) < 1e-3,
                  f"{R[('coulomb', 0.0)]['C_raw']:.5f} vs {fr['C_raw']:.5f} ({d:+.1e})")
        if bo and ("coulomb", 5.0) in R:
            d = R[("coulomb", 5.0)]["C_raw"] - bo["C_raw"]
            check("V7 Coulomb μ=5 vs exact bonded", abs(d) < 1e-3,
                  f"{R[('coulomb', 5.0)]['C_raw']:.5f} vs {bo['C_raw']:.5f} ({d:+.1e})")
        cs = [r["C_raw"] for _, r in mus]
        check("V7 C non-decreasing with μ", all(b >= a - 2e-4 for a, b in zip(cs, cs[1:])),
              ", ".join(f"{c:.5f}" for c in cs))
        pen = [(pf, R[k]["C_raw"]) for pf, k in ((10.0, ("penalty", 10.0)), (100.0, ("coulomb", 0.0)),
                                                 (1000.0, ("penalty", 1000.0))) if k in R]
        if len(pen) == 3:
            lines.append("     penalty sensitivity (μ=0): " + ", ".join(f"factor {p:g}: C={c:.5f}" for p, c in pen))
            check("V7 penalty factor 100 vs 1000", abs(pen[1][1] - pen[2][1]) < 5e-4,
                  f"|dC| = {abs(pen[1][1] - pen[2][1]):.1e}")
        if ("stick", 0.01) in R and ("coulomb", 0.1) in R:
            d = R[("stick", 0.01)]["C_raw"] - R[("coulomb", 0.1)]["C_raw"]
            check("V7 stick slope 0.01 K vs 0.1 K (μ=0.1)", abs(d) < 1e-3, f"dC = {d:+.1e}")
        fig, ax = plt.subplots(figsize=(4.4, 3.3))
        xs = np.arange(len(mus))
        ax.plot(xs, cs, "o-", color=P.SERIES[0], ms=4, mec="white", label="Coulomb contact")
        if fr:
            ax.axhline(fr["C_raw"], color=P.SERIES[2], lw=1, ls="--", label="frictionless (exact)")
        if bo:
            ax.axhline(bo["C_raw"], color=P.SERIES[1], lw=1, ls="--", label="bonded (exact)")
        ax.set_xticks(xs, [f"{m:g}" for m, _ in mus])
        ax.set_xlabel("friction coefficient μ"); ax.set_ylabel("$C_{0.2}$")
        ax.set_title("3° taper, h/D = 2, elastic substrate"); ax.legend(fontsize=7)
        fig.savefig(OUT / "V7_contact.png"); plt.close(fig)

    # ---------------- fillet ----------------
    if any(k[0] == "fillet" for k in R):
        pts = [(f, R[("fillet", f)]) for f in FILLETS if ("fillet", f) in R]
        lines.append("     fillet (3°, frictionless, D_top = 1 µm): " +
                     ", ".join(f"r={f:g} µm: C={r['C_raw']:.5f}" for f, r in pts))
        fm = [(n, R[k]["C_raw"]) for n, k in ((12, ("filletmesh", 12)), (16, ("fillet", 0.1)),
                                              (24, ("filletmesh", 24))) if k in R]
        if len(fm) == 3:
            lines.append("     fillet 0.1 µm mesh: " + ", ".join(f"n={n}: C={c:.5f}" for n, c in fm))
            check("V8 mesh convergence with fillet, n=16 vs n=24", abs(fm[1][1] - fm[2][1]) < 2e-3,
                  f"|dC| = {abs(fm[1][1] - fm[2][1]):.1e}")
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        ax.plot([p[0] * 1000 for p in pts], [p[1]["C_raw"] for p in pts], "o-", color=P.SERIES[0],
                ms=4, mec="white")
        ax.set_xlabel("fillet radius (nm), $D_{top}$ = 1 µm"); ax.set_ylabel("$C_{0.2}$")
        ax.set_title("3° taper, frictionless, elastic substrate")
        fig.savefig(OUT / "V8_fillet.png"); plt.close(fig)

    if failed:
        lines.append(f"{len(failed)} cases FAILED: {list(failed)}")
    text = "\n".join(lines)
    (OUT / "validation_summary.txt").write_text(text, encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):  # piped output on Windows defaults to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)


if __name__ == "__main__":
    main()
