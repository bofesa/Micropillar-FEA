"""Run one case: FE solve (cached), homogeneous reference, 1D model, yield extraction.

The FE solution is cached under solver.work_dir/<hash of the FE-relevant config>, so
changing only post-processing settings (method, offset, slope, ...) never re-runs CalculiX.

Reference ("ideal geometry") curve: a single CAX8R element in homogeneous uniaxial
compression with the same material, kinematics (NLGEOM) and loading; i.e. what a perfect
uniaxial test on the input material would give. The same extraction method is applied to it.
"""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import numpy as np

from . import ccx_io
from .analysis import extract
from .config import fe_key, legacy_keys, reference_key, resolve
from .geometry import Geometry, Mesh, build_mesh, geometry_from_cfg
from .materials import material_from_cfg, substrate_material
from .model1d import tapered_curve

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock(key: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def _save_mesh(path: Path, mesh: Mesh) -> None:
    np.savez(path, nodes=mesh.nodes, elements=mesh.elements,
             **{f"elset_{k}": v for k, v in mesh.elsets.items()},
             **{f"nset_{k}": v for k, v in mesh.nsets.items()})


def load_mesh(path: Path) -> Mesh:
    d = np.load(path)
    return Mesh(d["nodes"], d["elements"],
                {k[6:]: d[k] for k in d.files if k.startswith("elset_")},
                {k[5:]: d[k] for k in d.files if k.startswith("nset_")})


def _solve(workdir: Path, mesh: Mesh, geo: Geometry, rc: dict, pillar, sub) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    ld, sol = rc["loading"], rc["solver"]
    t0 = time.perf_counter()
    force_set = ccx_io.write_inp(workdir / "job.inp", mesh, pillar, sub, rc["contact"],
                                 float(ld["max_strain"]) * geo.h, int(ld["n_increments"]),
                                 sol.get("linear_solver", "spooles"))
    attempts = ccx_io.run_ccx(workdir, sol["executable"], threads=int(sol["threads"]),
                              timeout=float(sol["timeout_s"]),
                              max_attempts=int(sol.get("max_attempts", 3)))
    dat = ccx_io.parse_dat(workdir / "job.dat")
    support = "OUTER" if sub is not None else "PBASE"
    rf, rfs, ut = dat[("RF", force_set)], dat[("RF", support)], dat[("U", "TOPREF")]
    times = np.array(sorted(rf))
    if abs(times[-1] - 1.0) > 1e-6:
        raise RuntimeError(f"Analysis stopped at step time {times[-1]:.4f} < 1")
    F = np.array([-rf[t][1] for t in times]) * ccx_io.AXI_FORCE_FACTOR
    F_sup = np.array([rfs[t][1] for t in times]) * ccx_io.AXI_FORCE_FACTOR
    delta = np.array([-next(iter(ut[t].values()))[1] for t in times])
    # lift-off indicator: smallest compressive nodal punch force / total force (<0 => tension)
    # (axis nodes excluded: zero tributary area, their reaction is numerical noise ~1e-7 F)
    rfn = dat.get(("RFN", "TOP"), {})
    off_axis = {int(i) + 1 for i in mesh.nsets["TOP"] if mesh.nodes[i, 0] > 1e-9}
    top_min = np.array([min(-v[1] for n, v in rfn[t].items() if n in off_axis)
                        * ccx_io.AXI_FORCE_FACTOR / F[k]
                        if t in rfn and F[k] > 0 else np.nan for k, t in enumerate(times)])
    ub = np.zeros_like(F)
    if sub is not None:
        ids = mesh.nsets["PBASE"]
        r = mesh.nodes[ids, 0]
        o = np.argsort(r)
        ub_nodes = dat[("U", "PBASE")]
        for k, t in enumerate(times):
            u = np.array([-ub_nodes[t][i + 1][1] for i in ids])[o]
            ub[k] = np.trapezoid(u * r[o], r[o]) / np.trapezoid(r[o], r[o])
    # keep n_field_frames evenly spaced field frames, then drop the (large) frd file
    nf = max(1, int(ld["n_field_frames"]))
    idx = np.unique(np.round(np.linspace(0, len(times) - 1, nf + 1)[1:]).astype(int))
    frames = ccx_io.parse_frd(workdir / "job.frd", len(mesh.nodes), keep_times=times[idx])
    np.savez(workdir / "frames.npz", **frames)
    if not rc["solver"].get("keep_frd", False):
        (workdir / "job.frd").unlink(missing_ok=True)
    raw = dict(time=np.r_[0.0, times], F=np.r_[0.0, F], F_support=np.r_[0.0, F_sup],
               delta=np.r_[0.0, delta], u_base=np.r_[0.0, ub], top_min_ratio=np.r_[np.nan, top_min],
               runtime_s=time.perf_counter() - t0, solver_attempts=attempts,
               solver_executable=str(sol["executable"]))
    np.savez(workdir / "raw.npz", **raw)
    return raw


def _cache_dir(rc: dict, key: str, legacy: str) -> Path:
    """Cache directory for `key`; migrates a result stored under the pre-v0.2 key."""
    work = Path(rc["solver"]["work_dir"])
    wd, old = work / key, work / legacy
    if not (wd / "raw.npz").exists() and (old / "raw.npz").exists() and old != wd:
        if wd.exists():
            import shutil
            shutil.rmtree(wd)
        old.rename(wd)
    return wd


def run_fe(rc: dict, force: bool = False) -> tuple[dict, Path]:
    key = fe_key(rc)
    with _lock(key):
        wd = _cache_dir(rc, key, legacy_keys(rc)[0])
        if (wd / "raw.npz").exists() and (wd / "mesh.npz").exists() and not force:
            return dict(np.load(wd / "raw.npz")), wd
        geo = geometry_from_cfg(rc)
        mesh = build_mesh(geo, rc["mesh"], punch=rc["contact"]["top"] == "coulomb")
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "config.json").write_text(json.dumps(rc, indent=1))
        _save_mesh(wd / "mesh.npz", mesh)
        raw = _solve(wd, mesh, geo, rc, material_from_cfg(rc["material"]), substrate_material(rc))
        return raw, wd


def run_reference(rc: dict, force: bool = False) -> dict:
    """Homogeneous uniaxial compression of one element: the ideal-geometry curve."""
    key = reference_key(rc)
    with _lock(key):
        wd = _cache_dir(rc, key, legacy_keys(rc)[1])
        if (wd / "raw.npz").exists() and not force:
            return dict(np.load(wd / "raw.npz"))
        geo = Geometry(D_top=2.0, h=1.0, taper_deg=0.0, substrate="none")
        mesh = build_mesh(geo, {"n_radial": 1, "n_axial": 1, "radial_bias": 0, "axial_bias": 0})
        rr = dict(rc)
        rr["contact"] = {"top": "frictionless", "mu": 0.0}
        rr["loading"] = dict(rc["loading"], n_field_frames=1)
        return _solve(wd, mesh, geo, rr, material_from_cfg(rc["material"]), None)


def load_frames(workdir: Path) -> tuple[Mesh, dict]:
    """Mesh and stored field frames {'time','U','S','PEEQ'} of a finished case."""
    mesh = load_mesh(Path(workdir) / "mesh.npz")
    return mesh, dict(np.load(Path(workdir) / "frames.npz"))


def analyse(rc: dict, raw: dict, ref: dict) -> dict:
    """Nominal curves, yield extraction for each compliance treatment, correction factors."""
    geo = geometry_from_cfg(rc)
    mat = material_from_cfg(rc["material"])
    an = rc["analysis"]
    E, sy = mat.E, mat.sigma_y
    sig = raw["F"] / geo.A_top
    curves = {"raw": raw["delta"] / geo.h}
    if rc["substrate"]["type"] != "none":
        Es, nus = float(rc["substrate"]["E"]), float(rc["substrate"]["nu"])
        d_sned = raw["F"] * (1 - nus ** 2) / (Es * geo.D_bot)    # Sneddon flat punch, foot diam.
        curves["sneddon"] = (raw["delta"] - d_sned) / geo.h
        curves["fe_base"] = (raw["delta"] - raw["u_base"]) / geo.h
    eps_ref = ref["delta"] / 1.0
    sig_ref = ref["F"] / math.pi
    x_ref = extract(eps_ref, sig_ref, an, sy, E)
    s_ref = x_ref["sigma_yield"]

    res = {"sigma_y_input": sy, "E_input": E, "sigma_ref": s_ref, "E_ref": x_ref["E_slope"],
           "eps_y_ref": x_ref["eps_yield"], "D_bot": geo.D_bot, "A_top": geo.A_top,
           "equilibrium_err": float(np.max(np.abs(raw["F"] - raw["F_support"]))
                                    / max(np.max(np.abs(raw["F"])), 1e-300)),
           "runtime_s": float(raw.get("runtime_s", np.nan)),
           "solver_attempts": int(raw.get("solver_attempts", 1)),
           "solver_executable": str(raw.get("solver_executable", "")),
           # < 0 would mean a top-face node pulls on the punch (contact would open there)
           "top_min_nodal_force_ratio": (float(np.nanmin(raw["top_min_ratio"]))
                                         if "top_min_ratio" in raw
                                         and np.isfinite(raw["top_min_ratio"]).any() else np.nan)}
    res.update(geo.analytical_corrections())
    if an["method"] == "offset":
        s_in = mat.offset_stress(float(an["offset_strain"]))
        res["sigma_offset_input_law"] = s_in
    for name, eps in curves.items():
        x = extract(eps, sig, an, sy, E)
        sa = x["sigma_yield"]
        res[f"sigma_app_{name}"] = sa
        res[f"eps_y_{name}"] = x["eps_yield"]
        res[f"E_slope_{name}"] = x["E_slope"]
        res[f"intercept_{name}"] = x["intercept"]
        res[f"sigma_prop_{name}"] = x["sigma_prop"]
        res[f"C_{name}"] = sa / s_ref          # C = σ_apparent / σ_ideal (> 1: over-estimate)
        if an["method"] == "offset":
            res[f"C_inputlaw_{name}"] = sa / res["sigma_offset_input_law"]
        d_eff = geo.D_top * math.sqrt(sa / s_ref) if sa > 0 else np.nan
        res[f"Deff_over_Dtop_{name}"] = d_eff / geo.D_top
        res[f"zeff_from_top_over_h_{name}"] = ((d_eff - geo.D_top) / (2 * geo.h * geo.tan_t)
                                               if geo.tan_t > 0 else np.nan)
    # 1D progressive-yield model with the same extraction (small strain, rigid base)
    e1, s1 = tapered_curve(geo, mat, float(rc["loading"]["max_strain"]))
    e0 = np.linspace(0.0, float(rc["loading"]["max_strain"]), 2001)
    s0 = mat.uniaxial_small_strain(e0)
    try:
        x1, x0 = extract(e1, s1, an, sy, E), extract(e0, s0, an, sy, E)
        res.update(sigma_app_1d=x1["sigma_yield"], sigma_ref_1d=x0["sigma_yield"],
                   C_1d=x1["sigma_yield"] / x0["sigma_yield"])
    except ValueError:
        res.update(sigma_app_1d=np.nan, sigma_ref_1d=np.nan, C_1d=np.nan)
    res["curves"] = {"sigma": sig, "eps": curves, "eps_ref": eps_ref, "sigma_ref": sig_ref,
                     "eps_1d": e1, "sigma_1d": s1, "eps_law": e0, "sigma_law": s0,
                     "F": raw["F"], "delta": raw["delta"], "u_base": raw["u_base"],
                     "time": raw["time"]}
    return res


def run_case(cfg: dict, force: bool = False) -> dict:
    rc = resolve(cfg)
    raw, wd = run_fe(rc, force)
    ref = run_reference(rc, force)
    res = analyse(rc, raw, ref)
    res["workdir"] = str(wd)
    res["config"] = rc
    return res


PRIMARY = {"none": "raw", "elastic": "sneddon", "elastoplastic": "sneddon"}
