"""Parameter sweeps: grid expansion, parallel execution, CSV output.

Sweep file (YAML):
    base: configs/some_case.yaml      # optional, relative to the sweep file
    fixed: {substrate.type: elastic}  # optional dotted overrides applied to every case
    grid:                             # full factorial over these dotted keys
      geometry.taper_deg: [0, 1, 2, 3, 5]
      contact:                        # a list of dicts sets several keys under that prefix
        - {top: frictionless}
        - {top: coulomb, mu: 0.1}
        - {top: bonded}
    jobs: 3                           # cases run concurrently
    threads_per_job: 4
    output: results/sweeps/taper.csv  # relative to the project root
"""
from __future__ import annotations

import copy
import csv
import itertools
import math
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from .config import get_dotted, load_config, set_dotted
from .platform_info import portable
from .simulate import run_case

PARAM_COLUMNS = [
    "geometry.D_top", "geometry.h_over_D", "geometry.taper_deg", "geometry.fillet_radius",
    "material.E", "material.nu", "material.sigma_y_over_E", "material.hardening.type",
    "material.hardening.Et_over_E", "substrate.type", "substrate.E", "substrate.nu",
    "substrate.radius_over_D", "contact.top", "contact.mu", "contact.penalty_factor",
    "contact.stick_ratio", "loading.max_strain",
    "loading.n_increments", "mesh.n_radial", "mesh.n_axial", "analysis.method",
    "analysis.offset_strain", "analysis.slope"]


def expand(sweep: dict, base_dir: Path) -> list[tuple[dict, dict]]:
    base = sweep.get("base")
    cfg0 = load_config((base_dir / base) if base else None, sweep.get("fixed"))
    grid = sweep.get("grid") or {}
    keys = list(grid)
    cases = []
    for combo in itertools.product(*[grid[k] for k in keys]) if keys else [()]:
        cfg = copy.deepcopy(cfg0)
        ov = {}
        for k, v in zip(keys, combo):
            if isinstance(v, dict):          # several keys under one grid axis
                for kk, vv in v.items():
                    ov[f"{k}.{kk}"] = vv
            else:
                ov[k] = v
        for k, v in ov.items():
            set_dotted(cfg, k, v)
        cases.append((ov, cfg))
    return cases


def contact_label(row: dict) -> str:
    """'frictionless' | 'bonded' | 'μ = 0.3' — one label per contact condition."""
    top = row.get("contact.top")
    if top == "coulomb":
        return f"μ = {float(row.get('contact.mu', 0)):g}"
    return str(top)


def result_row(res: dict) -> dict:
    rc = res["config"]
    row = {k: get_dotted(rc, k) for k in PARAM_COLUMNS}
    row["contact"] = contact_label(row)
    for k, v in res.items():
        if k not in ("curves", "config") and not isinstance(v, (dict, list)):
            row[k] = v
    return row


def run_cases(cfgs: list[dict], jobs: int = 3, threads_per_job: int | None = None,
              progress=None, force: bool = False) -> list:
    """Run configs concurrently; returns results (or Exception) in input order."""
    cfgs = [copy.deepcopy(c) for c in cfgs]
    if threads_per_job:
        for c in cfgs:
            c["solver"]["threads"] = threads_per_job
    out: list = [None] * len(cfgs)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        futs = {ex.submit(run_case, c, force): i for i, c in enumerate(cfgs)}
        for n, f in enumerate(as_completed(futs), 1):
            i = futs[f]
            try:
                out[i] = f.result()
            except Exception as e:  # keep going; failures are reported in the CSV
                e.trace = traceback.format_exc()
                out[i] = e
            if progress:
                progress(n, len(cfgs), i, out[i])
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.8g}" if isinstance(v, float) and math.isfinite(v)
                            else portable(v) if isinstance(v, str) else v)
                        for k, v in r.items()})


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_sweep_file(path: str | Path, project_root: Path, force: bool = False,
                   jobs: int | None = None) -> tuple[list[dict], Path]:
    path = Path(path)
    sweep = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases = expand(sweep, path.parent)
    n = len(cases)
    print(f"{n} cases, {jobs or sweep.get('jobs', 3)} concurrent")

    def prog(k, total, i, r):
        tag = "FAILED: " + str(r).splitlines()[0] if isinstance(r, Exception) else \
            f"C_raw={r['C_raw']:.4f}  ({r['runtime_s']:.0f} s)"
        print(f"[{k}/{total}] {cases[i][0]}  {tag}", flush=True)

    results = run_cases([c for _, c in cases], jobs or sweep.get("jobs", 3),
                        sweep.get("threads_per_job", 4), prog, force)
    rows = []
    for (ov, cfg), r in zip(cases, results):
        if isinstance(r, Exception):
            rows.append({**ov, "error": str(r).splitlines()[0]})
        else:
            rows.append(result_row(r))
    out = project_root / sweep.get("output", f"results/sweeps/{path.stem}.csv")
    write_csv(rows, out)
    print(f"wrote {portable(out)}")
    return rows, out
