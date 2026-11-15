"""Benchmark one representative simulation with every available solver configuration.

    python scripts/benchmark_solvers.py [--threads 1 4] [--solvers spooles pastix]
                                        [--repeats 3] [--builds clangarm64 ccx_MT] [--no-save]

Representative case: configs/validation_base.yaml with a 3 deg taper, elastic substrate,
frictionless top (≈1300 CAX8R elements, 200 increments). Every runnable CalculiX build found
by pillarfea.platform_info is run with each thread count and linear solver, `--repeats`
times, in isolated work directories (result cache bypassed, crash-retry disabled so that
crashes are counted). Configurations run one at a time so timings are not disturbed.

Selection: among configurations with NO crash in any repeat, the one with the smallest
median wall time; all completed runs must agree (C within 1e-4). The choice is written to
the solver profile used by `solver.executable: auto`. Keep the machine awake while this
runs; sleep inflates timings (see README).
"""
from __future__ import annotations

import argparse
import copy
import shutil
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pillarfea.config import load_config, resolve  # noqa: E402
from pillarfea.platform_info import discover_solvers, host_arch, portable, save_profile  # noqa: E402
from pillarfea.simulate import analyse, run_fe, run_reference  # noqa: E402
from pillarfea.sweep import write_csv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="*", default=[1, 4])
    ap.add_argument("--solvers", nargs="*", default=["spooles", "pastix"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--builds", nargs="*", default=None,
                    help="only builds whose label contains one of these substrings")
    a = ap.parse_args()
    base = load_config(ROOT / "configs" / "validation_base.yaml",
                       {"geometry.taper_deg": 3.0, "substrate.type": "elastic",
                        "contact.top": "frictionless"})
    host = host_arch()
    sols = discover_solvers()
    if a.builds:
        sols = [s for s in sols if any(b in s.label for b in a.builds)]
    print(f"host {host}; {len(sols)} solver builds; {a.repeats} repeats per configuration")
    bench_dir = Path(resolve(copy.deepcopy(base))["solver"]["work_dir"]).parent / "bench"
    runs, configs = [], []
    for s in sols:
        # the classic x64 ccx.exe is single-threaded; its _MT sibling is the multithreaded build
        thread_list = a.threads if (s.multithread_variant or s.native) else [1]
        for t in thread_list:
            for ls in a.solvers:
                cfg_rows = []
                for rep in range(a.repeats):
                    tag = f"{s.label}|{ls}|t{t}"
                    c = copy.deepcopy(base)
                    wd = bench_dir / (tag.replace("/", "_").replace("|", "_").replace(" ", "_")
                                      + f"_r{rep}")
                    c["solver"].update(executable=s.path, threads=t, linear_solver=ls,
                                       work_dir=str(wd), max_attempts=1)
                    rc = resolve(c)
                    shutil.rmtree(wd, ignore_errors=True)
                    row = dict(build=s.label, arch=s.arch, mode="native" if s.native else "emulated",
                               linear_solver=ls, threads=t, repeat=rep, path=s.path)
                    t0 = time.perf_counter()
                    try:
                        raw, _ = run_fe(rc, force=True)
                        ref = run_reference(rc, force=True)
                        res = analyse(rc, raw, ref)
                        row.update(wall_s=round(time.perf_counter() - t0, 1), C_raw=res["C_raw"],
                                   sigma_app_raw=res["sigma_app_raw"], status="ok")
                    except Exception as e:
                        kind = type(e).__name__
                        row.update(wall_s=round(time.perf_counter() - t0, 1),
                                   status=f"{kind}: " + str(e).splitlines()[0][:100])
                    runs.append(row)
                    cfg_rows.append(row)
                    print(f"{tag:52s} r{rep} {row['status'][:38]:38s} {row['wall_s']:7.1f} s"
                          + (f"  C={row['C_raw']:.6f}" if row["status"] == "ok" else ""),
                          flush=True)
                    if rep == 0 and row["status"].startswith("CcxError"):
                        break  # configuration not supported by this build; don't repeat
                ok = [r for r in cfg_rows if r["status"] == "ok"]
                configs.append(dict(build=s.label, mode=cfg_rows[0]["mode"], linear_solver=ls,
                                    threads=t, path=s.path, runs=len(cfg_rows), ok=len(ok),
                                    failures=len(cfg_rows) - len(ok),
                                    median_wall_s=(statistics.median(r["wall_s"] for r in ok)
                                                   if ok else float("nan"))))
    date = time.strftime("%Y-%m-%d %H:%M")
    print("\nconfiguration summary:")
    for c in configs:
        print(f"  {c['build']:42s} {c['linear_solver']:8s} t{c['threads']}  "
              f"{c['ok']}/{c['runs']} ok  median {c['median_wall_s']:.0f} s")
    okruns = [r for r in runs if r["status"] == "ok"]
    out = ROOT / "results" / "benchmarks" / f"benchmark_{host}_{time.strftime('%Y%m%d_%H%M')}.csv"
    write_csv(runs, out)
    write_csv(configs, out.with_name(out.stem + "_summary.csv"))
    print(f"wrote {portable(out)} (+ _summary.csv)")
    if not okruns:
        return
    cs = [r["C_raw"] for r in okruns]
    spread = max(cs) - min(cs)
    print(f"C spread across all completed runs: {spread:.2e} "
          f"({'consistent' if spread < 1e-4 else 'INCONSISTENT - investigate'})")
    reliable = [c for c in configs if c["failures"] == 0 and c["ok"] > 0]
    if not reliable:
        print("no configuration completed every repeat; no profile written")
        return
    best = min(reliable, key=lambda c: c["median_wall_s"])
    print(f"selected (fastest crash-free): {best['build']} ({best['mode']}), "
          f"{best['linear_solver']}, {best['threads']} threads, median {best['median_wall_s']:.0f} s")
    if not a.no_save and spread < 1e-4:
        p = save_profile({"path": best["path"], "threads": best["threads"],
                          "linear_solver": best["linear_solver"]}, configs, date)
        print(f"solver profile written to {portable(p)}")


if __name__ == "__main__":
    main()
