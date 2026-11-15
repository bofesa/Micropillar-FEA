"""Run a parameter sweep defined in a YAML file and write a CSV (+ a C-vs-parameter plot).

    python scripts/run_sweep.py configs/sweeps/taper_quick.yaml [--jobs 3] [--force]
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
import yaml  # noqa: E402

from pillarfea import plotting as P  # noqa: E402
from pillarfea.sweep import contact_label, run_sweep_file  # noqa: E402
from pillarfea.platform_info import portable  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    rows, out = run_sweep_file(a.sweep, ROOT, a.force, a.jobs)
    sweep = yaml.safe_load(Path(a.sweep).read_text(encoding="utf-8"))
    grid = list((sweep.get("grid") or {}).keys())
    ok = [r for r in rows if "error" not in r]
    if not grid or not ok:
        return
    P.style()
    x = grid[0]
    group = grid[1] if len(grid) > 1 else None
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    others = grid[2:]

    def val(r, k):  # dict-valued grid axes (e.g. contact) are compared via their label
        return r.get("contact") if k == "contact" else str(r.get(k))

    if others:  # plot only the first value of the remaining grid parameters
        first = {}
        for k in others:
            v0 = sweep["grid"][k][0]
            first[k] = (contact_label({f"contact.{kk}": vv for kk, vv in v0.items()})
                        if isinstance(v0, dict) else str(v0))
        ok = [r for r in ok if all(val(r, k) == v for k, v in first.items())]
        ax.set_title(", ".join(f"{k} = {v}" for k, v in first.items()), fontsize=8)
    P.plot_series(ax, ok, x, "C_raw", group, analytical=True)
    ax.set_title(ax.get_title() or "FE (symbols) vs. analytical corrections (thin lines)",
                 fontsize=8)
    fig.savefig(out.with_suffix(".png"))
    print(f"plot: {portable(out.with_suffix('.png'))}")


if __name__ == "__main__":
    main()
