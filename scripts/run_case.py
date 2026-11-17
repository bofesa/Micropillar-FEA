"""Run a single case and write its figures/CSV.

    python scripts/run_case.py configs/example_case.yaml [--set geometry.taper_deg=3 ...] [--force]
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
from pillarfea.config import load_config  # noqa: E402
from pillarfea.simulate import load_frames, run_case  # noqa: E402
from pillarfea.sweep import result_row, write_csv  # noqa: E402
from pillarfea.platform_info import awake, portable  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-sleep", action="store_true", help="let the computer sleep during the run (default: sleep is blocked)")
    ap.add_argument("config", nargs="?")
    ap.add_argument("--set", nargs="*", default=[], help="dotted.key=value overrides")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ov = {}
    for s in a.set:
        k, v = s.split("=", 1)
        ov[k] = yaml.safe_load(v)
    res = run_case(load_config(a.config, ov), force=a.force)
    name = Path(a.config).stem if a.config else "default"
    out = Path(a.out) if a.out else ROOT / "results" / "cases" / name
    out.mkdir(parents=True, exist_ok=True)
    P.style()
    for v in res["curves"]["eps"]:
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        P.plot_stress_strain(ax, res, v)
        fig.savefig(out / f"stress_strain_{v}.png"); plt.close(fig)
    mesh, fr = load_frames(res["workdir"])
    fig, axs = plt.subplots(1, 2, figsize=(8, 4.0), layout="constrained")
    P.plot_field(axs[0], mesh, fr, len(fr["time"]) - 1, "PEEQ")
    P.plot_field(axs[1], mesh, fr, len(fr["time"]) - 1, "mises")
    fig.savefig(out / "fields_end.png"); plt.close(fig)
    write_csv([result_row(res)], out / "result.csv")
    for k, v in result_row(res).items():
        print(f"{k:32s} {portable(v) if isinstance(v, str) else v}")
    print(f"\nwritten to {portable(out)}")


if __name__ == "__main__":
    with awake("--allow-sleep" not in sys.argv):
        main()
