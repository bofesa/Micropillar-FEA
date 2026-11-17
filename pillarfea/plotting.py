"""Publication-style plots (matplotlib). All functions draw into a given Axes/Figure."""
from __future__ import annotations

import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.tri import Triangulation

from .ccx_io import von_mises

# fixed categorical order (CVD-validated): blue, orange, aqua, yellow, magenta, green, violet, red
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

VARIANT_LABEL = {"raw": "raw (punch displacement)",
                 "sneddon": "Sneddon-corrected",
                 "fe_base": "FE foot-displacement corrected"}


def style() -> None:
    mpl.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.8,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "lines.linewidth": 1.6, "legend.frameon": False, "savefig.dpi": 300,
        "savefig.bbox": "tight", "figure.dpi": 110, "axes.spines.top": False,
        "axes.spines.right": False, "xtick.direction": "out", "ytick.direction": "out",
    })


def plot_stress_strain(ax, res: dict, variant: str = "raw", show_1d: bool = True,
                       show_law: bool = True) -> None:
    """Nominal curve, ideal-geometry reference, input law, offset/bilinear construction."""
    c, an = res["curves"], res["config"]["analysis"]
    eps = c["eps"][variant] * 100
    ax.plot(c["eps_ref"] * 100, c["sigma_ref"], color=INK2, lw=1.2, ls="--",
            label="ideal geometry (homogeneous FE)")
    if show_law:
        ax.plot(c["eps_law"] * 100, c["sigma_law"], color=SERIES[5], lw=1.0, ls=":",
                label="input law (small strain)")
    if show_1d and len(c["eps_1d"]):
        ax.plot(c["eps_1d"] * 100, c["sigma_1d"], color=SERIES[2], lw=1.2, label="1D tapered model")
    ax.plot(eps, c["sigma"], color=SERIES[0], label=f"FE pillar, {VARIANT_LABEL.get(variant, variant)}")
    E, b = res[f"E_slope_{variant}"], res[f"intercept_{variant}"]
    sy, ey = res[f"sigma_app_{variant}"], res[f"eps_y_{variant}"]
    if an["method"] == "offset" and np.isfinite(sy):
        off = an["offset_strain"]
        s_line = np.linspace(0, sy * 1.15, 2)
        ax.plot(((s_line - b) / E + off) * 100, s_line, color=SERIES[1], lw=1.0,
                label=f"{off * 100:g} % offset line")
    elif an["method"] == "bilinear" and np.isfinite(sy):
        ax.plot([0, ey * 100], [0, sy], color=SERIES[1], lw=1.0, label="bilinear fit")
    if np.isfinite(sy):
        ax.plot([ey * 100], [sy], "o", ms=6, mfc=SERIES[1], mec="white", mew=1.2, zorder=5)
        ax.annotate(f"{sy:.0f} MPa (C = {res[f'C_{variant}']:.3f})", (ey * 100, sy),
                    xytext=(8, -14), textcoords="offset points", color=INK, fontsize=8)
    ax.axhline(res["sigma_ref"], color=INK2, lw=0.6, ls=(0, (1, 2)))
    ax.set_xlabel("nominal strain, $\\delta/h$ (%)")
    ax.set_ylabel("nominal stress, $F/A_{top}$ (MPa)")
    ax.set_xlim(0, float(eps.max()))
    ax.set_ylim(0, None)
    ax.legend(loc="lower right")


def plot_field(ax, mesh, frames: dict, k: int, field: str = "PEEQ", deformed: float = 1.0,
               zoom: float | None = 1.6, vmax: float | None = None, cbar_ax=None,
               elset: str | None = None, clip_pct: float = 99.0):
    """Contour of 'PEEQ' or 'mises' at frame k, mirrored about the axis.

    The colour scale is clipped at the `clip_pct` percentile of the pillar's nodal values
    (colour bar extended), because the sharp foot corner / punch edge are singular and would
    otherwise saturate the scale. Use clip_pct=100 for the true maximum.
    """
    U = frames["U"][k].astype(float)
    xy = mesh.nodes + deformed * np.nan_to_num(U)
    if field == "PEEQ":
        v = np.nan_to_num(frames["PEEQ"][k].astype(float))
        label = "PEEQ"
    else:
        v = von_mises(frames["S"][k].astype(float))
        label = "von Mises (MPa)"
    if elset is None:   # everything except the (contact) punch block
        ids = np.concatenate([v for k, v in mesh.elsets.items() if k != "PUNCH"])
        els = mesh.elements[ids]
    else:
        els = mesh.elements[mesh.elsets[elset]]
    tri = np.vstack([els[:, [0, 4, 7]], els[:, [4, 1, 5]], els[:, [5, 2, 6]],
                     els[:, [6, 3, 7]], els[:, [4, 5, 7]], els[:, [5, 6, 7]]])
    pn = np.unique(mesh.elements[mesh.elsets["PILLAR"]])
    vtrue = float(np.nanmax(v[np.unique(tri)]))
    fixed = vmax is not None
    if vmax is None:
        vmax = max(float(np.nanpercentile(v[pn], clip_pct)), 1e-12)
    extend = "max" if vtrue > vmax * 1.0001 else "neither"
    for sgn in (1, -1):
        T = Triangulation(sgn * xy[:, 0], xy[:, 1], tri)
        m = ax.tripcolor(T, v, shading="gouraud", cmap=SEQ, vmin=0, vmax=vmax,
                         rasterized=True)
    ax.set_aspect("equal")
    ax.grid(False)
    if zoom:
        _zoom_window(ax, mesh, zoom, symmetric=True)
    ax.set_xlabel("r (µm)")
    ax.set_ylabel("z (µm)")
    cb = ax.figure.colorbar(m, ax=ax if cbar_ax is None else None, cax=cbar_ax, shrink=0.8,
                            extend=extend)
    cb.set_label(label)
    if extend == "max" or fixed:
        note = (f"fixed scale, vmax = {vmax:.3g}" if fixed
                else f"scale clipped at {clip_pct:g}th pct.") + f"\nmax {vtrue:.3g}"
        ax.text(0.02, 0.98, note,
                transform=ax.transAxes, va="top", ha="left", fontsize=7, color=INK2,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    cb.outline.set_visible(False)
    return m


def plot_mesh(ax, mesh, zoom: float | None = 1.6) -> None:
    for e in mesh.elements:
        p = mesh.nodes[e[[0, 4, 1, 5, 2, 6, 3, 7, 0]]]
        ax.plot(p[:, 0], p[:, 1], color=INK2, lw=0.3)
    ax.set_aspect("equal")
    ax.grid(False)
    if zoom:
        _zoom_window(ax, mesh, zoom, symmetric=False)
    ax.set_xlabel("r (µm)")
    ax.set_ylabel("z (µm)")


def _zoom_window(ax, mesh, zoom: float, symmetric: bool) -> None:
    """View window that scales in both directions: zoom = half-width / foot radius.
    The substrate depth shown grows with the zoom (1.6 reproduces the original framing:
    0.25 h of substrate); the top stays at 1.05 h. Clipped to the model extent."""
    h = mesh.nodes[:, 1].max()
    rb = mesh.nodes[mesh.nsets["PBASE"], 0].max()
    rmax, zmin = mesh.nodes[:, 0].max(), mesh.nodes[:, 1].min()
    w = min(zoom * rb, rmax)
    depth = max(0.25 * h, 0.25 * h + (zoom - 1.6) * rb * 1.2)
    ax.set_xlim(-w if symmetric else 0, w)
    ax.set_ylim(max(-depth, zmin), 1.05 * h)


ANALYTICAL = [  # key, label, linestyle, marker
    ("C_avg_diameter", "avg. diameter $(D_t+D_b)/2$", (0, (4, 2)), "^"),
    ("C_avg_area", "avg. area $(A_t+A_b)/2$", (0, (1, 1.5)), "v"),
    ("C_avg_stress", "avg. stress $(\\sigma_t+\\sigma_b)/2$", (0, (5, 1.5, 1, 1.5)), "D"),
    ("C_avg_stress_integral", "avg. stress $\\langle F/A(z)\\rangle$ ($D_b/D_t$)",
     (0, (6, 1.5, 1, 1.5, 1, 1.5)), "s"),
]
GEOMETRIC_KEYS = ("geometry.h_over_D", "geometry.taper_deg", "geometry.D_top")


def plot_analytical(ax, rows: list[dict], x: str, color: str = INK2, label: bool = True) -> None:
    """Thin lines + small markers for the geometric corrections (depend on geometry only)."""
    pts = {}
    for r in rows:
        if all(k in r and r[k] not in (None, "") for k, *_ in ANALYTICAL):
            pts.setdefault(float(r[x]), r)
    if not pts:
        return
    xs = sorted(pts)
    for key, lab, ls, mk in ANALYTICAL:
        ax.plot(xs, [float(pts[v][key]) for v in xs], ls=ls, marker=mk, ms=2.8, lw=0.8,
                color=color, mfc=color, mec=color, zorder=1,
                label=f"analytical: {lab}" if label else None)


LABELS = {
    "geometry.taper_deg": "taper angle θ (°)", "geometry.h_over_D": "aspect ratio $h/D_{top}$",
    "geometry.fillet_radius": "fillet radius (µm)", "geometry.D_top": "$D_{top}$ (µm)",
    "material.hardening.Et_over_E": "$E_t/E$", "material.sigma_y_over_E": "$\\sigma_y/E$",
    "material.nu": "ν", "contact.mu": "friction coefficient μ", "contact": "contact",
    "C_raw": "$C_{0.2}$", "value": "value",
}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]


def _group_order(g):
    """Physical order for contact labels (frictionless < μ ascending < bonded), else numeric."""
    s = str(g)
    if s == "frictionless":
        return (0, 0.0, s)
    if s.startswith("μ = "):
        return (1, float(s[4:]), s)
    if s == "bonded":
        return (2, 0.0, s)
    try:
        return (1, float(s), s)
    except ValueError:
        return (3, 0.0, s)


def plot_series(ax, rows: list[dict], x: str, y: str, group: str | None = None,
                xlabel: str | None = None, ylabel: str | None = None, marker=None,
                analytical: bool = False) -> None:
    """y vs x from result rows, one line per value of `group` (fixed colour order, distinct
    marker per group so overlapping series stay visible). analytical=True adds the geometric
    corrections (per group in the group colour if the group is a geometric parameter)."""
    groups = sorted({r.get(group) for r in rows}, key=_group_order) if group else [None]
    for i, g in enumerate(groups):
        sel = sorted([r for r in rows if group is None or r.get(group) == g],
                     key=lambda r: float(r[x]))
        xs = [float(r[x]) for r in sel]
        ys = [float(r[y]) for r in sel]
        col = SERIES[i % len(SERIES)]
        mk = marker or MARKERS[i % len(MARKERS)]
        lab = None if g is None else (str(g) if group == "contact"
                                      else f"{LABELS.get(group, group)} = {g}")
        ax.plot(xs, ys, marker=mk, ms=4.5 - 0.3 * i, mec="white", mew=0.6, color=col,
                lw=1.6, label=lab, zorder=3 + i)
        if analytical and group in GEOMETRIC_KEYS:
            plot_analytical(ax, sel, x, color=col, label=(i == 0))
    if analytical and group not in GEOMETRIC_KEYS:
        plot_analytical(ax, rows, x)
    ax.set_xlabel(xlabel if xlabel and xlabel != x else LABELS.get(x, x))
    ax.set_ylabel(ylabel if ylabel and ylabel != y else LABELS.get(y, y))
    if group or analytical:
        ax.legend(fontsize=7)
