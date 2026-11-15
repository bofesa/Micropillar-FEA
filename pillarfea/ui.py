"""Desktop UI (Tk + matplotlib):  python -m pillarfea.ui [config.yaml]

Left: parameters. Right: tabs for the stress-strain curve, contour fields, mesh, a summary
table and one-parameter scans. FE results are cached, so changing only the analysis
settings (method, offset, slope) re-evaluates instantly.
"""
from __future__ import annotations

import copy
import queue
import sys
import threading
import time
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from . import plotting as P  # noqa: E402
from .config import get_dotted, load_config, resolve, save_config, set_dotted  # noqa: E402
from .geometry import build_mesh, geometry_from_cfg  # noqa: E402
from .platform_info import portable  # noqa: E402
from .simulate import load_frames, run_case  # noqa: E402
from .sweep import result_row, run_cases, write_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# (dotted key, label, kind) ; kind: float | int | auto (float or 'auto') | choice list | pct
FORM = [
    ("Geometry", [
        ("geometry.D_top", "Top diameter D_top (µm)", "float"),
        ("geometry.h_over_D", "Aspect ratio h / D_top", "float"),
        ("geometry.taper_deg", "Taper angle (°, sidewall)", "float"),
        ("geometry.fillet_radius", "Foot fillet radius (µm)", "float"),
    ]),
    ("Material (pillar)", [
        ("material.E", "Young's modulus E (MPa)", "float"),
        ("material.nu", "Poisson's ratio", "float"),
        ("material.sigma_y_over_E", "Yield strain σy / E", "float"),
        ("material.hardening.Et_over_E", "Tangent modulus Et / E", "float"),
    ]),
    ("Substrate", [
        ("substrate.type", "Substrate", ["elastic", "elastoplastic", "none"]),
        ("substrate.E", "E_sub (MPa, blank = pillar)", "optfloat"),
        ("substrate.nu", "ν_sub (blank = pillar)", "optfloat"),
        ("substrate.radius_over_D", "Domain radius / D_top", "float"),
    ]),
    ("Top contact", [
        ("contact.top", "Punch contact", ["frictionless", "bonded", "coulomb"]),
        ("contact.mu", "Friction coefficient μ", "float"),
    ]),
    ("Loading & mesh", [
        ("loading.max_strain", "Max nominal strain (auto)", "auto"),
        ("loading.n_increments", "Load increments", "int"),
        ("mesh.n_radial", "Elements across radius", "int"),
        ("solver.threads", "Solver threads (auto)", "autoint"),
    ]),
    ("Yield extraction", [
        ("analysis.method", "Method", ["offset", "bilinear"]),
        ("analysis.offset_strain", "Offset strain (%)", "pct"),
        ("analysis.slope", "Elastic slope", ["fit", "material"]),
        ("analysis.bilinear_max_strain", "Bilinear fit up to strain (auto)", "auto"),
    ]),
]

TIPS = {
    "geometry.D_top": "Top diameter of the pillar in µm (> 0).\nResults (C) are scale-invariant; only "
                      "forces scale with D².\nTypical: 0.5–5 µm.",
    "geometry.h_over_D": "Pillar height / top diameter (> 0).\nStudy range 1.5–3; < 1 or > 5 is outside "
                         "what has been validated.",
    "geometry.taper_deg": "Sidewall angle from the pillar axis (half-angle of the cone), degrees.\n"
                          "Valid: 0 ≤ θ < 45. D_bot = D_top + 2·h·tan θ.\nTypical FIB pillars: 0–5°.",
    "geometry.fillet_radius": "Foot fillet radius in µm (≥ 0), tangent to sidewall and substrate.\n"
                              "0 = sharp re-entrant corner. Default 0.1 µm (typical after FIB).\n"
                              "Must fit in the lower half of the pillar.",
    "material.E": "Young's modulus of the pillar, MPa (> 0).",
    "material.nu": "Poisson's ratio, 0 ≤ ν < 0.5.",
    "material.sigma_y_over_E": "Initial yield stress / E (yield strain), > 0.\nStudy range 0.005–0.02.",
    "material.hardening.Et_over_E": "Linear hardening: tangent modulus of the uniaxial stress–strain "
                                    "curve / E.\nValid: 0 ≤ Et/E < 1 (0 = perfectly plastic).\n"
                                    "Study range 0.02–0.10. Tabulated hardening: use a config file.",
    "substrate.type": "none: rigid frictionless base (pillar foot on rollers)\n"
                      "elastic: linear elastic hemisphere\n"
                      "elastoplastic: same flow curve as the pillar (pillar milled into bulk)",
    "substrate.E": "Substrate Young's modulus, MPa. Blank = same as pillar.",
    "substrate.nu": "Substrate Poisson's ratio, 0 ≤ ν < 0.5. Blank = same as pillar.",
    "substrate.radius_over_D": "Radius of the hemispherical substrate domain / D_top (outer surface fixed).\n"
                               "Must be ≥ 5 × foot radius; C is insensitive above ~20 (validated 20–80).",
    "contact.top": "frictionless: punch prescribes axial displacement, radial sliding free (exact)\n"
                   "bonded: no slip, μ → ∞ limit (exact)\n"
                   "coulomb: penalty contact with a stiff punch block and Coulomb friction μ\n"
                   "(slower, ~2×; penalty settings in the config file)",
    "contact.mu": "Coulomb friction coefficient, μ ≥ 0 (typical diamond–metal 0.05–0.3).\n"
                  "Only used with contact = coulomb; μ = 0 gives frictionless contact.\n"
                  "Ignored for frictionless (μ = 0) and bonded (μ → ∞).",
    "loading.max_strain": "Final nominal strain δ/h. 'auto' = 2.5·σy/E + 0.015.\n"
                          "Must extend past the offset/bilinear yield point.",
    "loading.n_increments": "Number of equal displacement increments (≥ 20).\n"
                            "100 is converged to ~3e-5 in C; 200 is the default.",
    "mesh.n_radial": "Elements across the top radius (≥ 4).\nC converged to ~1e-4 at 16 (default); "
                     "cost grows ~quadratically.",
    "solver.threads": "Threads per CalculiX run. 'auto' = benchmark profile (else 4).\n"
                      "On a 12-core machine 4 is fastest; 8 is slower.",
    "analysis.method": "offset: intersection with the elastic line shifted by the offset strain\n"
                       "bilinear: two-segment elastic + linear-hardening fit; yield = break point",
    "analysis.offset_strain": "Offset in percent (> 0), e.g. 0.2 or 0.5.\nOnly used by the offset method.",
    "analysis.slope": "fit: slope fitted to the loading curve (10–50 % of σy), as in experiments\n"
                      "material: input E (then substrate compliance matters)",
    "analysis.bilinear_max_strain": "Upper strain limit of the bilinear fit ('auto' = 2·σy/E + 0.01).\n"
                                    "Result depends on this range when yielding is gradual.",
}


class Tooltip:
    """Minimal hover tooltip for Tk widgets."""

    def __init__(self, widgets, text, delay=400):
        self.text, self.delay, self.tip, self.job = text, delay, None, None
        for w in widgets:
            w.bind("<Enter>", self._schedule, add="+")
            w.bind("<Leave>", self._hide, add="+")
            w.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, e):
        self._hide()
        self.job = e.widget.after(self.delay, lambda: self._show(e.widget))

    def _show(self, w):
        x, y = w.winfo_rootx() + 20, w.winfo_rooty() + w.winfo_height() + 4
        self.tip = tk.Toplevel(w)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left", background="#fcfcfb",
                 foreground=P.INK, relief="solid", borderwidth=1, padx=6, pady=4,
                 font=("Segoe UI", 9)).pack()

    def _hide(self, e=None):
        if self.job:
            try:
                (e.widget if e else self.tip).after_cancel(self.job)
            except Exception:
                pass
            self.job = None
        if self.tip:
            self.tip.destroy()
            self.tip = None


SCAN_KEYS = ["geometry.taper_deg", "geometry.h_over_D", "geometry.fillet_radius", "contact.mu",
             "material.hardening.Et_over_E", "material.sigma_y_over_E", "material.nu",
             "contact.top", "substrate.type", "substrate.E", "mesh.n_radial",
             "loading.n_increments", "substrate.radius_over_D"]


class App(tk.Tk):
    def __init__(self, cfg_path: str | None = None):
        super().__init__()
        self.title("pillarfea — micropillar yield-stress correction")
        self.geometry("1400x860")
        P.style()
        self.cfg = load_config(cfg_path)
        self.vars: dict[str, tk.Variable] = {}
        self.result = None
        self.frames = None
        self.mesh = None
        self.q: queue.Queue = queue.Queue()
        self.busy = False
        self._build()
        self._load_into_form(self.cfg)
        self.after(100, self._poll)

    # ------------------------------------------------------------------ layout
    def _build(self):
        left = ttk.Frame(self, padding=6)
        left.pack(side="left", fill="y")
        for title, fields in FORM:
            lf = ttk.LabelFrame(left, text=title, padding=4)
            lf.pack(fill="x", pady=3)
            for r, (key, label, kind) in enumerate(fields):
                lab = ttk.Label(lf, text=label)
                lab.grid(row=r, column=0, sticky="w")
                v = tk.StringVar()
                self.vars[key] = v
                if isinstance(kind, list):
                    w = ttk.Combobox(lf, textvariable=v, values=kind, width=12, state="readonly")
                else:
                    w = ttk.Entry(lf, textvariable=v, width=14)
                w.grid(row=r, column=1, sticky="e", padx=4, pady=1)
                if key in TIPS:
                    Tooltip([lab, w], TIPS[key])
            lf.columnconfigure(0, weight=1)
        bf = ttk.Frame(left)
        bf.pack(fill="x", pady=6)
        self.run_btn = ttk.Button(bf, text="Run / update", command=self.on_run)
        self.run_btn.grid(row=0, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(bf, text="Load config…", command=self.on_load).grid(row=1, column=0, sticky="ew")
        ttk.Button(bf, text="Save config…", command=self.on_save).grid(row=1, column=1, sticky="ew")
        ttk.Button(bf, text="Export figures + CSV", command=self.on_export).grid(row=2, column=0, columnspan=2, sticky="ew", pady=2)
        bf.columnconfigure((0, 1), weight=1)
        self.force = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="force re-run (ignore cache)", variable=self.force).pack(anchor="w")
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(left, textvariable=self.status, wraplength=300, foreground=P.INK2).pack(anchor="w", pady=4)
        self.pbar = ttk.Progressbar(left, mode="indeterminate")
        self.pbar.pack(fill="x")

        nb = ttk.Notebook(self)
        nb.pack(side="right", fill="both", expand=True)
        self.nb = nb
        # curve tab
        t = ttk.Frame(nb); nb.add(t, text="Stress–strain")
        ctl = ttk.Frame(t); ctl.pack(fill="x")
        ttk.Label(ctl, text="strain measure:").pack(side="left")
        self.variant = tk.StringVar(value="raw")
        self.variant_cb = ttk.Combobox(ctl, textvariable=self.variant, values=["raw"], width=10, state="readonly")
        self.variant_cb.pack(side="left", padx=4)
        self.variant_cb.bind("<<ComboboxSelected>>", lambda e: self.draw_curve())
        self.show1d = tk.BooleanVar(value=True)
        self.showlaw = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctl, text="1D model", variable=self.show1d, command=self.draw_curve).pack(side="left", padx=6)
        ttk.Checkbutton(ctl, text="input law", variable=self.showlaw, command=self.draw_curve).pack(side="left")
        self.fig_curve, self.cv_curve = self._canvas(t, (8, 6))
        # fields tab
        t = ttk.Frame(nb); nb.add(t, text="Fields")
        ctl = ttk.Frame(t); ctl.pack(fill="x")
        self.field = tk.StringVar(value="PEEQ")
        ttk.Combobox(ctl, textvariable=self.field, values=["PEEQ", "mises"], width=8, state="readonly").pack(side="left")
        ttk.Label(ctl, text=" frame:").pack(side="left")
        self.frame_k = tk.IntVar(value=0)
        self.frame_scale = ttk.Scale(ctl, from_=0, to=1, orient="horizontal", length=300,
                                     command=lambda v: self._set_frame(int(float(v))))
        self.frame_scale.pack(side="left", padx=4)
        ttk.Button(ctl, text="at offset yield", command=self._frame_at_yield).pack(side="left")
        ttk.Label(ctl, text=" deform ×").pack(side="left")
        self.defscale = tk.StringVar(value="1")
        ttk.Entry(ctl, textvariable=self.defscale, width=5).pack(side="left")
        self.zoom = tk.StringVar(value="1.6")
        ttk.Label(ctl, text=" zoom (× foot radius, 0 = all)").pack(side="left")
        ttk.Entry(ctl, textvariable=self.zoom, width=5).pack(side="left")
        ttk.Label(ctl, text=" colour clip pct").pack(side="left")
        self.clip = tk.StringVar(value="99")
        ttk.Entry(ctl, textvariable=self.clip, width=5).pack(side="left")
        ttk.Button(ctl, text="redraw", command=self.draw_fields).pack(side="left", padx=4)
        self.field.trace_add("write", lambda *a: self.draw_fields())
        self.fig_field, self.cv_field = self._canvas(t, (8, 6))
        # mesh tab
        t = ttk.Frame(nb); nb.add(t, text="Mesh")
        ttk.Button(t, text="preview mesh for current parameters", command=self.draw_mesh).pack(anchor="w")
        self.fig_mesh, self.cv_mesh = self._canvas(t, (8, 6))
        # summary tab
        t = ttk.Frame(nb); nb.add(t, text="Summary")
        self.summary = tk.Text(t, font=("Consolas", 10), wrap="none")
        self.summary.pack(fill="both", expand=True)
        # scan tab
        t = ttk.Frame(nb); nb.add(t, text="Parameter scan")
        ctl = ttk.Frame(t); ctl.pack(fill="x")
        ttk.Label(ctl, text="parameter").pack(side="left")
        self.scan_key = tk.StringVar(value="geometry.taper_deg")
        ttk.Combobox(ctl, textvariable=self.scan_key, values=SCAN_KEYS, width=28).pack(side="left", padx=4)
        ttk.Label(ctl, text="values").pack(side="left")
        self.scan_vals = tk.StringVar(value="0, 1, 2, 3, 5")
        ttk.Entry(ctl, textvariable=self.scan_vals, width=30).pack(side="left", padx=4)
        ttk.Label(ctl, text="parallel jobs").pack(side="left")
        self.scan_jobs = tk.StringVar(value="3")
        ttk.Entry(ctl, textvariable=self.scan_jobs, width=4).pack(side="left", padx=4)
        ttk.Button(ctl, text="Run scan", command=self.on_scan).pack(side="left", padx=6)
        self.fig_scan, self.cv_scan = self._canvas(t, (8, 6))
        self.scan_rows = None

    def _canvas(self, parent, size):
        fig = Figure(figsize=size, layout="constrained")
        cv = FigureCanvasTkAgg(fig, master=parent)
        NavigationToolbar2Tk(cv, parent).update()
        cv.get_tk_widget().pack(fill="both", expand=True)
        return fig, cv

    # ------------------------------------------------------------ form <-> cfg
    def _load_into_form(self, cfg):
        for title, fields in FORM:
            for key, _, kind in fields:
                v = get_dotted(cfg, key)
                if kind == "pct":
                    v = f"{float(v) * 100:g}"
                elif v is None:
                    v = ""
                self.vars[key].set(str(v))

    def _form_to_cfg(self) -> dict:
        cfg = copy.deepcopy(self.cfg)
        for title, fields in FORM:
            for key, label, kind in fields:
                s = self.vars[key].get().strip()
                try:
                    if isinstance(kind, list):
                        v = s
                    elif kind == "int":
                        v = int(float(s))
                    elif kind == "float":
                        v = float(s)
                    elif kind == "pct":
                        v = float(s) / 100.0
                    elif kind == "optfloat":
                        v = None if s in ("", "none", "None") else float(s)
                    elif kind == "auto":
                        v = "auto" if s in ("", "auto") else float(s)
                    elif kind == "autoint":
                        v = "auto" if s in ("", "auto") else int(float(s))
                except ValueError:
                    raise ValueError(f"Invalid value for '{label}': {s!r}")
                set_dotted(cfg, key, v)
        cfg["material"]["hardening"]["type"] = cfg["material"]["hardening"].get("type", "linear")
        return cfg

    # ------------------------------------------------------------------ actions
    def on_load(self):
        p = filedialog.askopenfilename(initialdir=ROOT / "configs", filetypes=[("YAML", "*.yaml *.yml")])
        if p:
            self.cfg = load_config(p)
            self._load_into_form(self.cfg)
            self.status.set(f"Loaded {p}")

    def on_save(self):
        p = filedialog.asksaveasfilename(initialdir=ROOT / "configs", defaultextension=".yaml",
                                         filetypes=[("YAML", "*.yaml")])
        if p:
            save_config(self._form_to_cfg(), p)
            self.status.set(f"Saved {p}")

    def _start(self, label, fn):
        if self.busy:
            return
        self.busy = True
        self.run_btn.state(["disabled"])
        self.pbar.start(12)
        self.t0 = time.time()
        self.status.set(label)

        def work():
            try:
                self.q.put(("ok", fn()))
            except Exception as e:
                self.q.put(("err", (e, traceback.format_exc())))
        threading.Thread(target=work, daemon=True).start()

    def on_run(self):
        try:
            cfg = self._form_to_cfg()
            resolve(cfg)
        except Exception as e:
            messagebox.showerror("Invalid parameters", str(e))
            return
        self.cfg = cfg
        force = self.force.get()
        self._start("Running CalculiX (cached cases return immediately)…",
                    lambda: ("case", run_case(cfg, force=force)))

    def on_scan(self):
        try:
            cfg = self._form_to_cfg()
            key = self.scan_key.get().strip()
            raw = [s.strip() for s in self.scan_vals.get().split(",") if s.strip()]
            vals = []
            for s in raw:
                try:
                    vals.append(float(s))
                except ValueError:
                    vals.append(s)
            cfgs = []
            for v in vals:
                c = copy.deepcopy(cfg)
                set_dotted(c, key, int(v) if key.startswith(("mesh.", "loading.n_")) else v)
                resolve(c)
                cfgs.append(c)
            jobs = int(self.scan_jobs.get())
        except Exception as e:
            messagebox.showerror("Invalid scan", str(e))
            return
        self.cfg = cfg

        def fn():
            def prog(k, n, i, r):
                self.q.put(("status", f"scan: {k}/{n} cases finished"))
            res = run_cases(cfgs, jobs, 4 if jobs <= 2 else 3, prog)
            return ("scan", (key, vals, res))
        self._start(f"Scanning {key} over {len(vals)} values…", fn)

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "status":
                    self.status.set(payload)
                    continue
                self.busy = False
                self.run_btn.state(["!disabled"])
                self.pbar.stop()
                if kind == "err":
                    e, tb = payload
                    self.status.set(f"Error: {e}")
                    messagebox.showerror("Error", f"{e}\n\n{tb[-1500:]}")
                elif payload[0] == "case":
                    self._show_case(payload[1])
                elif payload[0] == "scan":
                    self._show_scan(*payload[1])
        except queue.Empty:
            pass
        self.after(150, self._poll)

    # ------------------------------------------------------------------ display
    def _show_case(self, res):
        self.result = res
        v = list(res["curves"]["eps"])
        self.variant_cb["values"] = v
        if self.variant.get() not in v:
            self.variant.set(v[0])
        self.mesh, self.frames = load_frames(res["workdir"])
        self.frame_scale.configure(to=len(self.frames["time"]) - 1)
        self.draw_curve()
        self._frame_at_yield()
        self.draw_mesh(res["config"])
        self._write_summary(res)
        self.status.set(f"Done in {time.time() - self.t0:.1f} s (FE solve {res['runtime_s']:.0f} s; "
                        f"cached runs show the original solve time).")

    def draw_curve(self):
        if not self.result:
            return
        self.fig_curve.clear()
        ax = self.fig_curve.add_subplot(111)
        P.plot_stress_strain(ax, self.result, self.variant.get(), self.show1d.get(), self.showlaw.get())
        g = self.result["config"]["geometry"]
        ax.set_title(f"θ = {g['taper_deg']:g}°, h/D = {g['h_over_D']:g}, "
                     f"{self.result['config']['contact']['top']} top, "
                     f"{self.result['config']['substrate']['type']} substrate")
        self.cv_curve.draw_idle()

    def _set_frame(self, k):
        if self.frames is None:
            return
        if k != self.frame_k.get():
            self.frame_k.set(k)
            self.draw_fields()

    def _frame_at_yield(self):
        if not self.result or self.frames is None:
            return
        r, var = self.result, self.variant.get()
        eps = r["curves"]["eps"][var]
        t = np.interp(r[f"eps_y_{var}"], eps, r["curves"]["time"]) \
            if np.isfinite(r[f"eps_y_{var}"]) else 1.0
        k = int(np.argmin(np.abs(self.frames["time"] - t)))
        self.frame_k.set(k)
        self.frame_scale.set(k)
        self.draw_fields()

    def draw_fields(self):
        if self.frames is None:
            return
        k = self.frame_k.get()
        self.fig_field.clear()
        ax = self.fig_field.add_subplot(111)
        try:
            ds, zm, cp = float(self.defscale.get()), float(self.zoom.get()), float(self.clip.get())
        except ValueError:
            ds, zm, cp = 1.0, 1.6, 99.0
        P.plot_field(ax, self.mesh, self.frames, k, self.field.get(), deformed=ds,
                     zoom=zm or None, clip_pct=cp)
        t = self.frames["time"][k]
        eps = t * self.result["config"]["loading"]["max_strain"]
        ax.set_title(f"{self.field.get()} at nominal strain {eps * 100:.2f} % "
                     f"(frame {k + 1}/{len(self.frames['time'])})")
        self.cv_field.draw_idle()

    def draw_mesh(self, cfg=None):
        try:
            rc = resolve(cfg or self._form_to_cfg())
            mesh = build_mesh(geometry_from_cfg(rc), rc["mesh"])
        except Exception as e:
            messagebox.showerror("Mesh", str(e))
            return
        self.fig_mesh.clear()
        a1, a2 = self.fig_mesh.subplots(1, 2)
        P.plot_mesh(a1, mesh, zoom=1.6)
        P.plot_mesh(a2, mesh, zoom=None)
        a1.set_title(f"{len(mesh.elements)} CAX8R elements, {len(mesh.nodes)} nodes")
        a2.set_title("whole domain")
        self.cv_mesh.draw_idle()

    def _write_summary(self, r):
        L = []
        rc = r["config"]
        an = rc["analysis"]
        meth = (f"{an['offset_strain'] * 100:g} % offset" if an["method"] == "offset"
                else f"bilinear fit to ε ≤ {an['bilinear_max_strain']:g}") + f", slope = {an['slope']}"
        L.append(f"Yield extraction: {meth}")
        L.append(f"Input σy                          {r['sigma_y_input']:10.1f} MPa")
        if "sigma_offset_input_law" in r:
            L.append(f"Input law σ at εp = offset        {r['sigma_offset_input_law']:10.1f} MPa (true stress)")
        L.append(f"Ideal-geometry σ (reference)      {r['sigma_ref']:10.1f} MPa (homogeneous uniaxial FE, same method)")
        L.append("")
        L.append(f"{'strain measure':34s} {'σ_app (MPa)':>12s} {'C':>8s} {'overest.':>9s} {'E_slope':>9s} {'D_eff/D_top':>11s} {'z_eff/h':>8s}")
        for v in r["curves"]["eps"]:
            sa, C = r[f"sigma_app_{v}"], r[f"C_{v}"]
            L.append(f"{P.VARIANT_LABEL.get(v, v):34s} {sa:12.1f} {C:8.4f} {(C - 1) * 100:8.2f}% "
                     f"{r[f'E_slope_{v}']:9.0f} {r[f'Deff_over_Dtop_{v}']:11.4f} {r[f'zeff_from_top_over_h_{v}']:8.3f}")
        L.append(f"{'1D progressive-yield model':34s} {r['sigma_app_1d']:12.1f} {r['C_1d']:8.4f} "
                 f"{(r['C_1d'] - 1) * 100:8.2f}%   (vs its own small-strain reference {r['sigma_ref_1d']:.1f})")
        for k, lab in (("C_avg_diameter", "analytical: average diameter"),
                       ("C_avg_area", "analytical: mean of top/bottom area"),
                       ("C_avg_stress", "analytical: average stress (σ_t+σ_b)/2"),
                       ("C_avg_stress_integral", "analytical: height-avg. stress")):
            if k in r:
                L.append(f"{lab:34s} {'':12s} {r[k]:8.4f} {(r[k] - 1) * 100:8.2f}%")
        L.append("")
        L.append(f"Foot diameter D_bot = {r['D_bot']:.4f} µm;  force-equilibrium error = {r['equilibrium_err']:.1e}")
        lo = r.get("top_min_nodal_force_ratio", np.nan)
        flag = "n/a (cached run)" if not np.isfinite(lo) else \
            ("OK, whole top face in compression" if lo > -1e-3 else "WARNING: tensile nodal force on top face, contact would open")
        L.append(f"Top-face lift-off check: min nodal force / F = {lo:.2e}  -> {flag}")
        L.append(f"z_eff/h: depth below the top where the pillar diameter equals D_eff")
        L.append(f"FE solve time {r['runtime_s']:.0f} s ({r.get('solver_attempts', 1)} attempt(s)), "
                 f"solver {r.get('solver_executable') or '(not recorded)'}")
        L.append(f"Work dir {portable(r['workdir'])}")
        self.summary.delete("1.0", "end")
        self.summary.insert("1.0", "\n".join(L))

    def _show_scan(self, key, vals, res):
        rows = []
        for v, r in zip(vals, res):
            if isinstance(r, Exception):
                rows.append({"value": v, "error": str(r)})
            else:
                rows.append(dict(value=v, **result_row(r)))
        self.scan_rows = rows
        self.fig_scan.clear()
        ax = self.fig_scan.add_subplot(111)
        ok = [r for r in rows if "error" not in r]
        numeric = all(isinstance(r["value"], float) for r in ok)
        x = [r["value"] if numeric else i for i, r in enumerate(ok)]
        variants = [v for v in ("raw", "sneddon", "fe_base") if ok and f"C_{v}" in ok[0]]
        for i, v in enumerate(variants):
            ax.plot(x, [r[f"C_{v}"] for r in ok], "o-", color=P.SERIES[i], ms=4, mec="white",
                    label=P.VARIANT_LABEL[v])
        ax.plot(x, [r["C_1d"] for r in ok], "s--", color=P.INK2, ms=3.2, lw=1, label="1D model")
        if numeric:
            P.plot_analytical(ax, ok, "value")
        else:
            ax.set_xticks(x, [str(r["value"]) for r in ok])
        ax.set_xlabel(key)
        ax.set_ylabel("$C = \\sigma_{apparent} / \\sigma_{ideal}$")
        ax.legend(fontsize=7)
        self.cv_scan.draw_idle()
        out = ROOT / "results" / "ui_scans" / f"scan_{key}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        write_csv(rows, out)
        bad = [r for r in rows if "error" in r]
        self.status.set(f"Scan done ({len(bad)} failed). CSV: {out}")

    def on_export(self):
        if not self.result:
            return
        d = ROOT / "results" / "ui_exports" / time.strftime("%Y%m%d_%H%M%S")
        d.mkdir(parents=True, exist_ok=True)
        self.fig_curve.savefig(d / "stress_strain.png")
        self.fig_curve.savefig(d / "stress_strain.pdf")
        self.fig_field.savefig(d / "field.png")
        save_config(self.result["config"], d / "config.yaml")
        write_csv([result_row(self.result)], d / "result.csv")
        c = self.result["curves"]
        cols = {"F_uN": c["F"], "delta_um": c["delta"], "u_base_um": c["u_base"], "sigma_nom_MPa": c["sigma"]}
        cols.update({f"eps_{k}": v for k, v in c["eps"].items()})
        np.savetxt(d / "curve.csv", np.column_stack(list(cols.values())), delimiter=",",
                   header=",".join(cols), comments="")
        self.status.set(f"Exported to {d}")


def main():
    App(sys.argv[1] if len(sys.argv) > 1 else None).mainloop()


if __name__ == "__main__":
    main()
