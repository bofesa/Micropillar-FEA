# pillarfea — yield-stress corrections for tapered micropillar compression

A small Python framework around the open-source FE solver **CalculiX** that quantifies how
pillar geometry, end constraint, substrate and material parameters bias the yield stress
*measured* in a micropillar compression test, relative to the yield stress an ideal
uniaxial test on the same material would give.

The central output is the correction factor

$$C = \frac{\sigma_{y,\mathrm{apparent}}}{\sigma_{y,\mathrm{ideal}}}$$

where both stresses are extracted from nominal (engineering) stress–strain curves with
**the same method** (0.2 % offset by default, any offset, or a bilinear elastic–plastic fit).
`C > 1` means the pillar test over-estimates the yield stress by (C − 1); the true yield stress is σ_apparent / C.

> Status (stage 2): frictionless, Coulomb-frictional and fully bonded punch contact; foot
> fillet; rigid / elastic / elastic-plastic substrate; linear or tabulated hardening;
> validation suite, UI, sweeps. Function fitting of C is deliberately left to the user
> (all inputs and outputs are in the sweep CSVs).

---

## 1. Installation

Tested on Windows 11 ARM64 (Snapdragon X Elite) with native ARM64 Python 3.12; nothing is
ARM-specific, so the same steps work on x64 Windows.

1. **Python environment** (kept outside OneDrive so it does not sync):
   ```powershell
   python -m venv $env:USERPROFILE\.venvs\pillarfea
   & $env:USERPROFILE\.venvs\pillarfea\Scripts\python.exe -m pip install -r requirements.txt
   ```
   `.vscode/settings.json` points VS Code at this interpreter.

2. **CalculiX** — install the build(s) for this machine:
   ```powershell
   python scripts/install_ccx.py           # default build for this architecture
   python scripts/install_ccx.py --all     # also the alternative build (for benchmarking)
   python -m pillarfea.platform_info       # shows host arch, builds found, the selection
   ```
   Default: on x64 the native MSYS2 ucrt64 build; on ARM64 the x64 zip build (emulated),
   because the native ARM64 build crashes intermittently (§6).
   Builds go to `%LOCALAPPDATA%\pillarfea\ccx\` (no spaces in the path, outside OneDrive):
   * `msys2-clangarm64` — native Windows ARM64 (MSYS2 package; only its .exe/.dll files
     are extracted, no MSYS2 installation; package checksums verified);
   * `msys2-ucrt64` — native Windows x64 (installed instead of the above on x64 machines);
   * `CalculiX-2.23.0-win-x64` — x64 zip from
     <https://github.com/calculix/CalculiX-Windows> (`ccx.exe` single-threaded,
     `ccx_MT.exe` multithreaded); runs emulated on ARM64.

   On Linux use the distribution package (`apt install calculix-ccx`). An explicit path can
   always be given in `solver.executable` or the `PILLARFEA_CCX` environment variable.

3. **Benchmark once per machine** (≈15–40 min, keep the machine awake):
   ```powershell
   python scripts/benchmark_solvers.py
   ```
   This runs one representative case with every build × thread count × linear solver,
   several times each, checks that all give the same result, and stores the fastest
   configuration *that never crashed* in `%LOCALAPPDATA%\pillarfea\solver_profile.json`.
   With `solver.executable: auto` (the default) that profile is used; without a profile, the
   first native build is used.

4. Check: `python -m pytest -q` (19 fast tests, no CalculiX needed).

### Architecture handling
`pillarfea/platform_info.py` detects the host's native architecture (also from an emulated
Python, via `IsWow64Process2`) and reads each executable's architecture from its PE/ELF
header, so builds are classified as native or emulated regardless of file names; builds that
cannot run on the host are ignored. There is no architecture-specific tuning: the benchmark
decides.

## 2. Running

All commands from the project root, with the venv's `python`.

| Task | Command |
|---|---|
| Interactive UI | `python -m pillarfea.ui [configs/some_case.yaml]` (or VS Code launch config *pillarfea UI*) |
| One case, figures + CSV | `python scripts/run_case.py configs/example_case.yaml --set geometry.taper_deg=5` |
| Parameter sweep | `python scripts/run_sweep.py configs/sweeps/taper_quick.yaml` |
| Validation suite | `python scripts/validate.py [--only uniaxial taper mesh increments substrate]` |
| Why does C differ from the 1D model? | `python scripts/diagnose_sections.py configs/example_case.yaml` |
| Install / benchmark solvers | `python scripts/install_ccx.py --all`, `python scripts/benchmark_solvers.py` |
| Unit tests | `python -m pytest -q` |

### The UI
* **Left panel** – all routine parameters. *Run / update* solves the case. FE results are
  cached by a hash of the FE-relevant parameters, so changing only the yield-extraction
  settings (method, offset %, slope) re-evaluates instantly; tick *force re-run* to bypass
  the cache.
* **Stress–strain** – FE curve with the offset/bilinear construction, the ideal-geometry
  reference, the input law and the 1D model. *Strain measure* selects raw / Sneddon-corrected
  / FE-foot-corrected strain (see §4).
* **Fields** – equivalent plastic strain or von Mises stress on the deformed, mirrored
  section; the slider moves through the stored frames, *at offset yield* jumps to the frame
  nearest the extracted yield point. The colour scale is clipped at a percentile (default
  99 %) because the corners are singular (§6); the true maximum is printed on the colour bar.
* **Mesh** – preview of the mesh for the current parameters (no solve).
* **Summary** – all extracted quantities.
* **Parameter scan** – vary one parameter over a list of values (runs in parallel), plot C,
  and write a CSV to `results/ui_scans/`.
* *Export figures + CSV* writes the curve/field figures (PNG, PDF), the config, the result
  row and the full force–displacement curve to `results/ui_exports/<timestamp>/`.

### Configuration
`pillarfea/defaults.yaml` documents every parameter. A case config only contains the keys it
changes; unknown keys are rejected (catches typos). Sweep files list a base config, fixed
overrides and a grid of dotted keys (`configs/sweeps/*.yaml`); the full factorial is run and
one CSV row per case is written. A grid axis may also be a list of dicts that set several
keys together, e.g. `contact: [{top: frictionless}, {top: coulomb, mu: 0.1}, {top: bonded}]`
(the CSV then has a `contact` label column). Units: µm, MPa, µN. Results are scale-invariant
except through the absolute fillet radius.

Sweeps provided: `taper_quick.yaml` (6 tapers × 4 contact conditions, ≈40 min) and
`full.yaml` (648 cases, ≈8–10 h; interrupted sweeps resume from the cache).

### Result reuse (cache)
Every FE solution is stored under `%LOCALAPPDATA%\pillarfea\runs\<key>` (≈1.5 MB per case),
where the key is a hash of **only the inputs that change the FE solution**, in canonical form:
* post-processing (method, offset, slope window, …), solver build/threads/linear solver and
  the number of stored field frames are excluded, so re-analysing never re-solves;
* numbers are normalised (`3` = `3.0`), `auto` values are resolved before hashing, and
  inputs that have no effect are dropped (μ unless contact is Coulomb, substrate size and
  substrate mesh when there is no substrate, the unused hardening description);
* the ideal-geometry reference depends only on material, max strain and increments, so one
  reference serves every geometry/contact/substrate variant.
Sweeps, scans, the UI, validation and diagnostics all go through the same cache, so any
case already solved anywhere is reused. Concurrent requests for the same case in one
process wait for a single solve. `--force` / *force re-run* bypasses the cache. Results from
v0.1 (old key format) are migrated automatically on first use.

**Sleep.** While a run, scan, sweep, validation or benchmark is in progress, idle system
sleep is blocked (Windows `SetThreadExecutionState`, macOS `caffeinate`; no-op elsewhere)
and released when it finishes. The screen may still turn off. Closing the lid can still
sleep, depending on the Windows power settings. To let the computer sleep, untick
*prevent sleep while running* in the UI, or pass `--allow-sleep` to the scripts.

### Computational cost
CalculiX internally expands axisymmetric elements into a thin 3D wedge (2–3× the DOFs of
a true 2D model). Benchmark of the representative case (≈1300 CAX8R elements,
200 increments) on a 12-core Snapdragon X Elite:

| Build | Mode | Threads | Completed runs | Typical wall time |
|---|---|---|---|---|
| MSYS2 clangarm64 `ccx.exe` | native ARM64 | 4 | 4 of 8 (rest crashed) | ≈ 60 s |
| MSYS2 clangarm64 `ccx.exe` | native ARM64 | 1 | 3 of 5 (rest crashed) | ≈ 100–115 s |
| CalculiX-Windows `ccx_MT.exe` | emulated x64 | 4 | 5 of 5 | ≈ 150–210 s (median 183 s) |
| CalculiX-Windows `ccx_MT.exe` | emulated x64 | 1 | 1 of 1 | ≈ 295 s |
| CalculiX-Windows `ccx.exe` | emulated x64 | 1 | 1 of 1 valid* | ≈ 325 s |
| any build, PaStiX solver | – | – | 0 (x64: "PASTIX library is not linked"; ARM64: crash) | – |

All completed runs gave identical results (C = 1.094427). \*A second run was invalid
because the laptop went to sleep. The native build would be ≈3× faster but is unreliable, so
the benchmark selects **emulated `ccx_MT.exe` with 4 threads** on this machine. Raw data:
`results/benchmarks/`.

For sweeps, 3 concurrent jobs × 4 threads suit a 12-core machine.

## 3. Model and assumptions

**Geometry.** Truncated cone: top diameter `D_top`, height `h = (h/D)·D_top`, sidewall angle
θ measured from the pillar axis (half-angle of the cone), so `D_bot = D_top + 2h·tanθ`.
Circular foot fillet of radius `fillet_radius` (absolute, µm; default 0.1 µm as typical
after FIB milling; 0 = sharp corner) tangent to sidewall and substrate surface. The
substrate is a hemisphere of radius `R_sub` (default 40 D_top) centred on the pillar foot,
fixed on its outer surface; this approximates an elastic half-space (sensitivity checked in
validation V6). The substrate surface outside the pillar is traction-free.

**Mesh.** Block-structured 8-node quadratic, reduced-integration axisymmetric elements
(CAX8R; avoids volumetric locking under plastic incompressibility). Refinement towards the
pillar surface, the top and the foot; the substrate is meshed by rays graded geometrically
away from the foot. Hoop strain/stress are part of the CAX8R formulation.

**Material.** Isotropic linear elasticity + von Mises (J2) plasticity, associated flow,
isotropic hardening; flow curve = true stress vs. true plastic strain, either linear
(`Et/E` is the slope of the uniaxial stress–strain curve, so the plastic modulus is
`H = E·Et/(E−Et)`) or tabulated (extended beyond the last point with the slope of the last
segment). Geometric nonlinearity is on (NLGEOM): CalculiX then uses its finite-strain
multiplicative J2 model and reports Cauchy stress. **PEEQ** (CalculiX/Abaqus name) is the
equivalent plastic strain, ε̄ᵖ = ∫ √(2/3 ε̇ᵖ:ε̇ᵖ) dt: the accumulated plastic strain at a
point. It is 0 where the material is still elastic; in uniaxial loading it equals the
plastic strain (PEEQ = 0.002 ↔ the 0.2 % offset); the current flow stress is the hardening
law evaluated at PEEQ.

**Substrate.** `none` = rigid frictionless base (pillar foot on rollers); `elastic` =
linear elastic (E, ν default to the pillar's); `elastoplastic` = same flow curve as the
pillar (as for a pillar milled into bulk material).

**Punch contact.** The punch is flat and (effectively) rigid, the pillar top is flat:
* `frictionless`: prescribed axial displacement on the top face, radial displacement free;
* `bonded`: prescribed axial displacement and zero radial displacement (no slip, μ → ∞);
* `coulomb`: a separate punch block (radius 1.3 R_top, E = 1000 E, ν = 0) driven on its
  top face, in surface-to-surface penalty contact with the pillar top (slave) and Coulomb
  friction μ with a stick slope λ (`*FRICTION`; μ = 0 omits it). Penalty stiffness
  K = `penalty_factor` · E / (smallest top-face node spacing), λ = `stick_ratio` · K.
  Validated against the two exact limits (§5): μ = 0 → frictionless within 0.08 %,
  μ = 5 → bonded within 0.04 %; penalty-induced error ≈ 0.05–0.1 % in C. Coulomb runs take
  ≈2× longer than the exact modes.
These are exact for a rigid flat punch while the top face stays in compression. This is
checked in every run: `top_min_nodal_force_ratio` is the smallest nodal punch force on the
top face (axis excluded) divided by the total force over the whole loading; a value below
−10⁻³ would mean the contact opens there (flagged in the UI summary). Runs solved before
this check was added report NaN.

**Loading.** Displacement-controlled, monotonic, `n_increments` equal increments up to
`max_strain` (auto: `2.5 σy/E + 1.5 %`).

## 4. Yield extraction — definitions

* Nominal stress `σ = F / A_top`, with `A_top` the **initial top area**.
* Nominal strain, three variants:
  * `raw` = δ_punch / h (what the instrument measures, including substrate sink-in);
  * `sneddon` = (δ_punch − F(1−ν_s²)/(E_s·D_bot)) / h — the usual flat-punch-on-half-space
    correction, evaluated with the foot diameter D_bot (not including the fillet);
  * `fe_base` = (δ_punch − ū_foot) / h, where ū_foot is the area-averaged axial displacement
    of the pillar foot from the FE solution (the "exact" pillar shortening; not measurable).
* Elastic slope: least-squares line fitted over `slope_window` (default 10–50 % of σy) of the
  loading curve (`slope: fit`), or the input E (`slope: material`).
* **Offset method:** first intersection of the curve with the line of that slope shifted by
  `offset_strain` (linear interpolation between increments).
* **Bilinear method:** continuous two-segment least-squares fit through the origin on
  ε ≤ `bilinear_max_strain`; yield = stress at the break point. The result depends on the fit
  range when yielding is gradual.
* **Reference (ideal geometry)** `σ_ref`: the same method applied to the engineering curve of
  a single element in homogeneous uniaxial compression with the same material and
  kinematics. This is what a perfect uniaxial test would report, so a perfect cylinder gives
  `C = 1` by construction.
* Also reported (offset method): `sigma_offset_input_law` = input true flow stress at plastic
  strain = offset, and `C_inputlaw = σ_app / sigma_offset_input_law`. With σy/E = 0.02 this
  differs from `σ_ref` by ≈1.5 %, because in compression F/A₀ exceeds the true stress as the
  section grows. `C` (ideal geometry) is the primary quantity.
* `D_eff/D_top = sqrt(σ_app/σ_ref)`: diameter whose area turns the measured force into the
  ideal yield stress; `z_eff/h`: depth below the top at which the pillar has that diameter.
* `sigma_prop_*`: proportional limit (first 0.5 % deviation from the fitted elastic line).
  Mesh-level first yield is not reported: it occurs at singular corners at vanishing load.
* **Analytical (geometric) corrections**, shown as thin lines in all C plots and stored as
  `C_avg_diameter`, `C_avg_area`, `C_avg_stress`. Each replaces A_top by an effective area A_eff,
  C = A_eff / A_top, using the cone (D_bot = D_top + 2h tanθ, fillet excluded):
  average diameter A_eff = π((D_t+D_b)/2)²/4; average area A_eff = (A_t + A_b)/2;
  average stress σ = (F/A_t + F/A_b)/2, i.e. C = 2/(1 + (D_t/D_b)²); height-averaged stress
  `C_avg_stress_integral` (mean of F/A(z) over the height, exact for a cone) C = D_b/D_t. They depend on geometry only (no material, friction or offset).
* `C_1d`: same extraction on the 1D progressive-yield model (each horizontal slice in
  uniaxial stress F/A(z), small strain, rigid base), relative to its own small-strain uniaxial
  reference. An independent sanity check of the taper effect, not a reference solution.

**Note on compliance.** With a fitted slope, any *linear* series compliance only shears the
curve and leaves the offset yield force unchanged, so `raw` and `sneddon` give identical C
(confirmed in V2). The compliance treatment matters for the modulus, and for the yield stress
only when the slope is fixed (`slope: material`) or the substrate yields.

## 5. Validation status

See `results/validation/` (figures, `validation_cases.csv`, `validation_summary.txt`).

Base case: h/D = 2, σy/E = 0.02, Et/E = 0.05, ν = 0.3, 0.2 % offset with fitted slope.
All 20 checks pass (`python scripts/validate.py`; re-running from the cache takes seconds).

| Check | Result |
|---|---|
| V1 straight cylinder, rigid frictionless base vs ideal-geometry reference | C = 1.00000 |
| V1 ideal-geometry FE vs independent analytical finite-strain estimate | σ0.2 2041.1 vs 2041.3 MPa (−0.01 %) |
| V1/V3 force equilibrium (punch force vs support reaction) | ≤ 5×10⁻⁶ relative |
| V2 cylinder on elastic substrate | C = 1.0006 (raw = Sneddon, as expected), 1.0014 (FE foot) |
| V4 mesh convergence, 3°, n = 8 → 24 elements across radius | frictionless: C = 1.09431 → 1.09450; bonded: 1.13075 → 1.12969 (slower, singular punch edge; default n = 16 within ≈0.05 %) |
| V5 load increments 50 / 100 / 200 / 400 | C = 1.09410 / 1.09439 / 1.09443 / 1.09442 (100 would suffice) |
| V6 substrate radius 20 / 40 / 80 D_top | C changes < 1×10⁻⁵; apparent modulus changes 0.6 % |
| Native ARM64 vs emulated x64 build | identical C (1.094427) |
| V7 Coulomb μ = 0 vs exact frictionless (3°) | 1.09359 vs 1.09443 (−0.08 %) |
| V7 Coulomb μ = 5 vs exact bonded (3°) | 1.12941 vs 1.12988 (−0.04 %) |
| V7 C vs μ (0, 0.1, 0.3, 5) | 1.0936, 1.1233, 1.1291, 1.1294 — monotonic |
| V7 penalty factor 10 / 100 / 1000 (μ = 0) | 1.09126 / 1.09359 / 1.09383 (default 100) |
| V8 fillet radius 0 / 50 / 100 / 200 nm (3°, D_top = 1 µm) | 1.09443 / 1.09445 / 1.09476 / 1.09527 |
| V8 mesh with 100 nm fillet, n = 12 / 16 / 24 | 1.09474 / 1.09476 / 1.09477 |

**Taper effect (validation base case), C and apparent over-estimate (C − 1):**

| θ | rigid base, frictionless | rigid base, bonded | elastic substrate, frictionless | elastic substrate, bonded | 1D model |
|---|---|---|---|---|---|
| 0° | 1.0000 (0.0 %) | 1.0009 (+0.1 %) | 1.0007 (+0.1 %) | 1.0020 (+0.2 %) | 1.0000 |
| 1° | 1.0436 (+4.4 %) | 1.0588 (+5.9 %) | 1.0412 (+4.1 %) | 1.0564 (+5.6 %) | 1.0333 (+3.3 %) |
| 2° | 1.0712 (+7.1 %) | 1.0978 (+9.8 %) | 1.0691 (+6.9 %) | 1.0955 (+9.6 %) | 1.0494 (+4.9 %) |
| 3° | 1.0963 (+9.6 %) | 1.1320 (+13.2 %) | 1.0945 (+9.4 %) | 1.1298 (+13.0 %) | 1.0619 (+6.2 %) |
| 5° | 1.1438 (+14.4 %) | 1.1927 (+19.3 %) | 1.1422 (+14.2 %) | 1.1906 (+19.1 %) | 1.0820 (+8.2 %) |

Comparison with the earlier screening at 3° (≈6 % 1D, 9 % free top, 14 % constrained
top): reproduced independently (6.2 %, 9.6 %, 13.2 %). Findings:
* **End constraint alone is negligible** at h/D = 2 (bonded vs frictionless at 0°: 0.1 %);
  the extra over-estimate of a bonded top is a taper × constraint interaction.
* **Why the FE exceeds the 1D model** (`scripts/diagnose_sections.py`): the yielding top
  zone is laterally confined by the wider, still-elastic material below. At offset yield
  (3°, frictionless) the section-mean radial and hoop stresses near the top are ≈ −100 MPa,
  so the axial stress exceeds the flow stress by ≈5 %. With weak hardening (plastic modulus
  ≈ 2.6 σy) this cuts the plastic strain near the top by a factor ≈3 relative to 1D, so a
  higher load is needed to reach 0.2 % offset. The 1D model ignores this confinement.
  (Incidentally, section-mean von Mises stress equals the flow stress at the mean PEEQ to
  0.1 MPa throughout the plastic zone — a consistency check of the plasticity.)
* With a bonded top a dead-metal cap forms under the punch and plasticity starts in a bowl
  below it (`results/validation/V3_fields_3deg_bonded.png`).
* The substrate (same E and ν as the pillar) changes C by < 0.3 % relative to a rigid base;
  with a fitted slope its (linear) compliance does not affect C at all, only the apparent
  modulus. A stiffer or softer elastic substrate should therefore change C very little
  (not yet run).

**Taper × contact sweep** (`configs/sweeps/taper_quick.yaml`: h/D = 2, σy/E = 0.02,
Et/E = 0.05, ν = 0.3, D_top = 1 µm, 100 nm fillet, elastic substrate with the pillar's E and ν (not the Si defaults), R_sub = 40 D_top; plot
`results/sweeps/taper_quick.png`, data `taper_quick.csv`):

| θ | frictionless | μ = 0.1 | μ = 0.3 | bonded | 1D | avg. diameter | avg. area | avg. stress (2-point) | avg. stress (height) |
|---|---|---|---|---|---|---|---|---|---|
| 0° | 1.0011 | 1.0034 | 1.0027 | 1.0027 | 1.0002 | 1 | 1 | 1 | 1 |
| 0.5° | 1.0248 | 1.0321 | 1.0328 | 1.0330 | 1.0219 | 1.0352 | 1.0355 | 1.0343 | 1.0349 |
| 1° | 1.0414 | 1.0545 | 1.0563 | 1.0565 | 1.0333 | 1.0710 | 1.0723 | 1.0673 | 1.0698 |
| 2° | 1.0693 | 1.0916 | 1.0953 | 1.0958 | 1.0494 | 1.1446 | 1.1494 | 1.1301 | 1.1397 |
| 3° | 1.0947 | 1.1237 | 1.1296 | 1.1301 | 1.0619 | 1.2206 | 1.2317 | 1.1881 | 1.2096 |
| 5° | 1.1426 | 1.1811 | 1.1902 | 1.1912 | 1.0820 | 1.3806 | 1.4112 | 1.2913 | 1.3500 |

* **Friction:** μ = 0.3 is indistinguishable from bonded (≤ 0.1 %); μ = 0.1 is already
  close to it (≤ 0.7 %). Only near-frictionless contact behaves like the free-top limit.
  (At 0° the μ = 0.1 / 0.3 order is reversed by 7×10⁻⁴, within the penalty accuracy.)
* **Analytical corrections over-correct beyond ≈1°.** They all replace A_top by a larger
  mean area, implicitly assuming the whole pillar governs yield; in reality the narrow top
  controls offset yield. At 0.5° they match the bonded FE result (≈3 %), at 3° they predict
  a 19–23 % over-estimate where the FE gives 9.5 % (frictionless) to 13 % (bonded), and at
  5° 29–41 % vs 14–19 % (the two-point average stress is the least severe). Applying them to tapered pillars would *under*-estimate the yield
  stress substantially.
* The 100 nm fillet changes C by < 0.1 % (D_top = 1 µm, 3°).

Not yet validated: substrate stiffness different from the pillar, elastoplastic substrate, tabulated hardening, the bilinear method on FE
curves (unit-tested on synthetic curves only).

## 6. Limitations and open points

* **Coulomb contact is a penalty method**: a small artificial compliance and penetration
  remain (≈0.05–0.1 % in C at the default penalty, verified against the exact limits).
  The stick slope λ written to `*FRICTION` had no measurable effect in CalculiX 2.23's
  surface-to-surface mode (10× change in λ → force history identical to 8×10⁻⁸), i.e. the
  solver evidently uses its own tangential regularisation there; `stick_ratio` is kept as
  an input but should not be expected to change results. The
  lift-off check (below) applies to the exact modes only; in Coulomb mode the contact
  algorithm handles separation itself.
* **Singularities.** The sharp re-entrant foot corner (no fillet) and, for bonded contact,
  the punch edge are stress singularities: local peak PEEQ/stress are mesh-dependent and
  "first yield" occurs at vanishing load. The global force–displacement curve and C
  converge (V4); local peak values do not and should not be interpreted.
* The punch is rigid (no indenter compliance), perfectly aligned and flat; no punch
  misalignment, no top-edge rounding, no FIB damage layer, no size effects.
* Contact is assumed to cover the whole top face throughout (checked per run, see §3; the
  validation runs that carry the check show no tension beyond numerical noise, ≤ 3×10⁻⁵ F).
* Sneddon correction uses D_bot and the substrate's elastic constants; for an
  elastoplastic substrate it is applied regardless (as experimentally).
* CalculiX specifics, verified: axisymmetric reaction forces are printed per 2° segment
  (×180 applied); `.dat` and `.frd` share one output frequency, so fields are written every
  increment and pruned to `n_field_frames` frames after the run; the Windows build prints
  3-digit exponents in `.frd` files (handled by the parser).
* The Windows CalculiX package is a community build marked "not fully tested"; the
  validation suite should be re-run whenever the solver binary changes.
* **The native ARM64 build (MSYS2 clangarm64, ccx 2.23) crashes intermittently** with an
  access violation (0xC0000005) on identical input: 2 of 6 repeated runs in testing, with
  1 and 4 threads, at varying increments. Its PaStiX solver crashes on every run. Completed
  runs agree with the x64 build to all printed digits. A crash without a CalculiX `*ERROR`
  message triggers an automatic re-run (`solver.max_attempts`, default 3; recorded in the
  case's `crashes.log` and `solver_attempts`), and the benchmark never auto-selects a
  configuration that crashed. Genuine CalculiX errors are not retried.
* Timings are only meaningful while the machine is awake; a laptop going to sleep during
  the benchmark inflated one run from ≈5 min to 67 min. Idle sleep is now blocked
  during runs (see *Sleep* above), but closing the lid can still suspend the machine.

## 7. Project layout

```
pillarfea/
  defaults.yaml   all parameters with documentation
  config.py       load/merge configs, resolve 'auto' values, cache keys
  geometry.py     pillar geometry and structured CAX8R mesh
  materials.py    elastic-plastic material, hardening tables, uniaxial law
  ccx_io.py       CalculiX input writer, runner, .dat/.frd parsers
  simulate.py     run a case (cached), ideal-geometry reference, analysis
  analysis.py     elastic fit, offset intersection, bilinear fit
  model1d.py      1D tapered progressive-yield model
  sweep.py        grid expansion, parallel runs, CSV
  plotting.py     publication-style figures
  ui.py           Tk desktop UI
  platform_info.py  host/executable architecture, solver discovery and selection
scripts/          run_case.py, run_sweep.py, validate.py, diagnose_sections.py,
                  install_ccx.py, benchmark_solvers.py
configs/          validation_base.yaml, examples, sweeps/
tests/            fast unit tests
results/          validation outputs, sweep CSVs, UI exports
```
