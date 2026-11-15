"""Architecture detection and result-cache key tests (no CalculiX needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pillarfea.config import fe_key, load_config, reference_key, resolve  # noqa: E402
from pillarfea.platform_info import exe_arch, host_arch, runs_on  # noqa: E402


def rc(**ov):
    o = {"solver.executable": "x"}
    o.update({k.replace("__", "."): v for k, v in ov.items()})
    return resolve(load_config(overrides=o))


def test_host_arch_known():
    assert host_arch() in ("arm64", "x64", "x86")


def test_python_exe_arch_matches_a_runnable_arch():
    a = exe_arch(sys.executable)
    assert a in ("arm64", "x64", "x86")
    assert runs_on(a, host_arch()) is not None


def test_runs_on_rules():
    assert runs_on("arm64", "arm64") == "native"
    assert runs_on("x64", "x64") == "native"
    assert runs_on("arm64", "x64") is None
    assert runs_on(None, "x64") is None


def test_fallback_selection_avoids_known_unstable(monkeypatch):
    import pillarfea.platform_info as pi
    sols = [pi.Solver("/a/ccx.exe", "arm64", True, "msys2-clangarm64/bin/ccx.exe"),
            pi.Solver("/b/ccx.exe", "x64", False, "CalculiX-2.23.0-win-x64/bin/ccx.exe"),
            pi.Solver("/b/ccx_MT.exe", "x64", False, "CalculiX-2.23.0-win-x64/bin/ccx_MT.exe")]
    monkeypatch.setattr(pi, "load_profile", lambda: None)
    monkeypatch.setattr(pi, "discover_solvers", lambda: sols)
    assert pi.select_solver(4)[0] == "/b/ccx_MT.exe"
    assert pi.select_solver(1)[0] == "/b/ccx.exe"
    sols.append(pi.Solver("/c/ccx.exe", "arm64", True, "msys2-ucrt-or-other/bin/ccx.exe"))
    assert pi.select_solver(4)[0] == "/c/ccx.exe"     # a stable native build wins


def test_cache_key_ignores_number_format_and_postprocessing():
    a = rc(geometry__taper_deg=3, analysis__offset_strain=0.002)
    b = rc(geometry__taper_deg=3.0, analysis__offset_strain=0.005, analysis__method="bilinear",
           solver__threads=1, loading__n_field_frames=10)
    assert fe_key(a) == fe_key(b)
    assert reference_key(a) == reference_key(b)


def test_cache_key_ignores_irrelevant_parameters():
    # mu is irrelevant unless contact is coulomb; substrate details irrelevant without substrate
    assert fe_key(rc(contact__mu=0.3)) == fe_key(rc(contact__mu=0.0))
    assert fe_key(rc(substrate__type="none", substrate__radius_over_D=20.0)) == \
        fe_key(rc(substrate__type="none", substrate__radius_over_D=80.0))
    # auto resolves to the same value as the explicit one
    m = rc()["loading"]["max_strain"]
    assert fe_key(rc(loading__max_strain=m)) == fe_key(rc())


def test_cache_key_sensitive_to_physics():
    assert fe_key(rc(geometry__taper_deg=3.0)) != fe_key(rc(geometry__taper_deg=2.0))
    assert fe_key(rc(substrate__radius_over_D=20.0)) != fe_key(rc(substrate__radius_over_D=40.0))
    assert reference_key(rc(geometry__taper_deg=1.0)) == reference_key(rc(geometry__taper_deg=5.0))


def test_portable_strips_user_paths(monkeypatch, tmp_path):
    import os
    from pathlib import Path
    from pillarfea import platform_info as pi
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\someone\AppData\Local")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path(r"C:\Users\someone")))
    s = r"crashed in C:\Users\someone\AppData\Local\pillarfea\runs\case_1 and C:\Users\someone\x"
    out = pi.portable(s)
    assert "someone" not in out
    assert r"%LOCALAPPDATA%\pillarfea\runs\case_1" in out and r"~\x" in out
    assert pi.portable(pi._PROJECT_ROOT / "results" / "a.csv") == os.path.join(".", "results", "a.csv")
