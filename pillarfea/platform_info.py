"""Host-architecture detection, CalculiX discovery and solver selection.

* host_arch(): the machine's native architecture ('arm64' | 'x64' | 'x86'), also correct when
  Python itself runs emulated (Windows: IsWow64Process2).
* exe_arch(): architecture of an executable, read from its PE/ELF header (not its name).
* discover_solvers(): all CalculiX executables found (config/env, %LOCALAPPDATA%/pillarfea/ccx,
  PATH) that can run on this host, natively or under emulation.
* select_solver(): the benchmark profile written by scripts/benchmark_solvers.py if one exists
  for this host, else the first native build, else an emulated one.
No architecture-specific tuning is done; the benchmark decides.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_PE = {0xAA64: "arm64", 0x8664: "x64", 0x014C: "x86"}

# Builds observed to crash intermittently on identical input (reliability, not speed):
# MSYS2 clangarm64 ccx 2.23 -> access violation in 5 of 10 runs (Nov 2026, Snapdragon X).
# Used only when no benchmark profile exists; the benchmark itself measures reliability.
KNOWN_UNSTABLE = ("msys2-clangarm64",)
_ELF = {0xB7: "arm64", 0x3E: "x64", 0x03: "x86"}


def host_arch() -> str:
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.windll.kernel32
            pm, nm = wintypes.USHORT(), wintypes.USHORT()
            if k32.IsWow64Process2(k32.GetCurrentProcess(), ctypes.byref(pm), ctypes.byref(nm)):
                return _PE.get(nm.value, f"0x{nm.value:x}")
        except (AttributeError, OSError):
            pass
    m = platform.machine().lower()
    return {"aarch64": "arm64", "arm64": "arm64", "x86_64": "x64", "amd64": "x64",
            "i386": "x86", "i686": "x86", "x86": "x86"}.get(m, m)


def exe_arch(path: str | Path) -> str | None:
    try:
        with open(path, "rb") as f:
            head = f.read(4096)
    except OSError:
        return None
    if head[:2] == b"MZ" and len(head) >= 0x40:
        off = struct.unpack_from("<I", head, 0x3C)[0]
        if head[off:off + 4] == b"PE\0\0":
            return _PE.get(struct.unpack_from("<H", head, off + 4)[0])
    if head[:4] == b"\x7fELF":
        return _ELF.get(struct.unpack_from("<H", head, 18)[0])
    return None


def runs_on(arch: str | None, host: str) -> str | None:
    """'native', 'emulated' or None (cannot run)."""
    if arch is None:
        return None
    if arch == host:
        return "native"
    if sys.platform == "win32" and host == "arm64" and arch in ("x64", "x86"):
        return "emulated"          # Windows 11 on ARM emulates x64/x86
    if host == "x64" and arch == "x86":
        return "emulated"
    return None


@dataclass
class Solver:
    path: str
    arch: str
    native: bool
    label: str

    @property
    def multithread_variant(self) -> bool:
        return Path(self.path).stem.lower().endswith("_mt")


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def portable(text) -> str:
    """Strip user-specific path prefixes for output that may be shared (CSVs, logs):
    project root -> '.', %LOCALAPPDATA% -> '%LOCALAPPDATA%', home -> '~'.
    Internal code keeps absolute paths; this is for display/export only."""
    s = str(text)
    subs = [(str(_PROJECT_ROOT), "."), (os.environ.get("LOCALAPPDATA"), "%LOCALAPPDATA%"),
            (str(Path.home()), "~")]
    for prefix, rep in subs:
        if not prefix:
            continue
        for p in {prefix, prefix.replace("\\", "/")}:
            i = s.lower().find(p.lower())
            while i >= 0:
                s = s[:i] + rep + s[i + len(p):]
                i = s.lower().find(p.lower(), i + len(rep))
    return s


def ccx_home() -> Path:
    base = os.environ.get("LOCALAPPDATA") or (Path.home() / ".local" / "share")
    return Path(base) / "pillarfea" / "ccx"


def profile_path() -> Path:
    return ccx_home().parent / "solver_profile.json"


def _candidates() -> list[Path]:
    c: list[Path] = []
    if os.environ.get("PILLARFEA_CCX"):
        c.append(Path(os.environ["PILLARFEA_CCX"]))
    home = ccx_home()
    if home.exists():
        for pat in ("**/ccx.exe", "**/ccx_MT.exe", "**/bin/ccx"):
            c += sorted(home.glob(pat))
    for name in ("ccx", "ccx_MT", "ccx_2.23", "ccx_2.22", "ccx_2.21"):
        w = shutil.which(name)
        if w:
            c.append(Path(w))
    return c


def discover_solvers() -> list[Solver]:
    host = host_arch()
    out, seen = [], set()
    for p in _candidates():
        rp = str(p.resolve())
        if rp.lower() in seen or not p.exists():
            continue
        seen.add(rp.lower())
        a = exe_arch(p)
        mode = runs_on(a, host)
        if mode is None:
            continue
        try:
            label = str(p.resolve().relative_to(ccx_home().resolve())).replace("\\", "/")
        except ValueError:
            label = rp
        out.append(Solver(rp, a, mode == "native", label))
    out.sort(key=lambda s: (not s.native, s.multithread_variant))
    return out


def load_profile() -> dict | None:
    p = profile_path()
    if not p.exists():
        return None
    try:
        prof = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    if prof.get("host_arch") != host_arch() or not Path(prof["best"]["path"]).exists():
        return None
    return prof


def select_solver(threads="auto", linear_solver="auto") -> tuple[str, int, str, str]:
    """Return (executable, threads, linear_solver, reason); explicit threads/solver win."""
    prof = load_profile()
    if prof:
        b = prof["best"]
        t = int(b["threads"]) if threads in (None, "auto") else int(threads)
        ls = b.get("linear_solver", "spooles") if linear_solver in (None, "auto") else linear_solver
        return b["path"], t, ls, f"benchmark profile ({prof.get('date', '?')})"
    sols = discover_solvers()
    if not sols:
        raise FileNotFoundError(
            "No runnable CalculiX executable found. Run `python scripts/install_ccx.py`, or set "
            "solver.executable in the config / the PILLARFEA_CCX environment variable.")
    t = 4 if threads in (None, "auto") else int(threads)
    ls = "spooles" if linear_solver in (None, "auto") else linear_solver
    # no profile: native before emulated, known-unstable builds last, multithreaded variant
    # preferred when more than one thread is requested
    def rank(s):
        return (any(k in s.label for k in KNOWN_UNSTABLE), not s.native,
                (t > 1) != s.multithread_variant)
    s = min(sols, key=rank)
    note = " [known unstable]" if any(k in s.label for k in KNOWN_UNSTABLE) else ""
    return s.path, t, ls, (f"{'native' if s.native else 'emulated'} build{note}, no benchmark "
                           "profile (run scripts/benchmark_solvers.py)")


def save_profile(best: dict, results: list[dict], date: str) -> Path:
    p = profile_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"host_arch": host_arch(), "date": date, "best": best,
                             "results": results}, indent=1))
    return p


def describe() -> str:
    lines = [f"host architecture: {host_arch()} (Python: {platform.machine()}, "
             f"{platform.python_version()})"]
    for s in discover_solvers():
        flag = "  [known unstable]" if any(k in s.label for k in KNOWN_UNSTABLE) else ""
        lines.append(f"  {s.arch:6s} {'native' if s.native else 'emulated':9s} {s.label}{flag}")
    sel = select_solver()
    lines.append(f"selected: {sel[0]}, {sel[1]} threads, {sel[2]} [{sel[3]}]")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
