"""Install a CalculiX build matching this machine into %LOCALAPPDATA%/pillarfea/ccx/.

    python scripts/install_ccx.py            # build(s) for the host architecture
    python scripts/install_ccx.py --all      # every build that can run here (for benchmarking)
    python scripts/install_ccx.py --list     # show installed builds and their architecture

Sources
  msys2-clangarm64  native Windows ARM64, MSYS2 package mingw-w64-clang-aarch64-calculix-ccx
                    (dependencies resolved from the MSYS2 repository database; only .exe/.dll
                    files are extracted, no MSYS2 installation needed; SHA-256 verified)
  msys2-ucrt64      native Windows x64, MSYS2 package mingw-w64-ucrt-x86_64-calculix-ccx
  calculix-windows  Windows x64 zip from github.com/calculix/CalculiX-Windows (also runs
                    emulated on Windows ARM64)
On Linux use the distribution package (e.g. `apt install calculix-ccx`).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import re
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pillarfea.platform_info import ccx_home, discover_solvers, host_arch  # noqa: E402

MSYS2 = {
    "msys2-clangarm64": ("clangarm64", "mingw-w64-clang-aarch64-calculix-ccx", "arm64"),
    "msys2-ucrt64": ("ucrt64", "mingw-w64-ucrt-x86_64-calculix-ccx", "x64"),
}
CW_URL = ("https://github.com/calculix/CalculiX-Windows/raw/master/releases/"
          "CalculiX-2.23.0-win-x64.zip")


def _get(url: str) -> bytes:
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def _untar(data: bytes) -> tarfile.TarFile:
    if data[:4] == b"\x28\xb5\x2f\xfd":  # zstd magic
        import zstandard
        data = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)).read()
    return tarfile.open(fileobj=io.BytesIO(data))


def _read_db(repo: str) -> dict:
    tf = _untar(_get(f"https://mirror.msys2.org/mingw/{repo}/{repo}.db"))
    pkgs, provides = {}, {}
    for m in tf.getmembers():
        if not m.name.endswith("/desc"):
            continue
        txt = tf.extractfile(m).read().decode()
        sec, cur = {}, None
        for line in txt.splitlines():
            if line.startswith("%") and line.endswith("%"):
                cur = line.strip("%")
                sec[cur] = []
            elif line and cur:
                sec[cur].append(line)
        name = sec["NAME"][0]
        pkgs[name] = {"file": sec["FILENAME"][0], "sha256": sec.get("SHA256SUM", [None])[0],
                      "depends": [re.split(r"[<>=]", d)[0] for d in sec.get("DEPENDS", [])]}
        for p in sec.get("PROVIDES", []):
            provides[re.split(r"[<>=]", p)[0]] = name
    return pkgs, provides


def install_msys2(key: str) -> Path:
    repo, top, arch = MSYS2[key]
    print(f"[{key}] resolving dependencies of {top}")
    pkgs, provides = _read_db(repo)
    todo, seen = [top], []
    while todo:
        n = todo.pop()
        n = n if n in pkgs else provides.get(n, n)
        if n in seen:
            continue
        if n not in pkgs:
            raise RuntimeError(f"package {n} not found in {repo}")
        seen.append(n)
        todo += pkgs[n]["depends"]
    dest = ccx_home() / key / "bin"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  {len(seen)} packages: {', '.join(s.split('-', 4)[-1] for s in seen)}")
    for n in seen:
        p = pkgs[n]
        data = _get(f"https://mirror.msys2.org/mingw/{repo}/{p['file']}")
        if p["sha256"] and hashlib.sha256(data).hexdigest() != p["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {p['file']}")
        tf = _untar(data)
        for m in tf.getmembers():
            fn = m.name.split("/")
            if len(fn) == 3 and fn[1] == "bin" and m.isfile() and (
                    fn[2].lower().endswith(".dll") or fn[2] in ("ccx.exe",)):
                (dest / fn[2]).write_bytes(tf.extractfile(m).read())
    (dest.parent / "SOURCE.txt").write_text(
        f"MSYS2 {repo}: {top} and dependencies\n" + "\n".join(pkgs[n]["file"] for n in seen))
    print(f"  installed {dest / 'ccx.exe'}")
    return dest / "ccx.exe"


def install_calculix_windows() -> Path:
    key = "calculix-windows"
    data = _get(CW_URL)
    dest = ccx_home() / key
    zipfile.ZipFile(io.BytesIO(data)).extractall(dest)
    exe = next(dest.glob("**/bin/ccx.exe"))
    print(f"  installed {exe}")
    return exe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="install every build runnable on this host")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", choices=list(MSYS2) + ["calculix-windows"])
    a = ap.parse_args()
    if not a.list:
        if sys.platform != "win32":
            sys.exit("On Linux install the distribution package, e.g. `apt install calculix-ccx`.")
        h = host_arch()
        if a.only:
            todo = [a.only]
        elif h == "arm64":
            # the native MSYS2 ARM64 build crashes intermittently (see README), so the
            # reliable x64 build (emulated) is the default; --all adds the native one
            todo = ["calculix-windows"] + (["msys2-clangarm64"] if a.all else [])
        else:
            todo = ["msys2-ucrt64"] + (["calculix-windows"] if a.all else [])
        print(f"host architecture: {h}; installing {todo}")
        for k in todo:
            install_calculix_windows() if k == "calculix-windows" else install_msys2(k)
    print("\ninstalled solvers:")
    for s in discover_solvers():
        print(f"  {s.arch:6s} {'native' if s.native else 'emulated':9s} {s.path}")


if __name__ == "__main__":
    main()
