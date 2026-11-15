"""Configuration loading, merging, resolution of 'auto' values and hashing."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path

import yaml

from .platform_info import select_solver

_DEFAULTS = Path(__file__).with_name("defaults.yaml")


def default_config() -> dict:
    with open(_DEFAULTS, encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, over: dict | None, path: str = "") -> dict:
    """Merge `over` into a copy of `base`; unknown keys are an error (catches typos)."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in out:
            raise KeyError(f"Unknown config key '{path}{k}'")
        if isinstance(v, dict) and isinstance(out[k], dict):
            out[k] = deep_merge(out[k], v, f"{path}{k}.")
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, key: str, value) -> None:
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        if p not in d:
            raise KeyError(f"Unknown config key '{key}'")
        d = d[p]
    if parts[-1] not in d:
        raise KeyError(f"Unknown config key '{key}'")
    d[parts[-1]] = value


def get_dotted(cfg: dict, key: str):
    d = cfg
    for p in key.split("."):
        d = d[p]
    return d


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = default_config()
    if path:
        with open(path, encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    for k, v in (overrides or {}).items():
        set_dotted(cfg, k, v)
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def resolve(cfg: dict) -> dict:
    """Return a copy with all 'auto'/null entries replaced by concrete values."""
    c = copy.deepcopy(cfg)
    g, m, mesh, ld, an, sub, sol = (c[k] for k in
                                    ("geometry", "material", "mesh", "loading", "analysis",
                                     "substrate", "solver"))
    ey = float(m["sigma_y_over_E"])
    if ld["max_strain"] in (None, "auto"):
        ld["max_strain"] = round(2.5 * ey + 0.015, 6)
    if an["bilinear_max_strain"] in (None, "auto"):
        an["bilinear_max_strain"] = round(2.0 * ey + 0.01, 6)
    if mesh["n_axial"] in (None, "auto"):
        mesh["n_axial"] = int(math.ceil(0.6 * mesh["n_radial"] * 2.0 * g["h_over_D"]))
    if mesh["n_sub_inner"] in (None, "auto"):
        mesh["n_sub_inner"] = max(4, int(mesh["n_radial"]) // 2)
    if sub["E"] in (None, "auto"):
        sub["E"] = m["E"]
    if sub["nu"] in (None, "auto"):
        sub["nu"] = m["nu"]
    if sol["executable"] in (None, "auto"):
        (sol["executable"], sol["threads"], sol["linear_solver"],
         sol["selection"]) = select_solver(sol["threads"], sol.get("linear_solver", "auto"))
    else:
        sol["selection"] = "set in config"
        if sol["threads"] in (None, "auto"):
            sol["threads"] = 4
        if sol.get("linear_solver") in (None, "auto"):
            sol["linear_solver"] = "spooles"
    if sol["work_dir"] in (None, "auto"):
        sol["work_dir"] = str(Path(os.environ.get("LOCALAPPDATA", Path.home())) / "pillarfea" / "runs")
    ct = c["contact"]
    if ct["top"] not in ("frictionless", "bonded", "coulomb"):
        raise ValueError(f"Unknown contact.top '{ct['top']}'")
    if ct["top"] == "coulomb":
        if float(ct["mu"]) < 0:
            raise ValueError("contact.mu must be >= 0")
        if float(ct["penalty_factor"]) <= 0 or float(ct["stick_ratio"]) <= 0:
            raise ValueError("contact.penalty_factor and contact.stick_ratio must be > 0")
    if sub["type"] not in ("none", "elastic", "elastoplastic"):
        raise ValueError(f"Unknown substrate.type '{sub['type']}'")
    return c


def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _num(v):
    """Normalise numbers so 3, 3.0 and 3.0000000000001 hash identically."""
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 10)
    if isinstance(v, (list, tuple)):
        return [_num(x) for x in v]
    if isinstance(v, dict):
        return {k: _num(x) for k, x in v.items()}
    return v


def canonical_fe(rcfg: dict) -> dict:
    """Only the inputs that change the FE solution, with irrelevant entries removed.

    Post-processing (analysis), solver choice/threads, field-frame count and output options
    are excluded, so any case that differs only in those reuses the cached solution.
    """
    c = {k: copy.deepcopy(rcfg[k]) for k in
         ("geometry", "material", "substrate", "contact", "loading", "mesh")}
    c["loading"].pop("n_field_frames", None)
    # fillet stored relative to D_top (identical keys to pre-fillet_radius results)
    g = c["geometry"]
    g["fillet_over_D"] = float(g.pop("fillet_radius", 0.0) or 0.0) / float(g["D_top"])
    if c["contact"]["top"] != "coulomb":   # contact-algorithm settings are irrelevant
        c["contact"] = {"top": c["contact"]["top"], "mu": 0.0}
    h = c["material"]["hardening"]
    if h["type"] == "linear":
        h["plastic_strain"] = h["stress_over_sigma_y"] = None
    else:
        h["Et_over_E"] = None
    if c["substrate"]["type"] == "none":
        c["substrate"] = {"type": "none"}
        for k in ("n_sub_inner", "sub_inner_depth", "n_ring"):
            c["mesh"].pop(k, None)
    return _num(c)


def fe_key(rcfg: dict) -> str:
    return "case_" + _hash(canonical_fe(rcfg))


def reference_key(rcfg: dict) -> str:
    c = canonical_fe(rcfg)
    return "ref_" + _hash({"material": c["material"],
                           "loading": {k: c["loading"][k] for k in ("max_strain", "n_increments")}})


def legacy_keys(rcfg: dict) -> tuple[str, str]:
    """Cache keys used before canonicalisation (v0.1 runs), for migrating old results."""
    keys = ("geometry", "material", "substrate", "contact", "loading", "mesh")
    return ("case_" + _hash({k: rcfg[k] for k in keys}),
            "ref_" + _hash({"material": rcfg["material"], "loading": rcfg["loading"]}))
