"""Punch/contact mesh, fillet geometry, analytical corrections, grouped sweep axes."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pillarfea.ccx_io import write_inp  # noqa: E402
from pillarfea.config import fe_key, load_config, resolve  # noqa: E402
from pillarfea.geometry import Geometry, build_mesh, edge_faces, geometry_from_cfg  # noqa: E402
from pillarfea.materials import material_from_cfg  # noqa: E402
from pillarfea.sweep import contact_label, expand  # noqa: E402


def rc(**ov):
    o = {"solver.executable": "x", "geometry.D_top": 1.0}
    o.update({k.replace("__", "."): v for k, v in ov.items()})
    return resolve(load_config(overrides=o))


def test_punch_block_separate_and_faces():
    c = rc(contact__top="coulomb", contact__mu=0.2)
    geo = geometry_from_cfg(c)
    m = build_mesh(geo, c["mesh"], punch=True)
    assert "PUNCH" in m.elsets and (m.element_areas() > 0).all()
    # punch bottom nodes coincide with pillar-top height but are distinct nodes
    assert np.allclose(m.nodes[m.nsets["PBOT"], 1], geo.h)
    assert not set(m.nsets["PBOT"]) & set(m.nsets["TOP"])
    assert m.nodes[m.nsets["TOPREF"][0], 0] == 0.0
    slave = edge_faces(m, "PILLAR", "TOP")
    master = edge_faces(m, "PUNCH", "PBOT")
    assert len(slave) == c["mesh"]["n_radial"] and len(master) >= 8
    assert {f for _, f in slave} == {3} and {f for _, f in master} == {1}


def test_coulomb_deck(tmp_path):
    c = rc(contact__top="coulomb", contact__mu=0.3)
    geo = geometry_from_cfg(c)
    m = build_mesh(geo, c["mesh"], punch=True)
    mat = material_from_cfg(c["material"])
    fs = write_inp(tmp_path / "j.inp", m, mat, mat, c["contact"], 0.1, 50)
    txt = (tmp_path / "j.inp").read_text()
    assert fs == "PTOP"
    assert "*CONTACT PAIR, INTERACTION=SI1, TYPE=SURFACE TO SURFACE" in txt
    assert "*FRICTION\n0.3," in txt
    c0 = rc(contact__top="coulomb", contact__mu=0.0)
    write_inp(tmp_path / "k.inp", m, mat, mat, c0["contact"], 0.1, 50)
    assert "*FRICTION" not in (tmp_path / "k.inp").read_text()


def test_fillet_geometry_tangent():
    g = Geometry(1.0, 2.0, 3.0, fillet_r=0.1)
    zt = g.fillet_z
    straight = g.R_bot - zt * g.tan_t
    assert float(g.outer_radius(zt - 1e-9)) == pytest.approx(straight, abs=1e-6)
    assert float(g.outer_radius(0.0)) == pytest.approx(g.R_base)
    # slope continuity at the tangent point
    d = 1e-6
    s_arc = (float(g.outer_radius(zt - d)) - float(g.outer_radius(zt - 2 * d))) / d
    assert s_arc == pytest.approx(-g.tan_t, abs=1e-3)


def test_analytical_corrections():
    assert all(v == pytest.approx(1.0) for v in Geometry(1.0, 2.0, 0.0).analytical_corrections().values())
    g = Geometry(1.0, 2.0, 3.0)
    a = g.analytical_corrections()
    db = g.D_bot
    # average stress = mean of top and bottom stress F/A, relative to F/A_top
    assert a["C_avg_stress"] == pytest.approx(2 / (1 + 1 / db ** 2))
    assert a["C_avg_area"] == pytest.approx((1 + db ** 2) / 2)
    assert a["C_avg_diameter"] == pytest.approx(((1 + db) / 2) ** 2)
    assert all(v > 1 for v in a.values())
    # integral average stress = exact height average of F/A(z) for a cone
    z = np.linspace(0, g.h, 200001)
    inv = np.trapezoid(1 / (math.pi * g.outer_radius(z) ** 2), z) / g.h
    assert a["C_avg_stress_integral"] == pytest.approx(db)
    assert a["C_avg_stress_integral"] == pytest.approx(1 / (inv * g.A_top), rel=1e-6)


def test_grouped_grid_axis(tmp_path):
    sw = {"grid": {"geometry.taper_deg": [0, 3],
                   "contact": [{"top": "frictionless"}, {"top": "coulomb", "mu": 0.3}]}}
    cases = expand(sw, tmp_path)
    assert len(cases) == 4
    ov, cfg = cases[1]
    assert cfg["contact"]["top"] == "coulomb" and cfg["contact"]["mu"] == 0.3
    assert contact_label({"contact.top": "coulomb", "contact.mu": 0.3}) == "μ = 0.3"


def test_cache_keys_contact_and_fillet():
    # contact-algorithm settings do not matter for the exact (Dirichlet) contact modes
    assert fe_key(rc(contact__penalty_factor=5.0)) == fe_key(rc(contact__penalty_factor=50.0))
    # ... but they do for coulomb
    assert fe_key(rc(contact__top="coulomb", contact__penalty_factor=5.0)) != \
        fe_key(rc(contact__top="coulomb", contact__penalty_factor=50.0))
    assert fe_key(rc(contact__top="coulomb", contact__mu=0.1)) != \
        fe_key(rc(contact__top="coulomb", contact__mu=0.3))
    assert fe_key(rc(geometry__fillet_radius=0.0)) != fe_key(rc(geometry__fillet_radius=0.1))
