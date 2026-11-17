"""Fast unit tests (no CalculiX needed):  python -m pytest -q"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pillarfea.analysis import bilinear_fit, extract, offset_intersection  # noqa: E402
from pillarfea.ccx_io import _number_regex, von_mises  # noqa: E402
from pillarfea.config import load_config, resolve  # noqa: E402
from pillarfea.geometry import Geometry, build_mesh  # noqa: E402
from pillarfea.materials import material_from_cfg  # noqa: E402
from pillarfea.model1d import tapered_curve  # noqa: E402

E, SY, ET = 1e5, 2000.0, 5000.0
H = E * ET / (E - ET)
AN = {"method": "offset", "offset_strain": 0.002, "slope": "fit", "slope_window": [0.1, 0.5],
      "bilinear_max_strain": 0.05, "prop_limit_tol": 0.005}


def bilinear_curve(n=400, emax=0.06):
    e = np.linspace(0, emax, n)
    s = np.where(e < SY / E, E * e, SY + ET * (e - SY / E))
    return e, s


def test_offset_exact_on_bilinear():
    e, s = bilinear_curve(2000)
    x = extract(e, s, AN, SY, E)
    assert x["E_slope"] == pytest.approx(E, rel=1e-9)
    assert x["sigma_yield"] == pytest.approx(SY + H * 0.002, rel=1e-6)


def test_offset_variable():
    e, s = bilinear_curve(2000)
    x = extract(e, s, dict(AN, offset_strain=0.005), SY, E)
    assert x["sigma_yield"] == pytest.approx(SY + H * 0.005, rel=1e-6)


def test_bilinear_fit_recovers_yield():
    e, s = bilinear_curve(600)
    b = bilinear_fit(e, s, 0.05)
    assert b["sigma_y"] == pytest.approx(SY, rel=1e-6)
    assert b["H"] == pytest.approx(ET, rel=1e-6)


def test_offset_invariant_to_linear_compliance():
    """Fitted-slope offset yield is unchanged by adding a linear series compliance."""
    e, s = bilinear_curve(4000)
    x0 = extract(e, s, AN, SY, E)
    x1 = extract(e + 0.3 * s / E, s, AN, SY, E)
    assert x1["sigma_yield"] == pytest.approx(x0["sigma_yield"], rel=1e-6)


def test_material_offset_stress():
    c = load_config(overrides={"material.E": E, "material.sigma_y_over_E": SY / E,
                               "material.hardening.Et_over_E": ET / E})
    m = material_from_cfg(c["material"])
    assert m.offset_stress(0.002) == pytest.approx(SY + H * 0.002)
    assert m.uniaxial_small_strain(0.03) == pytest.approx(SY + ET * (0.03 - SY / E), rel=1e-5)


def test_1d_model_zero_taper_is_uniaxial():
    c = load_config(overrides={"material.E": E, "material.sigma_y_over_E": SY / E,
                               "material.hardening.Et_over_E": ET / E})
    m = material_from_cfg(c["material"])
    g = Geometry(1.0, 2.0, 0.0)
    e, s = tapered_curve(g, m, 0.05)
    assert np.allclose(s, m.uniaxial_small_strain(e), rtol=1e-4, atol=1e-6)


def test_1d_model_taper_overestimates():
    c = load_config(overrides={"material.E": E, "material.sigma_y_over_E": SY / E,
                               "material.hardening.Et_over_E": ET / E})
    m = material_from_cfg(c["material"])
    e0, s0 = tapered_curve(Geometry(1.0, 2.0, 0.0), m, 0.06)
    e3, s3 = tapered_curve(Geometry(1.0, 2.0, 3.0), m, 0.06)
    y0 = extract(e0, s0, AN, SY, E)["sigma_yield"]
    y3 = extract(e3, s3, AN, SY, E)["sigma_yield"]
    assert 1.03 < y3 / y0 < 1.10   # screening value ~6 %


@pytest.mark.parametrize("taper,fillet,sub", [(0, 0, "none"), (3, 0, "elastic"), (5, 0.1, "elastic")])
def test_mesh_volume_and_orientation(taper, fillet, sub):
    c = resolve(load_config(overrides={"geometry.taper_deg": taper, "geometry.fillet_radius": fillet,
                                       "geometry.D_top": 1.0, "substrate.type": sub,
                                       "solver.executable": "x", "contact.top": "frictionless"}))
    g = Geometry(1.0, 2.0, taper, fillet, sub, 40.0)
    m = build_mesh(g, c["mesh"])
    assert (m.element_areas() > 0).all()
    Vp = m.element_volumes()[m.elsets["PILLAR"]].sum()
    Vcone = math.pi * g.h / 3 * (g.R_top ** 2 + g.R_top * g.R_bot + g.R_bot ** 2)
    if fillet == 0:
        assert Vp == pytest.approx(Vcone, rel=1e-6)
    else:
        assert Vp > Vcone
    assert np.allclose(m.nodes[m.nsets["TOP"], 1], g.h)
    assert np.allclose(m.nodes[m.nsets["AXIS"], 0], 0.0)


def test_unknown_key_rejected():
    with pytest.raises(KeyError):
        load_config(overrides={"geometry.tapr_deg": 3})


def test_frd_number_parsing_windows_and_linux():
    win = "2.45208E-011-3.25037E+0014.04663E-011"
    lin = "2.45208E-11-3.25037E+014.04663E-11"
    assert [float(x) for x in _number_regex(win).findall(win)] == [2.45208e-11, -32.5037, 4.04663e-11]
    assert [float(x) for x in _number_regex(lin).findall(lin)] == [2.45208e-11, -32.5037, 4.04663e-11]


def test_von_mises_uniaxial():
    assert von_mises(np.array([0.0, -100.0, 0.0, 0.0])) == pytest.approx(100.0)


def test_all_sources_compile_and_ui_imports():
    """Catch syntax errors in modules not exercised by other tests (UI, scripts)."""
    import importlib
    import py_compile
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for f in list((root / "pillarfea").glob("*.py")) + list((root / "scripts").glob("*.py")):
        py_compile.compile(str(f), doraise=True)
    importlib.import_module("pillarfea.ui")
