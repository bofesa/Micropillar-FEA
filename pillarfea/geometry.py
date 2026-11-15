"""Pillar geometry and block-structured 8-node quadrilateral (CAX8R) mesh.

Coordinates: r = radial (CalculiX x), z = axial (CalculiX y). z = 0 is the substrate
surface (pillar foot), z = h the pillar top. Lengths in µm.

Blocks
  P  pillar: r = xi * R(z), 0 <= z <= h (mapped trapezoid, optional foot fillet)
  Q  substrate block under the pillar: 0 <= r <= R_base, -b <= z <= 0
  W  substrate "ring": rays from the right/bottom edges of Q to a quarter circle of
     radius R_sub (hemispherical substrate), geometrically graded outwards.
Corner nodes come from the block mappings; mid-side nodes are edge midpoints (except
on the pillar side wall, where they lie on the true surface, e.g. the fillet arc).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Geometry:
    D_top: float
    h: float
    taper_deg: float
    fillet_r: float = 0.0
    substrate: str = "elastic"
    R_sub: float = 40.0

    @property
    def R_top(self) -> float:
        return 0.5 * self.D_top

    @property
    def tan_t(self) -> float:
        return math.tan(math.radians(self.taper_deg))

    @property
    def R_bot(self) -> float:
        """Foot radius of the cone, without the fillet."""
        return self.R_top + self.h * self.tan_t

    @property
    def D_bot(self) -> float:
        return 2.0 * self.R_bot

    @property
    def fillet_t(self) -> float:
        """Tangent length of the fillet measured from the sharp foot corner."""
        if self.fillet_r <= 0:
            return 0.0
        return self.fillet_r * math.tan(0.5 * math.radians(90.0 - self.taper_deg))

    @property
    def fillet_z(self) -> float:
        return self.fillet_t * math.cos(math.radians(self.taper_deg))

    @property
    def R_base(self) -> float:
        """Radius where the pillar meets the flat substrate surface (incl. fillet)."""
        return self.R_bot + self.fillet_t

    @property
    def A_top(self) -> float:
        return math.pi * self.R_top ** 2

    def analytical_corrections(self) -> dict:
        """Simple geometric corrections C = σ_apparent/σ_true = A_eff / A_top (cone only,
        fillet excluded):
        avg_diameter: A_eff = pi/4 ((D_t + D_b)/2)^2
        avg_area:     A_eff = (A_top + A_bot) / 2   (mean of top and bottom areas)
        avg_stress:   σ_true = mean of the top and bottom stresses, (F/A_top + F/A_bot)/2,
                      i.e. C = 2 / (1 + (D_t/D_b)^2).
        avg_stress_integral: σ_true = height average of F/A(z); for a cone
                      <1/A> = 1/(pi r_t r_b), i.e. C = D_b / D_t."""
        dt, db = self.D_top, self.D_bot
        return {"C_avg_diameter": ((dt + db) / (2 * dt)) ** 2,
                "C_avg_area": (dt ** 2 + db ** 2) / (2 * dt ** 2),
                "C_avg_stress": 2 / (1 + (dt / db) ** 2),
                "C_avg_stress_integral": db / dt}

    def outer_radius(self, z):
        z = np.asarray(z, float)
        R = self.R_bot - z * self.tan_t
        if self.fillet_r > 0:
            rho, t = self.fillet_r, self.fillet_t
            arc = (self.R_bot + t) - np.sqrt(np.clip(rho ** 2 - (z - rho) ** 2, 0.0, None))
            R = np.where(z < self.fillet_z, arc, R)
        return R

    def validate(self) -> None:
        if self.D_top <= 0 or self.h <= 0:
            raise ValueError("D_top and h must be positive")
        if not 0 <= self.taper_deg < 45:
            raise ValueError("taper_deg must be in [0, 45)")
        if self.fillet_r < 0:
            raise ValueError("fillet_radius must be >= 0")
        if self.fillet_z >= 0.5 * self.h:
            raise ValueError("Fillet too large for the pillar height")
        if self.substrate != "none" and self.R_sub < 5 * self.R_base:
            raise ValueError("Substrate radius must be >= 5 x foot radius")


def geometry_from_cfg(c: dict) -> Geometry:
    g = c["geometry"]
    D = float(g["D_top"])
    geo = Geometry(D_top=D, h=float(g["h_over_D"]) * D, taper_deg=float(g["taper_deg"]),
                   fillet_r=float(g["fillet_radius"]), substrate=c["substrate"]["type"],
                   R_sub=float(c["substrate"]["radius_over_D"]) * D)
    geo.validate()
    return geo


@dataclass
class Mesh:
    nodes: np.ndarray                       # (N, 2) r, z
    elements: np.ndarray                    # (M, 8) 0-based, CAX8R ordering
    elsets: dict = field(default_factory=dict)
    nsets: dict = field(default_factory=dict)

    def element_areas(self) -> np.ndarray:
        """Area in the r-z plane of each element (straight-edge corner polygon)."""
        c = self.nodes[self.elements[:, :4]]
        x, y = c[..., 0], c[..., 1]
        return 0.5 * (x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y).sum(1)

    def element_volumes(self) -> np.ndarray:
        """Revolved volume per element (Pappus, straight-edge approximation)."""
        c = self.nodes[self.elements[:, :4]]
        x, y = c[..., 0], c[..., 1]
        cross = x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y
        A = 0.5 * cross.sum(1)
        xc = ((x + np.roll(x, -1, 1)) * cross).sum(1) / (6 * A)
        return 2 * math.pi * xc * A


# --- grading functions on [0,1] (n+1 corner positions) ----------------------------
def _fine_end(n: int, beta: float) -> np.ndarray:
    u = np.linspace(0.0, 1.0, n + 1)
    return u if beta <= 1e-9 else np.tanh(beta * u) / math.tanh(beta)


def _fine_start(n: int, beta: float) -> np.ndarray:
    return 1.0 - _fine_end(n, beta)[::-1]


def _fine_both(n: int, beta: float) -> np.ndarray:
    u = np.linspace(0.0, 1.0, n + 1)
    return u if beta <= 1e-9 else 0.5 * (1.0 + np.tanh(beta * (2 * u - 1)) / math.tanh(beta))


def _geometric(n: int, first_frac: float) -> np.ndarray:
    """n+1 positions on [0,1], geometric growth, first interval ~= first_frac."""
    u = np.linspace(0.0, 1.0, n + 1)
    if first_frac >= 1.0 / n:
        return u
    f = lambda lam: math.expm1(lam / n) / math.expm1(lam) - first_frac
    lo, hi = 1e-8, 300.0
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    lam = math.sqrt(lo * hi)
    return np.expm1(lam * u) / math.expm1(lam)


def _quadratic_grid(C: np.ndarray) -> np.ndarray:
    """Corner grid (A+1, B+1, 2) -> full quadratic point grid (2A+1, 2B+1, 2)."""
    a1, b1, _ = C.shape
    G = np.zeros((2 * a1 - 1, 2 * b1 - 1, 2))
    G[::2, ::2] = C
    G[1::2, ::2] = 0.5 * (C[:-1] + C[1:])
    G[::2, 1::2] = 0.5 * (C[:, :-1] + C[:, 1:])
    G[1::2, 1::2] = 0.25 * (C[:-1, :-1] + C[1:, :-1] + C[:-1, 1:] + C[1:, 1:])
    return G


def _grid_elements(idx: np.ndarray) -> np.ndarray:
    A, B = (idx.shape[0] - 1) // 2, (idx.shape[1] - 1) // 2
    out = []
    for a in range(A):
        for b in range(B):
            i, j = 2 * a, 2 * b
            out.append([idx[i, j], idx[i + 2, j], idx[i + 2, j + 2], idx[i, j + 2],
                        idx[i + 1, j], idx[i + 2, j + 1], idx[i + 1, j + 2], idx[i, j + 1]])
    return np.array(out, dtype=np.int64)


def build_mesh(geo: Geometry, m: dict, punch: bool = False) -> Mesh:
    nr, nz = int(m["n_radial"]), int(m["n_axial"])
    pts, blocks = [], {}
    count = 0

    def register(name, G):
        nonlocal count
        n = G.shape[0] * G.shape[1]
        idx = np.arange(count, count + n).reshape(G.shape[:2])
        pts.append(G.reshape(-1, 2))
        count += n
        blocks[name] = idx
        return idx

    # --- pillar block ----------------------------------------------------------
    xi = _fine_end(nr, float(m["radial_bias"]))
    z = _fine_both(nz, float(m["axial_bias"])) * geo.h
    CP = np.stack(np.broadcast_arrays(xi[:, None] * geo.outer_radius(z)[None, :],
                                      z[None, :]), -1).astype(float)
    CP[:, 0, 0] = xi * geo.R_base          # exact foot row (shared with the substrate)
    GP = _quadratic_grid(CP)
    zm = GP[-1, 1::2, 1]                   # side-wall mid-side nodes on the true surface
    GP[-1, 1::2, 0] = geo.outer_radius(zm)
    iP = register("P", GP)
    elsets = {"PILLAR": _grid_elements(iP)}

    nsets_raw = {"TOP": iP[:, -1], "PBASE": iP[:, 0], "TOPREF": iP[:1, -1], "AXIS": [iP[0, :]]}

    if geo.substrate != "none":
        a = geo.R_base
        b = a * float(m["sub_inner_depth"])
        nb = int(m["n_sub_inner"])
        zk = -b * (1.0 - _fine_end(nb, float(m["axial_bias"])))
        CQ = np.stack(np.broadcast_arrays(xi[:, None] * a, zk[None, :]), -1).astype(float)
        CQ[:, -1] = CP[:, 0]
        iQ = register("Q", _quadratic_grid(CQ))

        # boundary of Q: right edge top->bottom, then bottom edge outer->axis (corner pts)
        B = np.vstack([CQ[-1, ::-1], CQ[-2::-1, 0]])
        seg = np.linalg.norm(np.diff(B, axis=0), axis=1)
        h0 = np.empty(len(B))
        h0[0], h0[-1] = seg[0], seg[-1]
        h0[1:-1] = 0.5 * (seg[:-1] + seg[1:])
        h0 = np.convolve(np.pad(h0, 3, mode="edge"), np.ones(7) / 7, mode="valid")  # smooth
        # outer angle: blend of the radial direction and a uniform fan (avoids ray bunching)
        phi_rad = np.arctan2(B[:, 1], B[:, 0])
        phi_uni = np.linspace(0.0, -0.5 * math.pi, len(B))
        phi = 0.5 * (phi_rad + phi_uni)
        Qo = geo.R_sub * np.stack([np.cos(phi), np.sin(phi)], 1)
        Qo[0] = [geo.R_sub, 0.0]
        Qo[-1] = [0.0, -geo.R_sub]
        nring = int(m["n_ring"])
        CW = np.zeros((len(B), nring + 1, 2))
        for j in range(len(B)):
            L = np.linalg.norm(Qo[j] - B[j])
            s = _geometric(nring, h0[j] / L)
            CW[j] = B[j] + s[:, None] * (Qo[j] - B[j])
        CW[-1, :, 0] = 0.0
        CW[0, :, 1] = 0.0
        GW = _quadratic_grid(CW)
        # put outer mid-side nodes on the circle
        o = GW[:, -1]
        GW[:, -1] = o * (geo.R_sub / np.linalg.norm(o, axis=1))[:, None]
        iW = register("W", GW)
        elsets["SUBST"] = np.vstack([_grid_elements(iQ), _grid_elements(iW)])
        nsets_raw["AXIS"] += [iQ[0, :], iW[-1, :]]
        nsets_raw["OUTER"] = iW[:, -1]
        nsets_raw["SURF"] = iW[0, :]

    # --- merge coincident points, drop unused (cell-centre) points ----------------
    allp = np.vstack(pts)
    tol = 1e-9 * max(geo.D_top, 1.0)
    key = np.round(allp / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    el_all = {k: inv[v] for k, v in elsets.items()}
    used = np.unique(np.concatenate([v.ravel() for v in el_all.values()]))
    renum = -np.ones(len(first), dtype=np.int64)
    renum[used] = np.arange(len(used))
    nodes = allp[first][used]
    names = list(el_all)
    elements = np.vstack([renum[el_all[k]] for k in names])
    starts = np.cumsum([0] + [len(el_all[k]) for k in names])
    mesh_elsets = {k: np.arange(starts[i], starts[i + 1]) for i, k in enumerate(names)}

    nsets = {}
    for k, v in nsets_raw.items():
        arr = np.concatenate([np.ravel(x) for x in v]) if isinstance(v, list) else np.ravel(v)
        ids = renum[inv[arr]]
        nsets[k] = np.unique(ids[ids >= 0])

    mesh = Mesh(nodes, elements, mesh_elsets, nsets)
    # enforce counter-clockwise orientation
    neg = mesh.element_areas() < 0
    if neg.any():
        mesh.elements[neg] = mesh.elements[neg][:, [0, 3, 2, 1, 7, 6, 5, 4]]
    if (mesh.element_areas() <= 0).any():
        raise RuntimeError("Mesh generation produced inverted elements")
    if punch:
        _add_punch(mesh, geo, int(m["n_radial"]))
    return mesh


def _add_punch(mesh: Mesh, geo: Geometry, n_radial: int) -> None:
    """Append a separate (not merged) punch block on top of the pillar for contact.

    Radius 1.3 R_top (the pillar top edge can slide outwards), thickness 0.5 R_top,
    uniform mesh. Its bottom face coincides with the pillar top (zero initial gap); the
    punch nodes are deliberately NOT merged with the pillar nodes.
    Sets: PUNCH (elements), PTOP (driven face), PBOT (contact face), TOPREF := punch
    bottom node on the axis (punch displacement).
    """
    Rp, tp = 1.3 * geo.R_top, 0.5 * geo.R_top
    npr, npz = max(8, n_radial), 2
    C = np.stack(np.broadcast_arrays(np.linspace(0, Rp, npr + 1)[:, None],
                                     geo.h + np.linspace(0, tp, npz + 1)[None, :]), -1).astype(float)
    G = _quadratic_grid(C)
    idx = np.arange(G.shape[0] * G.shape[1]).reshape(G.shape[:2])
    els = _grid_elements(idx)
    used = np.unique(els)
    n0, m0 = len(mesh.nodes), len(mesh.elements)
    ren = -np.ones(idx.size, dtype=np.int64)
    ren[used] = n0 + np.arange(len(used))
    mesh.nodes = np.vstack([mesh.nodes, G.reshape(-1, 2)[used]])
    mesh.elements = np.vstack([mesh.elements, ren[els]])
    mesh.elsets["PUNCH"] = np.arange(m0, m0 + len(els))
    sel = lambda a: np.unique(ren[np.ravel(a)][ren[np.ravel(a)] >= 0])  # noqa: E731
    mesh.nsets["PTOP"] = sel(idx[:, -1])
    mesh.nsets["PBOT"] = sel(idx[:, 0])
    mesh.nsets["TOPREF"] = sel(idx[:1, 0])
    mesh.nsets["AXIS"] = np.unique(np.concatenate([mesh.nsets["AXIS"], sel(idx[0, :])]))
    if (mesh.element_areas() <= 0).any():
        raise RuntimeError("Punch mesh produced inverted elements")


def edge_faces(mesh: Mesh, elset: str, nset: str) -> list[tuple[int, int]]:
    """(element index, CalculiX face number 1-4) of element edges lying on a node set.
    Faces of quadrilateral (CAX8) elements: S1 = nodes 1-2, S2 = 2-3, S3 = 3-4, S4 = 4-1."""
    on = set(mesh.nsets[nset].tolist())
    out = []
    for e in mesh.elsets[elset]:
        c = mesh.elements[e, :4]
        for f, (a, b) in enumerate(((0, 1), (1, 2), (2, 3), (3, 0)), start=1):
            if c[a] in on and c[b] in on:
                out.append((int(e), f))
    return out
