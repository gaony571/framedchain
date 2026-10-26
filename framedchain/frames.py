"""Backbone atoms -> coordinates of the space.

Edge i is Cα_i → Cα_{i+1}. Its frame R_i = [T, U, V] (columns):
  T = direction of the edge, U = C=O direction of the peptide bond across the edge with its T component removed and
  normalised, V = T × U.
Coordinates are x_i = s_i · R_i / 2, s_i = l_i / l0 (l_i is the edge length; default l0 = 3.83 Å). The squared norm is 0.75 s_i².
The first column times 2·l0 is the edge vector, so the Cα positions are recovered from the coordinates alone.
"""
import numpy as np

L0 = 3.83        # mean length of a trans edge (Å), the reference for length ratios
CIS_MAX = 3.45   # edges shorter than this are cis (cis about 3.09 Å, trans about 3.83 Å, nothing in between)
CAPS = ('ACE', 'NME', 'NHE')


def frames_from_backbone(CA, C, O):
    """Per-residue Cα, C, O (..., L, 3) -> frames R (..., L−1, 3, 3) and edge lengths l (..., L−1)."""
    CA, C, O = (np.asarray(a, dtype=np.float64) for a in (CA, C, O))
    e = CA[..., 1:, :] - CA[..., :-1, :]
    l = np.linalg.norm(e, axis=-1)
    T = e / l[..., None]
    u = O[..., :-1, :] - C[..., :-1, :]
    u = u - (u * T).sum(-1, keepdims=True) * T
    U = u / np.linalg.norm(u, axis=-1, keepdims=True)
    return np.stack([T, U, np.cross(T, U)], -1), l


def to_coordinates(R, l, l0=L0):
    """Frames and edge lengths -> coordinates x_i = (l_i / l0) R_i / 2."""
    return (np.asarray(l) / l0)[..., None, None] * np.asarray(R) / 2.0


def coordinates_from_backbone(CA, C, O, l0=L0):
    """Per-residue Cα, C, O (..., L, 3) -> coordinates (..., L−1, 3, 3)."""
    R, l = frames_from_backbone(CA, C, O)
    return to_coordinates(R, l, l0)


def edge_lengths(x, l0=L0):
    """Coordinates -> edge lengths l_i = 2·l0·|first column|. Zero terms give 0."""
    return 2.0 * l0 * np.linalg.norm(np.asarray(x)[..., :, 0], axis=-1)


def frames_of(x):
    """Coordinates of a real structure -> frames R_i and length ratios s_i (= 2·|first column|, independent of l0)."""
    x = np.asarray(x)
    s = 2.0 * np.linalg.norm(x[..., :, 0], axis=-1)
    return 2.0 * x / s[..., None, None], s


def is_real(x, tol=1e-3):
    """Whether each term has the form s·R/2 (s > 0, R a rotation). Zero terms and general matrices give False."""
    x = np.asarray(x)
    s = 2.0 * np.linalg.norm(x[..., :, 0], axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        R = 2.0 * x / s[..., None, None]
        orth = np.abs(np.swapaxes(R, -1, -2) @ R - np.eye(3)).max((-1, -2)) < tol
        return (s > 0) & orth & (np.linalg.det(R) > 0)


def is_cis(x, cis_max=CIS_MAX, l0=L0):
    """(Non-zero) terms whose edge length is shorter than cis_max."""
    l = edge_lengths(x, l0)
    return (l > 0) & (l < cis_max)


def n_edges(x, eps=1e-12):
    """Number of non-zero terms (the stratum of a point (n, 3, 3)). Terms with squared norm ≤ eps count as zero."""
    return int(((np.asarray(x) ** 2).sum((-1, -2)) > eps).sum())


def ca_trace(x, l0=L0):
    """Coordinates (..., n, 3, 3) -> Cα (..., n+1, 3), first Cα at the origin. A zero term is an edge of length 0."""
    e = 2.0 * l0 * np.asarray(x)[..., :, :, 0]
    z = np.zeros(e.shape[:-2] + (1, 3))
    return np.concatenate([z, np.cumsum(e, -2)], -2)


def read_pdb_backbone(path, caps=CAPS):
    """PDB (several MODELs allowed) -> dict(N, CA, C, O: (F, L, 3), resnames). Keeps residues with all four atoms and
    drops the residues in caps (terminal caps). For several altLocs the first one is used. Residues are joined in file
    order; chain breaks are not checked (they appear as long edges)."""
    models, cur, order, names = [], {}, [], {}
    with open(path) as f:
        for line in f:
            if line.startswith('MODEL'):
                cur = {}
            elif line.startswith('ENDMDL'):
                models.append(cur)
                cur = {}
            elif line.startswith(('ATOM', 'HETATM')):
                an, rn = line[12:16].strip(), line[17:20].strip()
                if rn in caps:
                    continue
                key = (line[21], line[22:27])
                if key not in names:
                    names[key] = rn
                    order.append(key)
                if an in ('N', 'CA', 'C', 'O') and an not in cur.get(key, {}):
                    cur.setdefault(key, {})[an] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
    if cur:
        models.append(cur)
    keys = [k for k in order if all(k in m and len(m[k]) == 4 for m in models)]
    out = {a: np.array([[m[k][a] for k in keys] for m in models]) for a in ('N', 'CA', 'C', 'O')}
    out['resnames'] = [names[k] for k in keys]
    return out


def coordinates_from_pdb(path, l0=L0, caps=CAPS):
    """PDB -> coordinates (F, L−1, 3, 3)."""
    b = read_pdb_backbone(path, caps)
    return coordinates_from_backbone(b['CA'], b['C'], b['O'], l0)
