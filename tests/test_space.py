"""Tests of the definitions and proven properties of the space, on synthetic frame sequences (pytest)."""
import math

import numpy as np
import torch

import framespace as fs

LOC_C, LOC_O = np.array([1.42, 0.53, 0.01]), np.array([1.65, 1.74, 0.01])   # trans frame (local positions of C and O)


def chain(n, seed, noise=0.15, cis=()):
    """Coordinates (n, 3, 3) = s R / 2 of a real structure whose neighbouring edges have similar rotations. Trans s ≈ 1 ± 0.016, cis positions s = 0.81."""
    g = torch.Generator().manual_seed(seed)
    step = fs.expmap(torch.tensor([0.3, 1.2, 0.4], dtype=torch.float64))
    R = [torch.eye(3, dtype=torch.float64)]
    for _ in range(n - 1):
        R.append(R[-1] @ step @ fs.expmap(noise * torch.randn(3, generator=g, dtype=torch.float64)))
    s = 1 + 0.016 * torch.randn(n, generator=g, dtype=torch.float64)
    for i in cis:
        s[i] = 0.81
    return s[:, None, None] * torch.stack(R) / 2


def test_quaternion_roundtrip():
    R = fs.random_rotations(200, seed=1)
    assert torch.allclose(fs.quat_to_mat(fs.mat_to_quat(R)), R, atol=1e-10)


def test_procrustes_and_polar():
    U = fs.random_rotations(5, seed=2)
    assert torch.allclose(fs.procrustes(U.transpose(-1, -2) * 3.0), U, atol=1e-10)
    R, s = fs.polar(0.5 * U)
    assert torch.allclose(R, U, atol=1e-10) and torch.allclose(s, torch.full((5,), 0.5, dtype=torch.float64))


def test_coordinates_roundtrip():
    """Atoms -> coordinates -> Cα: Cα is recovered exactly from the coordinates alone, and lengths and cis bonds can be read off."""
    x = chain(30, 3, cis=(7,)).numpy()
    CA = fs.ca_trace(x)
    R, s = fs.frames_of(x)
    C = np.concatenate([CA[:-1] + R @ LOC_C, CA[-1:] + 1.0])
    O = np.concatenate([CA[:-1] + R @ LOC_O, CA[-1:] + 2.0])
    x2 = fs.coordinates_from_backbone(CA, C, O)
    assert np.allclose(x2, x, atol=1e-10) and fs.is_real(x).all()
    assert np.allclose(fs.edge_lengths(x), s * fs.L0) and fs.is_cis(x)[7] and fs.is_cis(x).sum() == 1
    assert np.allclose(np.linalg.norm(CA[1:] - CA[:-1], axis=-1), s * fs.L0)
    assert np.allclose((x ** 2).sum((-1, -2)), 0.75 * s ** 2)


def test_pdb_roundtrip(tmp_path):
    x = chain(25, 4, cis=(6,)).numpy()
    CA = fs.ca_trace(x)
    R, s = fs.frames_of(x)
    C = np.concatenate([CA[:-1] + R @ LOC_C, CA[-1:] + 1.0])
    O = np.concatenate([CA[:-1] + R @ LOC_O, CA[-1:] + 2.0])
    N = CA - 1.0
    p = tmp_path / 'x.pdb'
    with open(p, 'w') as f:
        k = 0
        for i in range(len(CA)):
            for name, xyz in (('N', N[i]), ('CA', CA[i]), ('C', C[i]), ('O', O[i])):
                k += 1
                f.write('ATOM  %5d  %-3s ALA A%4d    %8.3f%8.3f%8.3f  1.00  0.00\n' % (k, name, i + 1, *xyz))
    xp = fs.coordinates_from_pdb(str(p))
    assert xp.shape == (1, 25, 3, 3) and np.allclose(xp[0], x, atol=2e-3) and fs.is_cis(xp[0])[6]


def test_settable_constants():
    """l0 and the cis threshold are settable. With a different l0, Cα recovery with the same l0 is unchanged."""
    x = chain(12, 5, cis=(3,)).numpy()
    CA = fs.ca_trace(x)
    R, s = fs.frames_of(x)
    l = s * fs.L0
    x2 = fs.to_coordinates(R, l, l0=3.80)
    assert np.allclose(fs.ca_trace(x2, l0=3.80), CA) and np.allclose(fs.edge_lengths(x2, l0=3.80), l)
    assert fs.is_cis(x).sum() == 1 and fs.is_cis(x, cis_max=2.0).sum() == 0


def test_self_and_symmetry():
    x, y = chain(25, 1), chain(31, 2)
    d0, _ = fs.distance(x, x)
    assert float(d0) < 1e-6
    dxy, _ = fs.distance(x, y)
    dyx, _ = fs.distance(y, x)
    assert abs(float(dxy) - float(dyx)) < 1e-6


def test_rotation_invariance():
    x, y = chain(20, 3), chain(24, 4)
    U = fs.random_rotations(1, seed=9)[0]
    d1, _ = fs.sq_distance(x, y)
    d2, _ = fs.sq_distance(torch.einsum('xy,nyz->nxz', U, x), y)
    assert abs(float(d1) - float(d2)) < 1e-8


def test_zero_insertion_invariance():
    x, y = chain(20, 3), chain(26, 4)
    yz = torch.cat([y[:10], torch.zeros(4, 3, 3, dtype=y.dtype), y[10:], torch.zeros(3, 3, 3, dtype=y.dtype)])
    d1, _ = fs.sq_distance(x, y)
    d2, _ = fs.sq_distance(x, yz)
    assert abs(float(d1) - float(d2)) < 1e-9 and fs.n_edges(yz) == 26


def test_truncation_law():
    """d² between a structure and its prefix = sum of squared norms of the removed terms (0.75 × number of removed edges if all are trans)."""
    x = chain(40, 5, cis=(36,))
    for k in (1, 5, 13):
        d2, _ = fs.sq_distance(x, x[:-k])
        assert abs(float(d2) - float(fs.sq_norm(x[-k:]))) < 1e-8


def test_edge_shrink_law():
    """Distance to the structure with one edge scaled by f = (1 − f) × the norm of that term."""
    x = chain(20, 6)
    for f in (0.0, 0.25, 0.5, 0.9):
        y = x.clone()
        y[8] = f * y[8]
        d, _ = fs.distance(x, y)
        assert abs(float(d) - (1 - f) * float(x[8].norm())) < 1e-8


def test_triangle_inequality():
    S = [chain(n, s) for n, s in ((12, 1), (15, 2), (10, 3), (14, 4))]
    D = np.zeros((4, 4))
    for i in range(4):
        for j in range(i + 1, 4):
            D[i, j] = D[j, i] = float(fs.distance(S[i], S[j])[0])
    for i in range(4):
        for j in range(4):
            for k in range(4):
                assert D[i, k] <= D[i, j] + D[j, k] + 1e-6


def test_straight_line_is_shortest():
    x, y = chain(18, 6), chain(22, 7)
    d2, al = fs.sq_distance(x, y)
    a, b, cols = fs.aligned_columns(x, y, alignment=al)
    assert abs(float((a - b).pow(2).sum()) - float(d2)) < 1e-8
    d = math.sqrt(float(d2))
    for t in (0.25, 0.5, 0.75):
        g = fs.geodesic(a, b, t)
        assert abs(float(fs.distance(x, g)[0]) - t * d) < 1e-6
        assert abs(float(fs.distance(g, y)[0]) - (1 - t) * d) < 1e-6


def test_fixed_is_upper_bound():
    x, y = chain(15, 8), chain(15, 9)
    df = fs.fixed_sq_distance(x[None], y[None])[0, 0]
    do, _ = fs.sq_distance(x, y)
    assert float(do) <= float(df) + 1e-9


def test_scales():
    x, y = chain(15, 8), chain(19, 9)
    d, al = fs.distance(x, y)
    assert 0 <= fs.length_scaled(float(d), 15, 19) <= 1 and fs.sim_long(float(al.score), 15, 19) <= 1


def test_mean_point_of_rotated_copies():
    x = chain(16, 10)
    U = fs.random_rotations(8, seed=11)
    E = torch.einsum('fxy,nyz->fnxz', U, x)
    d2, _ = fs.sq_distance(fs.mean_point(E), x)
    assert float(d2) < 1e-10


def test_mean_point_converges_and_ignores_start():
    """Spread ensemble: converges; different starting structures give the same mean up to one rotation; the mean is a fixed point of align-and-average."""
    x = chain(16, 14)
    g = torch.Generator().manual_seed(15)
    E = torch.stack([torch.einsum('xy,nyz->nxz', fs.random_rotations(1, seed=20 + f)[0],
                                  torch.einsum('nxy,nyz->nxz', x, fs.expmap(0.3 * torch.randn(16, 3, generator=g, dtype=torch.float64))))
                     for f in range(10)])
    ms = []
    for s in (0, 4, 9):
        m, info = fs.mean_point(E, start=s, return_info=True)
        assert info['converged']
        ms.append(m)
    for m in ms[1:]:
        assert float(fs.fixed_sq_distance(ms[0][None], m[None])[0, 0]) < 1e-12
    m = ms[0]
    assert float((fs.align_to_reference(E, m).mean(0) - m).norm()) < 1e-8
    sv = torch.linalg.svdvals(m)
    assert float(sv.max()) <= 0.5 * float(x[:, :, 0].norm(dim=-1).max() * 2) + 1e-12 and float(sv.min()) < 0.49


def test_table_invariance_and_recovery():
    x = chain(20, 12, cis=(4,))
    U = fs.random_rotations(1, seed=13)[0]
    G = fs.relative_table(x)
    assert torch.allclose(G, fs.relative_table(torch.einsum('xy,nyz->nxz', U, x)), atol=1e-12)
    assert torch.allclose(fs.determinants(x), (2 * x[:, :, 0].norm(dim=-1)) ** 3 / 8, atol=1e-12)
    xr = fs.from_table(G)
    assert torch.allclose(fs.relative_table(xr), G, atol=1e-12) and float(fs.sq_distance(xr, x)[0]) < 1e-10
    xz = torch.cat([x[:5], torch.zeros(2, 3, 3, dtype=x.dtype), x[5:]])
    Gz = fs.relative_table(xz)
    assert float(Gz[5:7].abs().max()) == 0 and float(Gz[:, 5:7].abs().max()) == 0


def test_certificate():
    x, y = chain(10, 10, cis=(3,)), chain(12, 11)
    r = fs.bnb(x, y)
    assert r['certified'] and r['S_heur'] <= r['S_ub'] + 1e-9 and r['S_lb'] <= r['S_ub']


def _ensemble(base, F, seed, sigma=0.08):
    """Ensemble (F, n, 3, 3) with small rotational fluctuations on every edge of base and a global rotation per sample."""
    g = torch.Generator().manual_seed(seed)
    W = fs.expmap(sigma * torch.randn(F, len(base), 3, generator=g, dtype=torch.float64))
    E = torch.einsum('nxy,fnyz->fnxz', base, W)
    return torch.einsum('fxy,fnyz->fnxz', fs.random_rotations(F, seed=seed + 7), E)


def test_optimal_plan():
    """The assignment problem and the linear program give the same value; with weights the marginals of the plan are the weights."""
    rng = np.random.default_rng(2)
    C = rng.random((7, 7))
    P1 = fs.optimal_plan(C)
    P2 = fs.optimal_plan(C, np.ones(7), np.ones(7))
    assert abs((P1 * C).sum() - (P2 * C).sum()) < 1e-9
    a, b = rng.random(5), rng.random(8)
    P = fs.optimal_plan(rng.random((5, 8)), a, b)
    assert np.allclose(P.sum(1), a / a.sum(), atol=1e-9) and np.allclose(P.sum(0), b / b.sum(), atol=1e-9)


def test_ensemble_w2():
    """Ensemble W₂: zero to itself, symmetric, unchanged by rotating samples or inserting zero matrices, and the triangle inequality."""
    A = _ensemble(chain(14, 1), 6, 1)
    B = _ensemble(chain(14, 2), 6, 2)
    C = _ensemble(chain(16, 3), 6, 3)
    kw = dict(n_random=31)
    ab = fs.ensemble_w2(A, B, **kw)['W2']
    assert fs.ensemble_w2(A, A, **kw)['W2'] < 1e-6
    assert abs(ab - fs.ensemble_w2(B, A, **kw)['W2']) < 1e-6 * ab
    Ar = torch.einsum('fxy,fnyz->fnxz', fs.random_rotations(6, seed=9), A)
    assert abs(fs.ensemble_w2(Ar, B, **kw)['W2'] - ab) < 1e-6 * ab
    Az = torch.cat([A[:, :5], torch.zeros(6, 2, 3, 3, dtype=A.dtype), A[:, 5:]], 1)
    assert abs(fs.ensemble_w2(Az, B, **kw)['W2'] - ab) < 1e-6 * ab
    ac, bc = fs.ensemble_w2(A, C, **kw)['W2'], fs.ensemble_w2(B, C, **kw)['W2']
    assert ac <= ab + bc + 1e-9 and ab <= ac + bc + 1e-9 and bc <= ab + ac + 1e-9
    w = fs.ensemble_w2(A, B, wa=np.arange(1, 7), cost=fs.ensemble_w2(A, B, **kw)['cost'])
    assert np.allclose(w['plan'].sum(1), np.arange(1, 7) / 21, atol=1e-9)
