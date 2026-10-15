"""SQFORM_SPACE.md 의 확정된 정의와 성질 시험. 합성 프레임 열로 잰다 (pytest)."""
import math

import numpy as np
import torch

import framespace as fs

LOC_C, LOC_O = np.array([1.42, 0.53, 0.01]), np.array([1.65, 1.74, 0.01])   # 트랜스 틀 (1.4절 표)


def chain(n, seed, noise=0.15, cis=()):
    """이웃 변이 서로 비슷한 회전을 이루는 실제 구조의 좌표 (n, 3, 3) = s R / 2. 트랜스 s ≈ 1 ± 0.016, cis 자리는 s = 0.81."""
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
    """원자 -> 좌표 -> Cα: 좌표만으로 Cα 가 정확히 복원되고 길이·시스가 읽힌다 (1.2, 1.4절)."""
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
    """l0 와 시스 경계를 바꿀 수 있다. l0 를 바꿔도 Cα 복원은 같은 l0 로 하면 그대로다."""
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
    """구조와 그 앞부분 사이의 d² = 잘라낸 항의 노름² 합 (2.5절. 트랜스만이면 0.75 × 잘라낸 변 수)."""
    x = chain(40, 5, cis=(36,))
    for k in (1, 5, 13):
        d2, _ = fs.sq_distance(x, x[:-k])
        assert abs(float(d2) - float(fs.sq_norm(x[-k:]))) < 1e-8


def test_edge_shrink_law():
    """변 하나를 f 배로 줄인 구조와의 거리 = (1 − f) × 그 항의 노름 (2.3절)."""
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
