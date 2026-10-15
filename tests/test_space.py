"""공간의 확정된 성질 시험. 실제 단백질 자료 없이 합성 프레임 열로 잰다 (pytest)."""
import math

import numpy as np
import torch

import framespace as fs

# 시험용 원자 틀(변의 좌표계, Å): C_i, O_i 의 자리
LOC_C, LOC_O = np.array([1.42, 0.53, 0.01]), np.array([1.64, 1.74, 0.01])


def helixish(n, seed, noise=0.15):
    """이웃 변이 서로 비슷한 회전을 이루는 열 (n, 3, 3) = R/2."""
    g = torch.Generator().manual_seed(seed)
    step = fs.expmap(torch.tensor([0.3, 1.2, 0.4], dtype=torch.float64))
    R = [torch.eye(3, dtype=torch.float64)]
    for _ in range(n - 1):
        R.append(R[-1] @ step @ fs.expmap(noise * torch.randn(3, generator=g, dtype=torch.float64)))
    return torch.stack(R) / 2


def atoms_from(X, l):
    """시험용: 좌표와 길이로 잔기마다 N, CA, C, O (L = n + 1)."""
    X = np.asarray(X)
    R = 2 * X
    CA = fs.ca_trace(X, l)
    C = CA[:-1] + R @ LOC_C
    O = CA[:-1] + R @ LOC_O
    C = np.concatenate([C, C[-1:] + 1.0])
    O = np.concatenate([O, O[-1:] + 1.0])
    return CA - 1.0, CA, C, O


def test_quaternion_roundtrip():
    R = fs.random_rotations(200, seed=1)
    assert torch.allclose(fs.quat_to_mat(fs.mat_to_quat(R)), R, atol=1e-10)


def test_procrustes_and_polar():
    U = fs.random_rotations(5, seed=2)
    assert torch.allclose(fs.procrustes(U.transpose(-1, -2) * 3.0), U, atol=1e-10)
    R, s = fs.polar(0.5 * U)
    assert torch.allclose(R, U, atol=1e-10) and torch.allclose(s, torch.full((5,), 0.5, dtype=torch.float64))


def test_frames_roundtrip_and_pdb(tmp_path):
    X = helixish(30, 3).numpy()
    l = np.full(30, 3.8)
    N, CA, C, O = atoms_from(X, l)
    R, l2 = fs.frames_from_backbone(N, CA, C, O)
    assert np.allclose(l2, l) and np.allclose(fs.embed(R), X, atol=1e-10) and fs.is_real(X).all()
    p = tmp_path / 'x.pdb'
    with open(p, 'w') as f:
        k = 0
        for i in range(len(CA)):
            for name, xyz in (('N', N[i]), ('CA', CA[i]), ('C', C[i]), ('O', O[i])):
                k += 1
                f.write('ATOM  %5d  %-3s ALA A%4d    %8.3f%8.3f%8.3f  1.00  0.00\n' % (k, name, i + 1, *xyz))
    Xp, lp = fs.from_pdb(str(p))
    assert Xp.shape == (1, 30, 3, 3) and np.allclose(Xp[0], X, atol=2e-3) and np.allclose(lp[0], l, atol=2e-3)


def test_self_and_symmetry():
    A, B = helixish(25, 1), helixish(31, 2)
    d0, _ = fs.distance(A, A)
    assert float(d0) < 1e-6
    dab, _ = fs.distance(A, B, n_random=63)
    dba, _ = fs.distance(B, A, n_random=63)
    assert abs(float(dab) - float(dba)) < 1e-6


def test_rotation_invariance():
    A, B = helixish(20, 3), helixish(24, 4)
    U = fs.random_rotations(1, seed=9)[0]
    d1, _ = fs.sq_distance(A, B, n_random=63)
    d2, _ = fs.sq_distance(torch.einsum('xy,nyz->nxz', U, A), B, n_random=63)
    assert abs(float(d1) - float(d2)) < 1e-8


def test_zero_padding_invariance():
    A, B = helixish(20, 3), helixish(26, 4)
    Bp = torch.cat([B, torch.zeros(7, 3, 3, dtype=B.dtype)])
    d1, _ = fs.sq_distance(A, B, n_random=31)
    d2, _ = fs.sq_distance(A, Bp, n_random=31)
    assert abs(float(d1) - float(d2)) < 1e-9


def test_truncation_law():
    """A 와 A 의 앞 n−k 변 사이: d² = 0.75 k (정확히)."""
    A = helixish(40, 5)
    for k in (1, 5, 13):
        d2, _ = fs.sq_distance(A, A[:-k])
        assert abs(float(d2) - 0.75 * k) < 1e-8


def test_triangle_inequality():
    S = [helixish(n, s) for n, s in ((12, 1), (15, 2), (10, 3), (14, 4))]
    D = np.zeros((4, 4))
    for i in range(4):
        for j in range(i + 1, 4):
            D[i, j] = D[j, i] = float(fs.distance(S[i], S[j], n_random=127)[0])
    for i in range(4):
        for j in range(4):
            for k in range(4):
                assert D[i, k] <= D[i, j] + D[j, k] + 1e-6


def test_geodesic_is_straight():
    A, B = helixish(18, 6), helixish(22, 7)
    d2, al = fs.sq_distance(A, B, n_random=63)
    Ca, Cb, cols = fs.aligned_columns(A, B, alignment=al)
    assert abs(float((Ca - Cb).pow(2).sum()) - float(d2)) < 1e-8
    for t in (0.25, 0.5, 0.8):
        G = fs.geodesic(Ca, Cb, t)
        assert abs(float((G - Ca).pow(2).sum().sqrt()) - t * math.sqrt(float(d2))) < 1e-8
        assert abs(float((G - Cb).pow(2).sum().sqrt()) - (1 - t) * math.sqrt(float(d2))) < 1e-8


def test_fixed_vs_optimal():
    A, B = helixish(15, 8), helixish(15, 9)
    k = np.arange(15)
    df = fs.fixed_sq_distance(A[None], B[None], k, k)[0, 0]
    do, _ = fs.sq_distance(A, B, n_random=63)
    assert float(do) <= float(df) + 1e-9


def test_mean_point_of_rotated_copies():
    A = helixish(16, 10)
    U = fs.random_rotations(8, seed=11)
    E = torch.einsum('fxy,nyz->fnxz', U, A)
    M = fs.mean_point(E)
    d2, _ = fs.sq_distance(M, A)
    assert float(d2) < 1e-10


def test_certificate_small():
    A, B = helixish(10, 10), helixish(12, 11)
    r = fs.bnb(A, B)
    assert r['certified'] and r['S_heur'] <= r['S_ub'] + 1e-9 and r['S_lb'] <= r['S_ub']
