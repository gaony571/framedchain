"""자의 묶음(rulers) 시험: 정의와 계산의 일치, 삼각 부등식의 핵심(대응과 자세의 합성), ℓ → ∞ 에서 기본판.
framespace_wip/tests_stage0/test_rulers.py 를 옮긴 것이다. numba 가 없으면 건너뛴다."""
import math

import numpy as np
import pytest
import torch

import framespace as fs
from framespace.geometry import random_rotations

rulers = pytest.importorskip('framespace.rulers', reason='numba 가 필요하다 (pip install framespace[rulers])')
def backbone(n, seed):
    """3.8 Å 걸음의 매끈한 Cα 사슬과 그 좌표 (n−1, 3, 3) (O 는 그럴듯한 쪽에 둔다)."""
    rng = np.random.default_rng(seed)
    d = np.array([1.0, 0.0, 0.0])
    CA = [np.zeros(3)]
    for _ in range(n - 1):
        d = d + 0.6 * rng.normal(size=3)
        d /= np.linalg.norm(d)
        CA.append(CA[-1] + 3.8 * d)
    CA = np.array(CA)
    C = CA + 0.5 * np.roll(CA, -1, 0) - 0.5 * CA + 0.3 * rng.normal(size=CA.shape)
    O = C + 1.2 * np.cross(np.roll(CA, -1, 0) - CA + 1e-3, [0.0, 0.0, 1.0]) / 3.8
    return CA, fs.coordinates_from_backbone(CA, C, O)


def moved(P, seed):
    R = random_rotations(1, seed=seed).numpy()[0]
    return P @ R.T + np.array([5.0, -3.0, 2.0])


def random_match(n, m, rng):
    k = rng.integers(1, min(n, m) + 1)
    i = np.sort(rng.choice(n, k, replace=False))
    j = np.sort(rng.choice(m, k, replace=False))
    M = np.zeros((n, m), bool)
    M[i, j] = True
    return M


def random_pose(seed):
    U = random_rotations(1, seed=seed)[0]
    return U, torch.tensor(np.random.default_rng(seed).normal(size=3) * 3.0)


def compose(M1, M2):
    """대응 A–B 와 B–C 를 이어 A–C."""
    return (M1.astype(int) @ M2.astype(int)) > 0


def test_tm_identity_and_self():
    P, _ = backbone(30, 1)
    Q, _ = backbone(24, 2)
    r = rulers.tm_align(P, Q, d0=3.0)
    assert abs(r['D2'] - rulers.tm_cost(P, Q, r['match'], r['U'], r['t'], 3.0)) < 1e-9
    z = rulers.tm_align(P, moved(P, 3), d0=3.0)
    assert z['D2'] < 1e-8 and abs(z['S'] - 30) < 1e-8


def test_tm_composition_inequality():
    """정리의 핵심: 임의의 대응·자세 (π₁, g₁), (π₂, g₂) 를 이은 (π₂∘π₁, g₁g₂) 의 비용 제곱근은 둘의 합 이하."""
    rng = np.random.default_rng(0)
    for trial in range(200):
        A, B, C = (backbone(int(rng.integers(8, 20)), 10 * trial + k)[0] for k in range(3))
        d0 = float(rng.uniform(0.5, 6.0))
        M1, M2 = random_match(len(A), len(B), rng), random_match(len(B), len(C), rng)
        (U1, t1), (U2, t2) = random_pose(3 * trial), random_pose(3 * trial + 1)
        U, t = U1 @ U2, U1 @ t2 + t1
        lhs = math.sqrt(rulers.tm_cost(A, C, compose(M1, M2), U, t, d0))
        rhs = math.sqrt(rulers.tm_cost(A, B, M1, U1, t1, d0)) + math.sqrt(rulers.tm_cost(B, C, M2, U2, t2, d0))
        assert lhs <= rhs + 1e-9


def test_wl_composition_inequality():
    rng = np.random.default_rng(1)
    for trial in range(100):
        (_, A), (_, B), (_, C) = (backbone(int(rng.integers(8, 20)), 10 * trial + k) for k in range(3))
        ell, w = float(rng.uniform(0.5, 30.0)), float(rng.uniform(0.3, 2.0))
        M1, M2 = random_match(len(A), len(B), rng), random_match(len(B), len(C), rng)
        (U1, t1), (U2, t2) = random_pose(3 * trial), random_pose(3 * trial + 1)
        U, t = U1 @ U2, U1 @ t2 + t1
        lhs = math.sqrt(rulers.wl_cost(A, C, compose(M1, M2), U, t, ell, w))
        rhs = math.sqrt(rulers.wl_cost(A, B, M1, U1, t1, ell, w)) + math.sqrt(rulers.wl_cost(B, C, M2, U2, t2, ell, w))
        assert lhs <= rhs + 1e-9


def test_wl_identity_self_and_limit():
    _, x = backbone(30, 4)
    _, y = backbone(26, 5)
    r = rulers.wl_align(x, y, ell=6.0, w=0.7)
    assert abs(r['D2'] - rulers.wl_cost(x, y, r['match'], r['U'], r['t'], 6.0, 0.7)) < 1e-8
    Rm = random_rotations(1, seed=9)[0].numpy()
    z = rulers.wl_align(x, np.einsum('ab,nbc->nac', Rm, x), ell=4.0, p=rulers.edge_positions(x), q=rulers.edge_positions(x) @ Rm.T - 2.0)
    assert z['D2'] < 1e-8
    base, _ = fs.sq_distance(torch.as_tensor(x), torch.as_tensor(y))
    inf = rulers.wl_align(x, y, ell=math.inf)
    assert abs(inf['D2'] - float(base)) < 1e-8
    big = rulers.wl_align(x, y, ell=1e5)
    assert abs(big['D2'] - float(base)) < 1e-3


def test_optimized_triangle_small():
    """작은 사슬 8 개의 최적화 값으로 삼각 부등식 (고정 d₀ 와 고정 ℓ). 계산값은 상계라 위반은 최적화 오차뿐이다."""
    S = [backbone(int(n), 20 + k) for k, n in enumerate([14, 16, 18, 15, 20, 17, 13, 19])]
    N = len(S)
    Dt, Dw = np.zeros((N, N)), np.zeros((N, N))
    for a in range(N):
        for b in range(a + 1, N):
            Dt[a, b] = Dt[b, a] = math.sqrt(rulers.tm_align(S[a][0], S[b][0], d0=2.5, keep=None)['D2'])
            Dw[a, b] = Dw[b, a] = math.sqrt(rulers.wl_align(S[a][1], S[b][1], ell=5.0, keep=None)['D2'])
    assert rulers.triangle_rate(Dt, rel_tol=1e-6)[1] == 0
    assert rulers.triangle_rate(Dw, rel_tol=1e-6)[1] == 0


def test_linear_form_is_not_a_metric():
    """제곱근을 뺀 꼴 D² = (n + m)/2 − S (Sippl 2008 의 L_a + L_b − 2S 에 S_TM 을 넣은 것의 절반)는 삼각 부등식을 어긴다.
    a 의 10곳을 0.5 d₀ 옮긴 b, 같은 방향으로 두 배 옮긴 c: 짝마다 δ²(2r) > 2 δ²(r) (r² < 1/2) 라 선형 꼴은 깨지고 제곱근 꼴은 선다.
    ab, bc 는 항등 대응의 비용(참값의 상계)을 써서 위반을 작게 잡는다. ac 는 최적화 값(상계)이라, 참 최적이 이보다 15% 넘게 낮지 않으면 위반이다."""
    A, _ = backbone(60, 31)
    rng = np.random.default_rng(5)
    idx = np.linspace(3, 56, 10).round().astype(int)
    u = rng.normal(size=(10, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    d0 = 4.0
    B, C = A.copy(), A.copy()
    B[idx] += 0.5 * d0 * u
    C[idx] += 1.0 * d0 * u
    I, E, z = np.eye(60, dtype=bool), np.eye(3), np.zeros(3)
    ab, bc = rulers.tm_cost(A, B, I, E, z, d0), rulers.tm_cost(B, C, I, E, z, d0)
    ac = rulers.tm_align(A, C, d0)['D2']
    assert ac - (ab + bc) > 0.15 * ac
    assert math.sqrt(ac) < math.sqrt(rulers.tm_align(A, B, d0)['D2']) + math.sqrt(rulers.tm_align(B, C, d0)['D2'])


def test_tm_forms_and_d0():
    f = rulers.tm_forms(S=10.0, n=20, m=30)
    assert abs(f['lin_raw'] - 15.0) < 1e-12 and abs(f['sqrt_raw'] ** 2 - 15.0) < 1e-12
    assert abs(f['lin_norm'] - 0.6) < 1e-12 and abs(f['sqrt_norm'] ** 2 - 0.6) < 1e-12
    assert abs(rulers.d0_tm(100) - (1.24 * 85 ** (1 / 3) - 1.8)) < 1e-12 and rulers.d0_tm(10) == 0.5
    assert rulers.triangle_rate(np.array([[0, 1, 3], [1, 0, 1], [3, 1, 0.0]]))[1] == 1


def test_zero_padding_invariance():
    """길이 n 의 공간은 더 긴 공간의 부분공간이다: 영 항을 덧붙이거나 끼워도 값이 같다(기본 판, D_{w,ℓ})."""
    _, x = backbone(30, 7)
    _, y = backbone(25, 8)
    rng = np.random.default_rng(3)

    def ins(a, k):
        a = list(a)
        for _ in range(k):
            a.insert(int(rng.integers(0, len(a) + 1)), np.zeros((3, 3)))
        return np.array(a)

    base, _ = fs.sq_distance(torch.as_tensor(x), torch.as_tensor(y))
    xz, yz = ins(x, 20), ins(y, 35)
    padded, _ = fs.sq_distance(torch.as_tensor(xz), torch.as_tensor(yz))
    assert abs(float(base) - float(padded)) < 1e-8
    p, q = rulers.edge_positions(x), rulers.edge_positions(y)
    keep_x, keep_y = (xz ** 2).sum((1, 2)) > 0, (yz ** 2).sum((1, 2)) > 0
    pz, qz = np.zeros((len(xz), 3)), np.zeros((len(yz), 3))
    pz[keep_x], qz[keep_y] = p, q
    pz[~keep_x], qz[~keep_y] = 50.0, -50.0    # 영 항의 자리는 아무 데나: 값에 영향이 없어야 한다
    a = rulers.wl_align(x, y, ell=6.0, p=p, q=q)
    b = rulers.wl_align(xz, yz, ell=6.0, p=pz, q=qz)
    assert abs(a['D2'] - b['D2']) < 1e-8
    assert b['match'].shape == (len(xz), len(yz)) and not b['match'][~keep_x].any() and not b['match'][:, ~keep_y].any()
