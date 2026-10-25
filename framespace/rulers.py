"""자의 묶음 (정의역 문서 5절, 결과 문서 1.1·2.1): 위치 항을 넣은 D_{w,ℓ} 와 고정 d₀ 의 TM 꼴 거리 D_TM.

공통 틀(정의역 문서 2.2절 일반 보조정리). 항 a 의 특징과 그 사이의 거리 δ 가 강체 운동 G 에 불변이고,
빈칸 비용 c 가 G 불변이며 c(a) ≤ δ(a, b) + c(b) 이면
    D² = min over (순서를 지키는 부분 대응 π, g ∈ G) of Σ_{(i,j)∈π} δ(a_i, g b_j)² + Σ_{짝 없음} c²
는 (빈칸 비용 0 인 항을 끼우고 빼는 것과 공통 g 를 묶은) 몫 위의 거리다.

D_{w,ℓ}. 변 i 의 특징 (w·x_i, p_i/ℓ) (x_i = s_i R_i/2, p_i 는 변의 가운데 점), δ 는 그 유클리드 거리, c = w‖x‖.
    D² = w²(‖X‖² + ‖Y‖²) − S,  S = max Σ_{(i,j)∈π} [2w²⟨x_i, U y_j⟩ − ‖p_i − U q_j − t‖²/ℓ²].
  ℓ → ∞ 이면 기본판(framespace.sq_distance)에 w² 를 곱한 것이다. (w, ℓ) 의 값은 w²·D²(1, wℓ) 이므로 실질 매개변수는 wℓ 하나.
D_TM. 잔기 i 의 특징 P_i (Cα 자리), δ(a, b) = f(‖a − b‖/d₀), f(r) = r/√(1 + r²) (오목·증가·f(0) = 0 이라 거리), c² = 1/2.
    1 − δ² = 1/(1 + r²) 이 TM 의 잔기 점수이므로 D_TM² = (n + m)/2 − S_TM,  S_TM = max Σ_{(i,j)∈π} 1/(1 + r_ij²).
  d₀ 를 고정하면 거리다(TM-align 의 목적 함수 = d₀ 고정 D_TM). TM-score 는 d₀ = d₀(L), 길이로 나눔, 1 − S 변환이 더해진 것이고,
  1 − TM 의 비거리성은 짧은 쪽 정규화와 1 − S 변환에서 온다(결과 문서 2.1 TM 요인 설계).

계산. 자세(U, t)를 고정하면 대응은 동적계획법으로 정확하다. 대응을 고정하면
  D_{w,ℓ}: t 는 짝지은 가운데 점의 중심 차, U 는 합친 교차 공분산의 SVD 로 정확하다.
  D_TM: 가중 최소제곱(가중 1/(1 + r²)², 오목 함수의 위쪽 접선으로 줄이기)이라 값이 줄지 않는다.
번갈아 풀며 여러 자세에서 출발해 가장 좋은 것을 고른다. 출발: 조각 쌍 맞춤(TM-align 의 국소 맞춤), 틈 없는 겹침,
(search) 찾은 대응의 부분 조각 맞춤(TM-score 프로그램의 탐색), 밖에서 준 자세(예: TM-align 의 자세).
계산값은 D 의 상계이고 보증(분기 한정)은 없다. 핵심 고리는 numba 로 돈다(pip install framespace[rulers]).
이 모듈은 공통 정의역 논문의 실행본 framespace_wip/stage0/rulers.py 와 같은 계산이다(관행 열의 TM-align 재구현, 6 절 시간 표).
"""
import math

import numpy as np
from numba import njit

from .frames import ca_trace


def edge_positions(x):
    """좌표 (n, 3, 3) -> 변의 가운데 점 (n, 3) (Å, 첫 Cα 가 원점)."""
    ca = ca_trace(np.asarray(x))
    return 0.5 * (ca[..., 1:, :] + ca[..., :-1, :])


def d0_tm(L):
    """TM-score 의 길이 척도 d₀(L) = 1.24 (L − 15)^{1/3} − 1.8 (Å), TM-align 처럼 0.5 아래로는 내리지 않는다."""
    return max(0.5, 1.24 * (L - 15) ** (1 / 3) - 1.8) if L > 15 else 0.5


@njit(cache=True)
def _dp(K):
    """순서를 지키는 1대1 부분 대응의 최대 점수와 짝 (되짚기). K (n, m)."""
    n, m = K.shape
    F = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            v = F[i - 1, j - 1] + K[i - 1, j - 1]
            a = F[i - 1, j]
            b = F[i, j - 1]
            if a > v:
                v = a
            if b > v:
                v = b
            F[i, j] = v
    ii = np.empty(min(n, m), np.int64)
    jj = np.empty(min(n, m), np.int64)
    k = 0
    i, j = n, m
    while i > 0 and j > 0:
        if F[i, j] == F[i - 1, j]:
            i -= 1
        elif F[i, j] == F[i, j - 1]:
            j -= 1
        else:
            ii[k] = i - 1
            jj[k] = j - 1
            k += 1
            i -= 1
            j -= 1
    return F[n, m], ii[:k][::-1].copy(), jj[:k][::-1].copy()


@njit(cache=True)
def _procrustes(M):
    """argmax_{U ∈ SO(3)} tr(U M)."""
    W, S, Vh = np.linalg.svd(M)
    V = Vh.T
    d = np.linalg.det(V @ W.T)
    D = np.eye(3)
    D[2, 2] = 1.0 if d >= 0 else -1.0
    return V @ D @ W.T


@njit(cache=True)
def _kabsch(P, Q, ii, jj, w):
    """짝 (ii, jj), 가중 w 로 P_i ← U Q_j + t 의 최소제곱."""
    ws = w.sum()
    pc = np.zeros(3)
    qc = np.zeros(3)
    for k in range(len(ii)):
        pc += w[k] * P[ii[k]]
        qc += w[k] * Q[jj[k]]
    pc /= ws
    qc /= ws
    M = np.zeros((3, 3))
    for k in range(len(ii)):
        M += w[k] * np.outer(Q[jj[k]] - qc, P[ii[k]] - pc)
    U = _procrustes(M)
    return U, pc - U @ qc


@njit(cache=True)
def _tm_K(P, Q, U, t, inv):
    n, m = len(P), len(Q)
    Y = Q @ U.T
    K = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            d0 = P[i, 0] - Y[j, 0] - t[0]
            d1 = P[i, 1] - Y[j, 1] - t[1]
            d2 = P[i, 2] - Y[j, 2] - t[2]
            K[i, j] = 1.0 / (1.0 + (d0 * d0 + d1 * d1 + d2 * d2) * inv)
    return K


@njit(cache=True)
def _tm_refine(P, Q, U, t, inv, iters, irls, tol):
    """한 출발 자세에서 [DP → 가중 최소제곱] 을 번갈아. -> (S, U, t, ii, jj)."""
    S, ii, jj = _dp(_tm_K(P, Q, U, t, inv))
    for _ in range(iters):
        if len(ii) < 3:
            break
        Un, tn = U, t
        for _r in range(irls):
            Y = Q[jj] @ Un.T + tn
            r2 = ((P[ii] - Y) ** 2).sum(1) * inv
            Un, tn = _kabsch(P, Q, ii, jj, 1.0 / (1.0 + r2) ** 2)
        Sn, iin, jjn = _dp(_tm_K(P, Q, Un, tn, inv))
        if Sn <= S + tol * (1.0 + abs(S)):
            break
        S, U, t, ii, jj = Sn, Un, tn, iin, jjn
    return S, U, t, ii, jj


@njit(cache=True)
def _wl_K(x, p, y, q, U, t, w2, inv):
    n, m = len(x), len(y)
    K = np.empty((n, m))
    Uy = np.empty((m, 3, 3))
    for j in range(m):
        Uy[j] = U @ y[j]
    Yq = q @ U.T
    for i in range(n):
        for j in range(m):
            s = 0.0
            for a in range(3):
                for b in range(3):
                    s += x[i, a, b] * Uy[j, a, b]
            v = 2.0 * w2 * s
            if inv > 0:
                d0 = p[i, 0] - Yq[j, 0] - t[0]
                d1 = p[i, 1] - Yq[j, 1] - t[1]
                d2 = p[i, 2] - Yq[j, 2] - t[2]
                v -= inv * (d0 * d0 + d1 * d1 + d2 * d2)
            K[i, j] = v
    return K


@njit(cache=True)
def _wl_pose(x, p, y, q, ii, jj, w2, inv):
    k = len(ii)
    pc = np.zeros(3)
    qc = np.zeros(3)
    for a in range(k):
        pc += p[ii[a]]
        qc += q[jj[a]]
    pc /= k
    qc /= k
    M = np.zeros((3, 3))
    for a in range(k):
        M += w2 * (y[jj[a]] @ x[ii[a]].T)
        if inv > 0:
            M += inv * np.outer(q[jj[a]] - qc, p[ii[a]] - pc)
    U = _procrustes(M)
    return U, pc - U @ qc


@njit(cache=True)
def _wl_refine(x, p, y, q, U, t, w2, inv, iters, tol):
    S, ii, jj = _dp(_wl_K(x, p, y, q, U, t, w2, inv))
    for _ in range(iters):
        if len(ii) < 1:
            break
        Un, tn = _wl_pose(x, p, y, q, ii, jj, w2, inv)
        Sn, iin, jjn = _dp(_wl_K(x, p, y, q, Un, tn, w2, inv))
        if Sn <= S + tol * (1.0 + abs(S)):
            break
        S, U, t, ii, jj = Sn, Un, tn, iin, jjn
    return S, U, t, ii, jj


def _rand_rot(k, seed):
    q = np.random.default_rng(seed).normal(size=(k, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, a, b, c = q.T
    return np.stack([1 - 2 * (b * b + c * c), 2 * (a * b - c * w), 2 * (a * c + b * w),
                     2 * (a * b + c * w), 1 - 2 * (a * a + c * c), 2 * (b * c - a * w),
                     2 * (a * c - b * w), 2 * (b * c + a * w), 1 - 2 * (a * a + b * b)], -1).reshape(k, 3, 3)


def start_poses(P, Q, f=8, per_side=12, n_thread=24, n_random=8, seed=0):
    """출발 자세 [(U, t)]: 길이 f 조각 쌍의 맞춤(한 쪽에 약 per_side 개 시작점), 틈 없는 겹침 n_thread 개, 중심을 맞춘 무작위 회전."""
    n, m = len(P), len(Q)
    f = min(f, n, m)
    out = []
    one = np.ones(f)
    for i in np.unique(np.linspace(0, n - f, per_side).round().astype(int)):
        for j in np.unique(np.linspace(0, m - f, per_side).round().astype(int)):
            out.append(_kabsch(P, Q, np.arange(i, i + f), np.arange(j, j + f), one))
    for o in np.unique(np.linspace(-(m - f), n - f, n_thread).round().astype(int)):
        i = np.arange(max(0, o), min(n, m + o))
        if len(i) >= 3:
            out.append(_kabsch(P, Q, i, i - o, np.ones(len(i))))
    for R in _rand_rot(n_random, seed):
        out.append((R, P.mean(0) - R @ Q.mean(0)))
    return out


def sub_fragment_poses(P, Q, ii, jj, min_len=4, n_pos=8):
    """찾은 대응의 연속 부분(길이 k, k/2, k/4, …, min_len 까지, 자리마다 n_pos 개)을 맞춘 자세 (TM-score 의 탐색)."""
    out = []
    k = len(ii)
    L = k
    while L >= min_len:
        for s in np.unique(np.linspace(0, k - L, n_pos).round().astype(int)):
            out.append(_kabsch(P, Q, ii[s:s + L], jj[s:s + L], np.ones(L)))
        L //= 2
    return out


def _match(n, m, ii, jj):
    M = np.zeros((n, m), bool)
    M[ii, jj] = True
    return M


def _run(refine, poses, keep, score0):
    """모든 출발을 한 번 평가(score0)한 뒤 점수가 높은 keep 개를 다듬는다."""
    s = np.array([score0(U, t) for U, t in poses])
    order = np.argsort(-s)[:keep] if keep else np.arange(len(poses))
    best = None
    for k in order:
        r = refine(*poses[k])
        if best is None or r[0] > best[0]:
            best = r
    return best


def tm_align(P, Q, d0, poses=None, extra_poses=(), iters=50, keep=16, irls=3, search=True, tol=1e-10):
    """고정 d₀ 의 S_TM = max Σ_{짝} 1/(1 + ‖P_i − U Q_j − t‖²/d₀²). P (n, 3), Q (m, 3) Cα (Å).
    poses: 출발 자세 목록(주지 않으면 start_poses), extra_poses: 더할 자세(예: TM-align 의 자세), search: 찾은 대응의 부분 조각 탐색을 한 번 더.
    -> dict(S, D2 = (n + m)/2 − S, U, t, match (n, m) bool)."""
    P, Q = np.ascontiguousarray(P, np.float64), np.ascontiguousarray(Q, np.float64)
    inv = 1.0 / d0 ** 2
    poses = list(start_poses(P, Q) if poses is None else poses) + [(np.asarray(U, float), np.asarray(t, float)) for U, t in extra_poses]
    ref = lambda U, t: _tm_refine(P, Q, U, t, inv, iters, irls, tol)
    sc0 = lambda U, t: _dp(_tm_K(P, Q, U, t, inv))[0]
    best = _run(ref, poses, keep, sc0)
    if search and len(best[3]) >= 4:
        r = _run(ref, sub_fragment_poses(P, Q, best[3], best[4]), keep, sc0)
        if r[0] > best[0]:
            best = r
    S, U, t, ii, jj = best
    return dict(S=float(S), D2=max(0.0, 0.5 * (len(P) + len(Q)) - float(S)), U=U, t=t, match=_match(len(P), len(Q), ii, jj))


def tm_cost(P, Q, match, U, t, d0):
    """주어진 대응과 자세에서의 D_TM² (정의대로): Σ_짝 r²/(1 + r²) + (짝 없는 항의 수)/2."""
    P, Q, U, t = (np.asarray(a, np.float64) for a in (P, Q, U, t))
    i, j = np.nonzero(np.asarray(match))
    r2 = ((P[i] - (Q[j] @ U.T + t)) ** 2).sum(-1) / d0 ** 2
    return float((r2 / (1 + r2)).sum()) + 0.5 * (len(P) + len(Q) - 2 * len(i))


def wl_align(x, y, ell, w=1.0, p=None, q=None, poses=None, extra_poses=(), iters=50, keep=16, search=True, tol=1e-10):
    """D_{w,ℓ}² 와 정렬. x (n, 3, 3), y (m, 3, 3) 좌표, p, q 는 변의 가운데 점(주지 않으면 좌표에서 만든다).
    ell = inf 이면 위치 항이 없다(기본판 × w²; 출발은 항등 + 무작위 회전 127 개). 영 항은 미리 뺀다(값은 같다). -> dict(S, D2, U, t, match: 원래 번호의 (n, m))."""
    x, y = np.ascontiguousarray(x, np.float64), np.ascontiguousarray(y, np.float64)
    p = np.ascontiguousarray(edge_positions(x) if p is None else p, np.float64)
    q = np.ascontiguousarray(edge_positions(y) if q is None else q, np.float64)
    # 영 항(길이 0 인 변)은 빈칸 비용이 0 이고 짝지어도 이득이 0 이하라 값을 바꾸지 않는다(같은 점). 출발 자세를 흐리지 않게 미리 뺀다.
    kx, ky = (x ** 2).sum((1, 2)) > 1e-12, (y ** 2).sum((1, 2)) > 1e-12
    x, p, y, q = (np.ascontiguousarray(a) for a in (x[kx], p[kx], y[ky], q[ky]))
    w2 = w * w
    inv = 0.0 if math.isinf(ell) else 1.0 / ell ** 2
    if poses is None:
        if inv == 0.0:
            poses = [(R, np.zeros(3)) for R in np.concatenate([np.eye(3)[None], _rand_rot(127, 0)])]
        else:
            poses = start_poses(p, q)
    poses = list(poses) + [(np.asarray(U, float), np.asarray(t, float)) for U, t in extra_poses]
    ref = lambda U, t: _wl_refine(x, p, y, q, U, t, w2, inv, iters, tol)
    sc0 = lambda U, t: _dp(_wl_K(x, p, y, q, U, t, w2, inv))[0]
    best = _run(ref, poses, keep if inv else None, sc0)
    if search and inv and len(best[3]) >= 4:
        r = _run(ref, sub_fragment_poses(p, q, best[3], best[4]), keep, sc0)
        if r[0] > best[0]:
            best = r
    S, U, t, ii, jj = best
    D2 = max(0.0, w2 * float((x ** 2).sum() + (y ** 2).sum()) - float(S))
    ix, iy = np.nonzero(kx)[0], np.nonzero(ky)[0]
    return dict(S=float(S), D2=D2, U=U, t=t, match=_match(len(kx), len(ky), ix[ii], iy[jj]))


def wl_cost(x, y, match, U, t, ell, w=1.0, p=None, q=None):
    """주어진 대응과 자세에서의 D_{w,ℓ}² (정의대로)."""
    x, y, U, t = (np.asarray(a, np.float64) for a in (x, y, U, t))
    p = np.asarray(edge_positions(x) if p is None else p, np.float64)
    q = np.asarray(edge_positions(y) if q is None else q, np.float64)
    i, j = np.nonzero(np.asarray(match))
    mi = np.setdiff1d(np.arange(len(x)), i)
    mj = np.setdiff1d(np.arange(len(y)), j)
    c = w * w * float(((x[i] - np.einsum('ab,nbc->nac', U, y[j])) ** 2).sum())
    if not math.isinf(ell):
        c += float(((p[i] - (q[j] @ U.T + t)) ** 2).sum()) / ell ** 2
    return c + w * w * float((x[mi] ** 2).sum() + (y[mj] ** 2).sum())


# H-U2 의 요인 설계: (d₀ 고정 / d₀(L)) × (나누지 않음 / L̄ = (n + m)/2 로 나눔) × (제곱근 거리꼴 / 1 − S 꼴).
def tm_forms(S, n, m):
    """S_TM (해당 d₀ 로 계산한 값) -> 네 가지 꼴. 'sqrt_raw' 만 정리로 거리다(d₀ 고정일 때)."""
    Lb = 0.5 * (n + m)
    D2 = max(0.0, Lb - S)
    return {'sqrt_raw': math.sqrt(D2), 'lin_raw': D2, 'sqrt_norm': math.sqrt(D2 / Lb), 'lin_norm': 1.0 - S / Lb}


def triangle_rate(Dm, rel_tol=1e-9, abs_tol=1e-9, max_triples=None, seed=0):
    """대칭 행렬 Dm (N, N) 에서 세 점 묶음마다 가장 긴 변이 나머지 둘의 합보다 큰지. -> (위반 비율, 위반 수, 묶음 수, 최대 상대 위반).
    상대 위반 = (긴 변 − 합) / 긴 변 (D 에 대한 상대값). 긴 변 − 합 > abs_tol + rel_tol·긴 변 이면 위반. max_triples 를 주면 무작위로 그만큼만 본다."""
    Dm = np.asarray(Dm, float)
    N = len(Dm)
    if max_triples is None:
        a, b, c = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)]).T
    else:
        rng = np.random.default_rng(seed)
        T = np.sort(np.array([rng.choice(N, 3, replace=False) for _ in range(max_triples)]), 1)
        a, b, c = T.T
    e = np.stack([Dm[a, b], Dm[b, c], Dm[a, c]], 1)
    lo = np.sort(e, 1)
    s = lo[:, 0] + lo[:, 1]
    v = lo[:, 2] - s
    bad = v > abs_tol + rel_tol * lo[:, 2]
    rel = np.where(lo[:, 2] > 0, v / np.maximum(lo[:, 2], 1e-300), 0.0)
    return float(bad.mean()), int(bad.sum()), len(v), float(rel.max())
