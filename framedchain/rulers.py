"""The ruler family: D_{w,ℓ} with a position term, and the TM-form distance D_TM at fixed d₀.

Common frame (the general lemma of the paper). If the features of terms a and the distance δ between them are
invariant under the rigid motions G, the gap cost c is G-invariant, and c(a) ≤ δ(a, b) + c(b), then
    D² = min over (order-preserving partial correspondence π, g ∈ G) of Σ_{(i,j)∈π} δ(a_i, g b_j)² + Σ_{unmatched} c²
is a metric on the quotient (identifying insertion/removal of zero-cost terms and a common g).

D_{w,ℓ}. Feature of edge i: (w·x_i, p_i/ℓ) (x_i = s_i R_i/2, p_i the midpoint of the edge); δ is their Euclidean
distance and c = w‖x‖.
    D² = w²(‖X‖² + ‖Y‖²) − S,  S = max Σ_{(i,j)∈π} [2w²⟨x_i, U y_j⟩ − ‖p_i − U q_j − t‖²/ℓ²].
  As ℓ → ∞ this is the basic distance (framedchain.sq_distance) times w². The value at (w, ℓ) is w²·D²(1, wℓ), so the
  only effective parameter is wℓ.
D_TM. Feature of residue i: P_i (the Cα position); δ(a, b) = f(‖a − b‖/d₀), f(r) = r/√(1 + r²) (concave, increasing,
f(0) = 0, hence a metric); c² = 1/2.
    1 − δ² = 1/(1 + r²) is the TM residue score, so D_TM² = (n + m)/2 − S_TM,  S_TM = max Σ_{(i,j)∈π} 1/(1 + r_ij²).
  At fixed d₀ it is a metric (the objective of TM-align = D_TM at fixed d₀). TM-score adds d₀ = d₀(L), division by a
  length and the 1 − S transform; the non-metric behavior of 1 − TM comes from the shorter-length normalization and the
  1 − S transform (the TM factorial design of the paper).

Computation. For a fixed pose (U, t) the correspondence is exact by dynamic programming. For a fixed correspondence
  D_{w,ℓ}: t is the difference of the centroids of the matched midpoints and U is exact from the SVD of the combined
  cross-covariance;
  D_TM: weighted least squares (weights 1/(1 + r²)², a majorization by the upper tangent of the concave function), so
  the value never decreases.
The two are alternated from many poses and the best is kept. Starts: fragment-pair superpositions (the local
superposition of TM-align), gapless threading, (search) superpositions of sub-fragments of the found correspondence
(the search of the TM-score program), and poses given from outside (e.g. the TM-align pose).
The computed value is an upper bound on D; there is no certificate (branch and bound). The inner loops run in numba
(pip install framedchain[rulers]). This module is the same computation as framespace_wip/stage0/rulers.py used for the
paper (the comparisons of TM-score forms, the chaining test and the ranking of existing measures).
"""
import math

import numpy as np
from numba import njit

from .frames import ca_trace


def edge_positions(x):
    """Coordinates (n, 3, 3) -> edge midpoints (n, 3) (Å, first Cα at the origin)."""
    ca = ca_trace(np.asarray(x))
    return 0.5 * (ca[..., 1:, :] + ca[..., :-1, :])


def d0_tm(L):
    """Length scale of TM-score d₀(L) = 1.24 (L − 15)^{1/3} − 1.8 (Å), not below 0.5 (as in TM-align)."""
    return max(0.5, 1.24 * (L - 15) ** (1 / 3) - 1.8) if L > 15 else 0.5


@njit(cache=True)
def _dp(K):
    """Maximum score of an order-preserving one-to-one partial correspondence and its matches (traceback). K (n, m)."""
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
    """Weighted least squares P_i ← U Q_j + t over the matches (ii, jj) with weights w."""
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
    """Alternates [DP → weighted least squares] from one start pose. -> (S, U, t, ii, jj)."""
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
    """Start poses [(U, t)]: superpositions of fragment pairs of length f (about per_side start points per side), n_thread gapless threadings, and random rotations with centroids matched."""
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
    """Poses superposing contiguous parts of the found correspondence (lengths k, k/2, k/4, … down to min_len, n_pos positions each) (the search of TM-score)."""
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
    """Evaluates every start once (score0), then refines the keep best."""
    s = np.array([score0(U, t) for U, t in poses])
    order = np.argsort(-s)[:keep] if keep else np.arange(len(poses))
    best = None
    for k in order:
        r = refine(*poses[k])
        if best is None or r[0] > best[0]:
            best = r
    return best


def tm_align(P, Q, d0, poses=None, extra_poses=(), iters=50, keep=16, irls=3, search=True, tol=1e-10):
    """S_TM at fixed d₀: max Σ_{matches} 1/(1 + ‖P_i − U Q_j − t‖²/d₀²). P (n, 3), Q (m, 3) are Cα positions (Å).
    poses: list of start poses (default start_poses); extra_poses: poses to add (e.g. the TM-align pose); search: one
    more round of sub-fragment search on the found correspondence.
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
    """D_TM² at a given correspondence and pose (by definition): Σ_matches r²/(1 + r²) + (number of unmatched terms)/2."""
    P, Q, U, t = (np.asarray(a, np.float64) for a in (P, Q, U, t))
    i, j = np.nonzero(np.asarray(match))
    r2 = ((P[i] - (Q[j] @ U.T + t)) ** 2).sum(-1) / d0 ** 2
    return float((r2 / (1 + r2)).sum()) + 0.5 * (len(P) + len(Q) - 2 * len(i))


def wl_align(x, y, ell, w=1.0, p=None, q=None, poses=None, extra_poses=(), iters=50, keep=16, search=True, tol=1e-10):
    """D_{w,ℓ}² and the alignment. x (n, 3, 3), y (m, 3, 3) coordinates; p, q edge midpoints (computed from the
    coordinates if omitted). With ell = inf there is no position term (basic distance × w²; starts: identity + 127
    random rotations). Zero terms are removed beforehand (the value is the same).
    -> dict(S, D2, U, t, match: (n, m) in the original indices)."""
    x, y = np.ascontiguousarray(x, np.float64), np.ascontiguousarray(y, np.float64)
    p = np.ascontiguousarray(edge_positions(x) if p is None else p, np.float64)
    q = np.ascontiguousarray(edge_positions(y) if q is None else q, np.float64)
    # Zero terms (edges of length 0) have zero gap cost and no gain when matched, so they do not change the value (same point); removed so they do not blur the start poses.
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
    """D_{w,ℓ}² at a given correspondence and pose (by definition)."""
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


# Factorial design of the TM forms: (fixed d₀ / d₀(L)) × (no division / division by L̄ = (n + m)/2) × (square-root distance form / 1 − S form).
def tm_forms(S, n, m):
    """S_TM (computed at the given d₀) -> the four forms. At fixed d₀, 'sqrt_raw' (D_TM) and 'sqrt_norm' (D_TM/√L̄) are
    metrics (the theorem and the normalized metric of the paper); the linear forms 'lin_raw' and 'lin_norm' are not."""
    Lb = 0.5 * (n + m)
    D2 = max(0.0, Lb - S)
    return {'sqrt_raw': math.sqrt(D2), 'lin_raw': D2, 'sqrt_norm': math.sqrt(D2 / Lb), 'lin_norm': 1.0 - S / Lb}


def triangle_rate(Dm, rel_tol=1e-9, abs_tol=1e-9, max_triples=None, seed=0):
    """For a symmetric matrix Dm (N, N), checks for every triple whether the longest side exceeds the sum of the other two.
    -> (violation rate, number of violations, number of triples, maximum relative violation).
    Relative violation = (longest − sum) / longest. A violation is longest − sum > abs_tol + rel_tol·longest. With
    max_triples, only that many random triples are checked."""
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
