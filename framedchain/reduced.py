"""The reduced computation: mean-point starts and band-limited DP. Used to compute distances for many sample pairs of
same-protein ensembles.

Banded DP. The maximum score of an order-preserving partial correspondence whose matches are restricted to the band
B = {(i, j): lo_i ≤ j < lo_i + w}. If lo never decreases (the center line is monotone), storing only the band cells of
each row is exact:
  F_i[j] (j 1-based, y[:j]) equals F_{i−1}[j] for j < lo_i, and equals F_{i−1}[lo_{i−1} + w] for j > lo_{i−1} + w
  (the bands of earlier rows lie no further right). So the previous row's band values are moved over, extended to the
  right as a constant.
The optimum within the band is at most the full DP optimum (a more restricted problem), so distances computed from it
are upper bounds. "Widen the band when a match touches its edge" is a check, not a guarantee of exactness. Exactness is
measured against the full computation and branch and bound (in the paper: the upper-bound bias = 2 × (cross error −
floor error) against branch and bound).

Mean-point starts. Samples of a same-protein ensemble share the mean point and the edge numbering. Aligning the two
mean points generously (many starts) gives a rotation U0 and a correspondence π̄; with the rotation R_k that aligns
sample k to its own mean computed in advance, the start rotation of the pair (k, l) is R_k^T U0 R'_l.

Every ensemble computation of the paper follows this rule: 128 starts between the mean points, one start per sample
pair, band half-width 5 at first (20 for the pairs of the comparison across sequences with TM-score below 0.3, the
setting chosen after the bias check by the rule fixed in a later record, before the re-measurement; S2 Text 2 of the
paper) and doubled (up to the number of columns) only
for pairs touching the band edge, float64. The functions here are the same computation as the code that ran
(framespace_wip/stage0/band.py and red_pairs in code/r38_s0.py of the framedchain-paper repository).
"""
import numpy as np
import torch

from .distance import align, as_tensor, sq_norm
from .geometry import procrustes, random_rotations
from .operations import mean_point


def center_path(match, m):
    """Matches between the mean points (n, m) bool -> center column per row (n,) (monotone). Unmatched rows are filled
    linearly between the neighboring matched columns."""
    M = match.cpu().numpy() if torch.is_tensor(match) else np.asarray(match)
    n = M.shape[0]
    rows, cols = np.nonzero(M)
    if len(rows) == 0:
        return np.round(np.linspace(0, m - 1, n)).astype(int)
    c = np.interp(np.arange(n), rows, cols)
    return np.maximum.accumulate(np.clip(np.round(c).astype(int), 0, m - 1))


def band_windows(center, m, k):
    """Center columns (n,) and half-width k -> left ends of the band lo (n,) (non-decreasing) and width w = min(2k+1, m)."""
    w = int(min(2 * k + 1, m))
    lo = np.clip(np.asarray(center) - k, 0, m - w)
    return np.maximum.accumulate(lo).astype(int), w


def dp_score_band(Kb, lo):
    """Scores inside the band Kb (..., n, w) (cell (i, t) is column lo_i + t) -> maximum score (...,).
    The autograd gradient is the matched cells."""
    n, w = Kb.shape[-2], Kb.shape[-1]
    lo = torch.as_tensor(np.asarray(lo), device=Kb.device)
    F = Kb.new_zeros(Kb.shape[:-2] + (w + 1,))
    ar = torch.arange(w + 1, device=Kb.device)
    lo_p = lo[0]
    for i in range(n):
        idx = (ar + (lo[i] - lo_p)).clamp(max=w)          # move the previous row's band to this row's band (constant extension to the right)
        Fp = F[..., idx]
        G = torch.maximum(Fp[..., 1:], Fp[..., :-1] + Kb[..., i, :])
        F = torch.cat([Fp[..., :1], torch.cummax(G, -1).values], -1)
        lo_p = lo[i]
    return F[..., -1]


def align_band(x, y, lo, w, starts, iters=10, improve_tol=1e-7):
    """Alternates correspondence (banded DP) and rotation. x (P, n, 3, 3), y (P, m, 3, 3), starts (P, s, 3, 3); the band
    (lo, w) is the same for all pairs.
    Returns score S (P,), rotation U (P, 3, 3), matches (P, n, m) bool, and whether a match touched the band edge (P,)
    bool (matrix borders excluded)."""
    P, n, m = len(x), x.shape[1], y.shape[1]
    lo_t = torch.as_tensor(lo, device=x.device)
    col = lo_t[:, None] + torch.arange(w, device=x.device)[None]          # (n, w)
    yg = y[:, col]                                                        # (P, n, w, 3, 3)
    U = starts.clone()
    s = U.shape[1]
    best = torch.full((P, s), -1e30, dtype=x.dtype, device=x.device)
    bU, bG = U.clone(), None
    for it in range(iters + 1):
        with torch.no_grad():
            K = torch.einsum('pnxz,psnwxz->psnw', x, torch.einsum('psxy,pnwyz->psnwxz', U, yg))
        K = K.detach().requires_grad_(True)
        with torch.enable_grad():
            S = dp_score_band(K, lo)
            G, = torch.autograd.grad(S.sum(), K)
        with torch.no_grad():
            imp = S > best + improve_tol * (1 + best.abs())
            best = torch.where(imp, S, best)
            bU = torch.where(imp[..., None, None], U, bU)
            bG = G if bG is None else torch.where(imp[..., None, None], G, bG)
            if it == iters or not imp.any():
                break
            U = torch.where(imp[..., None, None], procrustes(torch.einsum('psnw,pnwxz,pnyz->psxy', G, yg, x)), U)
    k = best.argmax(1)
    ar = torch.arange(P, device=x.device)
    Gb = bG[ar, k] > 0.5                                                  # (P, n, w)
    match = torch.zeros(P, n, m, dtype=torch.bool, device=x.device)
    pi, ii, tt = torch.nonzero(Gb, as_tuple=True)
    match[pi, ii, col[ii, tt]] = True
    left = (tt == 0) & (lo_t[ii] > 0)
    right = (tt == w - 1) & (lo_t[ii] + w - 1 < m - 1)
    edge = torch.zeros(P, dtype=torch.bool, device=x.device)
    edge[pi[left | right]] = True
    return best[ar, k].detach(), bU[ar, k], match, edge


def self_rotations(E, m):
    """Rotations R_k (F, 3, 3) aligning each sample of ensemble E (F, n, 3, 3) to its mean point m: R_k E_k ≈ m
    (correspondence: edge index to edge index)."""
    return procrustes(torch.einsum('ikxz,kyz->ixy', E, m))


class MeanStart:
    """Aligns the mean points of two ensembles once and provides start rotations and the band for every sample pair.
    n_mean_starts: number of start rotations for aligning the mean points. A mean point is not a real structure, so many
    starts are used instead of a certificate. Pairs within one ensemble (floors) use MeanStart(E, E)."""

    def __init__(self, EA, EB, n_mean_starts=128, mean_kw=None, seed=0):
        self.EA = as_tensor(EA)
        self.EB = as_tensor(EB).to(self.EA)
        kw = mean_kw or {}
        self.mA, self.infoA = mean_point(self.EA, return_info=True, **kw)
        self.mB, self.infoB = mean_point(self.EB, return_info=True, **kw)
        al = align(self.mA, self.mB, n_random=n_mean_starts - 1, seed=seed)
        self.U0, self.match, self.S_mean = al.rotation, al.match, float(al.score)
        self.RA = self_rotations(self.EA, self.mA)
        self.RB = self_rotations(self.EB, self.mB)
        self.center = center_path(self.match, self.EB.shape[1])

    def starts(self, ia, ib, n_local=1, seed=0):
        """Start rotations (P, n_local, 3, 3) for the pairs (ia[p], ib[p]): the first is R_k^T U0 R'_l, the rest are random."""
        U = torch.einsum('pyx,yz,pzw->pxw', self.RA[ia], self.U0, self.RB[ib])[:, None]
        if n_local > 1:
            P = len(ia)
            Rr = random_rotations(P * (n_local - 1), seed=seed, dtype=U.dtype, device=U.device).view(P, n_local - 1, 3, 3)
            U = torch.cat([U, Rr], 1)
        return U


def pair_scores(ms, ia, ib, k0=5, iters=10, want_gap=False, membudget=3e9):
    """Applies the reduced computation to the sample pairs (ia[p], ib[p]). ms = MeanStart(EA, EB).
    Starts with band half-width k0 and re-solves, with the half-width doubled, only the pairs touching the band edge
    (until the band reaches the number of columns).
    membudget: memory estimate (bytes) that sets how many pairs are solved at once.
    Returns dict: d (P,) distances (upper bounds), gap (P,) gap part (sum of squared norms of unmatched terms; with
    want_gap, otherwise None), S (P,), half_width (P,) final half-width, widened (number of pairs widened from the first band)."""
    EA, EB = ms.EA, ms.EB
    n, m = EA.shape[1], EB.shape[1]
    ia = torch.as_tensor(np.asarray(ia), device=EA.device)
    ib = torch.as_tensor(np.asarray(ib), device=EA.device)
    nA, nB = (EA ** 2).sum((-1, -2)), (EB ** 2).sum((-1, -2))
    S = EA.new_zeros(len(ia))
    G = EA.new_zeros(len(ia)) if want_gap else None
    kused = np.zeros(len(ia), int)
    widened = 0
    todo = torch.arange(len(ia), device=EA.device)
    kk = k0
    while len(todo):
        lo, w = band_windows(ms.center, m, kk)
        step = max(1, int(membudget / (n * w * 9 * 8 * 4)))
        edge_all = torch.zeros(len(todo), dtype=torch.bool, device=EA.device)
        for c in range(0, len(todo), step):
            sel = todo[c:c + step]
            a, b = EA[ia[sel]], EB[ib[sel]]
            sc, _, M, edge = align_band(a, b, lo, w, ms.starts(ia[sel], ib[sel], n_local=1), iters=iters)
            S[sel] = sc
            edge_all[c:c + step] = edge
            if want_gap:
                G[sel] = (nA[ia[sel]] * (~M.any(2))).sum(1) + (nB[ib[sel]] * (~M.any(1))).sum(1)
        kused[todo.cpu().numpy()] = kk
        if w >= m:
            break
        todo = todo[edge_all]
        if kk == k0:
            widened = int(len(todo))
        kk *= 2
    d2 = (sq_norm(EA)[ia] + sq_norm(EB)[ib] - 2 * S).clamp_min(0)
    return dict(d=d2.sqrt().cpu().numpy(), gap=G.cpu().numpy() if want_gap else None, S=S.cpu().numpy(), half_width=kused, widened=widened)
