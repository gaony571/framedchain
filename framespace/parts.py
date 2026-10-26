"""Splitting and normalising the distance.

Matched part and gap part. At an optimal alignment (π, U)
    d² = Σ_{(i,j)∈π} ‖x_i − U y_j‖² + Σ_{unmatched i} ‖x_i‖² + Σ_{unmatched j} ‖y_j‖²  =  matched part + gap part
exactly. Replacing one match by two gaps lowers the cost when θ > 120° (whatever the length ratio), so the optimal
alignment breaks such a match: the gap part contains not only insertions and deletions but also edges that correspond
in sequence and are rotated by more than 120° (gap part ≠ number of insertions).
The split depends on the alignment, so when several alignments are nearly optimal their variation is reported separately.

Mean normalisation. D̂ = d / √(‖x‖² + ‖y‖²) is a metric at every length with values in [0, 1] (embedding in the
unit sphere of a Hilbert space).
"""
import numpy as np
import torch

from .distance import align, as_tensor, dp_score, sq_norm
from .geometry import procrustes, random_rotations


def split_sq_distance(x, y, alignment=None, **kw):
    """Splits d² into the matched part and the gap part. x (n, 3, 3), y (m, 3, 3) or batches (P, n, 3, 3), (P, m, 3, 3).
    Without alignment, align(x, y, **kw) is used (upper-bound computation).
    Returns dict: d2, matched (matched part), gap (gap part), n_matched (number of matches), gap_share (gap part / d²,
    0 if d² = 0), alignment."""
    x, y = as_tensor(x), as_tensor(y)
    y = y.to(x)
    al = alignment or align(x, y, **kw)
    single = x.dim() == 3
    X, Y = (x[None], y[None]) if single else (x, y)
    U = al.rotation[None] if single else al.rotation
    M = (al.match[None] if single else al.match).to(torch.bool)
    UY = torch.einsum('pxy,pmyz->pmxz', U, Y)
    diff = ((X[:, :, None] - UY[:, None]) ** 2).sum((-1, -2))                      # (P, n, m)
    matched = (diff * M).sum((1, 2))
    gap = ((X ** 2).sum((-1, -2)) * (~M.any(2))).sum(1) + ((Y ** 2).sum((-1, -2)) * (~M.any(1))).sum(1)
    d2 = matched + gap
    share = torch.where(d2 > 0, gap / d2.clamp_min(1e-300), torch.zeros_like(d2))
    out = dict(d2=d2, matched=matched, gap=gap, n_matched=M.sum((1, 2)), gap_share=share)
    if single:
        out = {k: v[0] for k, v in out.items()}
    out['alignment'] = al
    return out


def normalized_distance(x, y, d=None, **kw):
    """Mean-normalised distance D̂ = d / √(‖x‖² + ‖y‖²) ∈ [0, 1]. Without d, the distance is computed (upper bound)."""
    x, y = as_tensor(x), as_tensor(y)
    y = y.to(x)
    if d is None:
        a = align(x, y, **kw)
        d = (sq_norm(x) + sq_norm(y) - 2 * a.score).clamp_min(0).sqrt()
    return d / (sq_norm(x) + sq_norm(y)).sqrt()


def near_optimal_alignments(x, y, eps=0.005, n_random=127, iters=10, seed=0, improve_tol=1e-7):
    """Nearly optimal correspondences and their intersection (the joint gate of experiments 1b and 1c of the paper, and
    the alignment uncertainty of its first session).
    Runs the same computation as align (identity + n_random random starts, alternating correspondence and rotation) on
    one pair x (n, 3, 3), y (m, 3, 3), and collects the starts whose final score satisfies S ≥ (1 − eps) S_max as
    "nearly optimal".
    Returns dict: S_max, best_match (n, m) bool, scores (number of starts,), near (indices of nearly optimal starts),
    n_near, n_distinct (number of different correspondences found by the nearly optimal starts), intersection (n, m)
    bool (matches present in every nearly optimal correspondence).
    The same computation as align_all in code/r38_s0.py and the π̄ step in code/r38_p3.py of the framespace-paper
    repository (only the seed of the random start rotations is passed as seed)."""
    x, y = as_tensor(x), as_tensor(y)
    y = y.to(x)
    X, Y = x[None], y[None]
    I = torch.eye(3, dtype=X.dtype, device=X.device).expand(1, 1, 3, 3)
    U = torch.cat([I, random_rotations(n_random, seed=seed, dtype=X.dtype, device=X.device).view(1, n_random, 3, 3)], 1)
    best = torch.full((1, U.shape[1]), -1e30, dtype=X.dtype, device=X.device)
    bG = None
    for it in range(iters + 1):
        with torch.no_grad():
            K = torch.einsum('pnxz,psmxz->psnm', X, torch.einsum('psxy,pmyz->psmxz', U, Y))
        K = K.detach().requires_grad_(True)
        with torch.enable_grad():
            Sc = dp_score(K)
            Gr, = torch.autograd.grad(Sc.sum(), K)
        with torch.no_grad():
            imp = Sc > best + improve_tol * (1 + best.abs())
            best = torch.where(imp, Sc, best)
            bG = Gr if bG is None else torch.where(imp[..., None, None], Gr, bG)
            if it == iters or not imp.any():
                break
            U = torch.where(imp[..., None, None], procrustes(torch.einsum('psnm,pmxz,pnyz->psxy', Gr, Y, X)), U)
    scores = best[0].cpu().numpy()
    M = (bG[0] > 0.5).cpu().numpy()
    S_max = float(scores.max())
    k = int(scores.argmax())
    near = np.where(scores >= (1 - eps) * S_max)[0]
    return dict(S_max=S_max, best_match=M[k], scores=scores, near=near, n_near=int(len(near)),
                n_distinct=len({M[s].tobytes() for s in near}), intersection=np.logical_and.reduce(M[near], axis=0))
