"""The distance of the space.

A point is a finite sequence of 3×3 matrices. Inserting or removing zero matrices, or multiplying every term on
the left by the same rotation, gives the same point.
    d(x, y)² = min over (order-preserving partial correspondence π, rotation U) of
               Σ_{(i,j)∈π} ‖x_i − U y_j‖² + Σ_{i unmatched} ‖x_i‖² + Σ_{j unmatched} ‖y_j‖²
             = ‖x‖² + ‖y‖² − 2 S,   S = max Σ_{(i,j)∈π} ⟨x_i, U y_j⟩
Computation: for a fixed rotation the correspondence is exact by dynamic programming; for a fixed correspondence the
rotation is exact by a 3×3 SVD. The two are alternated from several random start rotations (128 by default) and the
best is kept. The computed value is an upper bound on d; certify.bnb certifies the global optimum.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch

from .geometry import procrustes, random_rotations


def dp_score(K):
    """Maximum score of an order-preserving one-to-one partial correspondence. K (..., n, m) -> (...,).
    F[i, j] = max(F[i−1, j], F[i, j−1], F[i−1, j−1] + K[i, j]). The autograd gradient is the matched cells (0/1)."""
    F = K.new_zeros(K.shape[:-2] + (K.shape[-1] + 1,))
    for i in range(K.shape[-2]):
        G = torch.maximum(F[..., 1:], F[..., :-1] + K[..., i, :])
        F = torch.cat([F[..., :1], torch.cummax(G, -1).values], -1)
    return F[..., -1]


@dataclass
class Alignment:
    """Result of align: score S (P,), rotation U (P, 3, 3) applied to y, and matches (P, n, m) bool."""
    score: torch.Tensor      # (P,) S
    rotation: torch.Tensor   # (P, 3, 3) rotation U applied to y
    match: torch.Tensor      # (P, n, m) bool


def as_tensor(x, device=None, dtype=None):
    """numpy array or tensor -> tensor (non-floating inputs become float64; device and dtype kept unless given)."""
    t = torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x)
    return t.to(device=device or t.device, dtype=dtype or (t.dtype if t.is_floating_point() else torch.float64))


def pad_stack(points, n=None):
    """Points of different lengths [(n_i, 3, 3)] -> (P, N, 3, 3), padded with zeros (the same points)."""
    points = [as_tensor(p) for p in points]
    n = n or max(len(p) for p in points)
    out = points[0].new_zeros((len(points), n, 3, 3))
    for i, p in enumerate(points):
        out[i, :len(p)] = p
    return out


def sq_norm(x):
    """Squared norm ‖x‖² of a point (n, 3, 3), or of each point of a batch (P, n, 3, 3) (tensor)."""
    return (x ** 2).sum((-1, -2, -3))


def align(x, y, starts=None, n_random=127, iters=10, seed=0, improve_tol=1e-7):
    """x (P, n, 3, 3), y (P, m, 3, 3) (or unbatched (n, 3, 3)). Without starts (P, s, 3, 3): identity + n_random random rotations.
    iters: maximum number of alternations between correspondence and rotation. improve_tol: a start stops when it
    improves by less than this (relative). seed: seed of the random start rotations.
    Inputs may be numpy arrays or tensors; y and starts are converted to the dtype and device of x.
    Returns an Alignment (score S, rotation U, matches)."""
    x = as_tensor(x)
    y = as_tensor(y).to(x)
    if starts is not None:
        starts = as_tensor(starts).to(x)
    single = x.dim() == 3
    if single:
        x, y = x[None], y[None]
        if starts is not None and starts.dim() == 3:
            starts = starts[None]
    P = len(x)
    if starts is None:
        I = torch.eye(3, dtype=x.dtype, device=x.device).expand(P, 1, 3, 3)
        Rr = random_rotations(P * n_random, seed=seed, dtype=x.dtype, device=x.device).view(P, n_random, 3, 3) if n_random else I[:, :0]
        starts = torch.cat([I, Rr], 1)
    U = starts.clone()
    s = U.shape[1]
    best = torch.full((P, s), -1e30, dtype=x.dtype, device=x.device)
    bU, bG = U.clone(), None
    for it in range(iters + 1):
        with torch.no_grad():
            K = torch.einsum('pnxz,psmxz->psnm', x, torch.einsum('psxy,pmyz->psmxz', U, y))
        K = K.detach().requires_grad_(True)
        with torch.enable_grad():
            S = dp_score(K)
            G, = torch.autograd.grad(S.sum(), K)
        with torch.no_grad():
            imp = S > best + improve_tol * (1 + best.abs())
            best = torch.where(imp, S, best)
            bU = torch.where(imp[..., None, None], U, bU)
            bG = G if bG is None else torch.where(imp[..., None, None], G, bG)
            if it == iters or not imp.any():
                break
            U = torch.where(imp[..., None, None], procrustes(torch.einsum('psnm,pmxz,pnyz->psxy', G, y, x)), U)
    k = best.argmax(1)
    ar = torch.arange(P, device=x.device)
    out = Alignment(best[ar, k].detach(), bU[ar, k], bG[ar, k] > 0.5)
    if single:
        out = Alignment(out.score[0], out.rotation[0], out.match[0])
    return out


def sq_distance(x, y, **kw):
    """d² and the alignment. x, y are (n, 3, 3) or batches (P, n, 3, 3)."""
    x, y = as_tensor(x), as_tensor(y)
    y = y.to(x)
    a = align(x, y, **kw)
    return (sq_norm(x) + sq_norm(y) - 2 * a.score).clamp_min(0), a


def distance(x, y, **kw):
    """Distance d(x, y) and the alignment (score S, rotation U applied to y, matches). x (n, 3, 3) and y (m, 3, 3), or
    batches (P, n, 3, 3), (P, m, 3, 3) zero-padded to a common length; numpy arrays or tensors. **kw goes to align
    (n_random, iters, seed, starts). The value is an upper bound on d (certify with bnb)."""
    d2, a = sq_distance(x, y, **kw)
    return d2.sqrt(), a


def fixed_sq_distance(x, y, kx=None, ky=None):
    """When the correspondence is known: d_fix² with the correspondence (kx ↔ ky; default: edge index to edge index)
    fixed and only the rotation optimized; exact with one SVD.
    All pairs of batches x (I, n, 3, 3), y (J, m, 3, 3) -> (I, J). An upper bound on d."""
    x, y = as_tensor(x), as_tensor(y).to(as_tensor(x))
    kx = np.arange(x.shape[1]) if kx is None else kx
    ky = np.arange(y.shape[1]) if ky is None else ky
    H = torch.einsum('jkcb,ikab->ijca', y[:, ky], x[:, kx])
    sv = torch.linalg.svdvals(H)
    S = sv[..., 0] + sv[..., 1] + torch.sign(torch.det(H)) * sv[..., 2]
    return (sq_norm(x)[:, None] + sq_norm(y)[None] - 2 * S).clamp_min(0)


# Scales. n and m are the numbers of non-zero terms.
def length_scaled(d, n, m):
    """d / √(0.75 (n + m)): between 0 and 1. For tables of pairs of different lengths."""
    return d / math.sqrt(0.75 * (n + m))


def sim_long(S, n, m):
    """S / (0.75 max(n, m)): similarity normalized by the longer structure."""
    return S / (0.75 * max(n, m))


def fixed_scaled(d_fix, n):
    """d_fix / √(2n): between 0 and 0.87. A scale within one protein."""
    return d_fix / math.sqrt(2 * n)


def match_to_map(match):
    """Match matrix (n, m) -> for each term of x the index of the matched term of y (−1 if unmatched)."""
    M = match.cpu().numpy() if torch.is_tensor(match) else np.asarray(match)
    mp = -np.ones(M.shape[0], int)
    k, l = np.nonzero(M)
    mp[k] = l
    return mp
