"""Certificate of the global optimum (branch and bound over rotations).

For terms of real structures x_k = s_k R_k / 2, the score of a term pair is ⟨x_k, U y_l⟩ = s_k t_l (⟨c_kl, u⟩² − 1/4)
(c_kl = q_k ⊗ conj(p_l) is the quaternion of the two frames, u is the quaternion of U, s and t are length ratios).
If the rotation-vector cube [−π, π]³ is divided into cubic cells, every rotation in a cell lies within the angle
ρ = √3 σ / 2 on S³ of the cell centre u0, so
score ≤ s_k t_l (cos²(max(0, φ0 − ρ)) − 1/4)   (φ0 = arccos|⟨c, u0⟩|; the inequality holds because length ratios are positive).
The correspondence DP is monotone in the scores, so this gives an upper bound for the cell. Cells whose upper bound is
below the best value + tolerance are discarded and the remaining cells are split into 8.
Only real structures (every term s·R/2 with s > 0) are accepted.
"""
import time

import numpy as np
import torch

from .distance import align, dp_score
from .geometry import mat_to_quat, quat_conj, quat_mul, quat_to_mat, rotvec_to_quat


def _frames_and_scale(x):
    s = 2.0 * x[..., :, 0].norm(dim=-1)                 # length ratio (the norm of the first column is s/2)
    return 2.0 * x / s[:, None, None], s


def bnb(x, y, rel_tol=0.005, tol=None, max_cells=2_000_000, n_starts=128, level0=3, n_refine=16, batch=None, iters=10, device=None, verbose=False):
    """x (n, 3, 3) and y (m, 3, 3) are real structures. Certifies the global maximum of S within a tolerance.
    The tolerance is tol (absolute, in units of S) or rel_tol × 0.75 max(n, m) (sim_long units; default 0.005).
    n_starts: number of start rotations for the initial lower bound. level0: initial grid (2^level0 cells per axis).
    n_refine: number of cells whose lower bound is refined at each level.
    max_cells: stop (certificate fails) if more cells than this remain. batch: cells per batch (default fitted to n·m).
    Returns S_heur (value from n_starts start rotations), S_lb, S_ub, tol, certified, the grid level, number of DPs, time."""
    t0 = time.time()
    dev = device or (x.device if torch.is_tensor(x) else 'cpu')
    x = torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x, dtype=torch.float64, device=dev)
    y = torch.as_tensor(np.asarray(y) if not torch.is_tensor(y) else y, dtype=torch.float64, device=dev)
    Rx, sx = _frames_and_scale(x)
    Ry, sy = _frames_and_scale(y)
    if not (sx > 0).all() or not (sy > 0).all():
        raise ValueError('bnb accepts only real structures (terms s·R/2 without zero terms)')
    C = quat_mul(mat_to_quat(Rx)[:, None], quat_conj(mat_to_quat(Ry))[None])      # (n, m, 4)
    W = sx[:, None] * sy[None]                                                     # (n, m)
    n, m = len(x), len(y)
    tol = rel_tol * 0.75 * max(n, m) if tol is None else tol
    S_h = LB = float(align(x, y, n_random=n_starts - 1, iters=iters).score)
    bs = batch or int(max(64, min(20000, 2.5e8 / (n * m))))
    sg = np.pi / 2 ** level0
    ax = (np.arange(2 ** level0) + 0.5) * 2 * sg - np.pi
    cen = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), -1).reshape(-1, 3)
    off = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * 0.5
    ndp, level, certified, hist = 0, level0, False, []
    while True:
        cen = cen[np.linalg.norm(np.maximum(np.abs(cen) - sg, 0), axis=1) <= np.pi]
        rho = np.sqrt(3) * sg / 2
        UB = np.empty(len(cen))
        CV = np.empty(len(cen))
        with torch.no_grad():
            for i in range(0, len(cen), bs):
                u0 = rotvec_to_quat(torch.tensor(cen[i:i + bs], device=dev, dtype=torch.float64))
                dot = torch.einsum('nmc,bc->bnm', C, u0).abs().clamp(max=1.0)
                UB[i:i + bs] = dp_score(W * (torch.cos((torch.acos(dot) - rho).clamp_min(0)) ** 2 - 0.25)).cpu().numpy() + 1e-9
                CV[i:i + bs] = dp_score(W * (dot ** 2 - 0.25)).cpu().numpy()
                ndp += 2 * len(u0)
        top = np.argsort(-CV)[:n_refine]
        U0 = quat_to_mat(rotvec_to_quat(torch.tensor(cen[top], device=dev, dtype=torch.float64)))
        LB = max(LB, float(align(x, y, starts=U0, iters=iters).score), float(CV.max()))
        keep = UB > LB + tol
        hist.append((level, len(cen), int(keep.sum()), float(UB.max()), LB))
        if verbose:
            print(f'  level {level}: cells {len(cen)}, kept {int(keep.sum())}, max UB {UB.max():.4f}, LB {LB:.4f}, {time.time() - t0:.1f}s', flush=True)
        if not keep.any():
            certified, UBm = True, LB + tol
            break
        if keep.sum() * 8 > max_cells:
            UBm = float(UB.max())
            break
        sg /= 2
        level += 1
        cen = (cen[keep][:, None, :] + off[None] * 2 * sg).reshape(-1, 3)
    return dict(S_heur=S_h, S_lb=LB, S_ub=UBm, tol=tol, certified=certified, levels=level, n_dp=ndp, sec=time.time() - t0, n=n, m=m, hist=hist)
