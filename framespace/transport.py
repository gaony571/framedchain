"""W₂ between ensembles (optimal transport over the distance of the space).

An ensemble is a (weighted) empirical distribution of structures. The ground cost is d² for every sample pair, and
    W₂² = min over transport plans P (marginals equal to the weights of the two ensembles) of Σ P_kl d²(A_k, B_l).
Because the ground distance is a metric, W₂ is a metric. The sample-pair d comes from distance.align (an upper bound),
so the resulting W₂ is an upper bound too. To put ensembles of different lengths in one table use the scale
distance.length_scaled(W2, n, m). Only the computation by definition is here; estimation conventions (sample sizes,
the reduced computation, floors) are not included.
"""
import math

import numpy as np
import torch

from .distance import align, as_tensor, sq_norm


def optimal_plan(C, a=None, b=None):
    """Exact optimal transport plan (I, J) for the cost matrix C (I, J). a, b are the weights of the two sides (need not sum to 1).
    Without weights and with I = J this is an assignment problem (scipy); otherwise POT is used if installed, else a
    linear program (scipy HiGHS)."""
    C = np.asarray(C, dtype=np.float64)
    I, J = C.shape
    if a is None and b is None and I == J:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(C)
        P = np.zeros_like(C)
        P[r, c] = 1.0 / I
        return P
    a = np.full(I, 1.0 / I) if a is None else np.asarray(a, dtype=np.float64) / np.sum(a)
    b = np.full(J, 1.0 / J) if b is None else np.asarray(b, dtype=np.float64) / np.sum(b)
    try:
        import ot
        return ot.emd(a, b, C, numItermax=10_000_000)
    except ImportError:
        pass
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix, vstack
    cols = np.arange(I * J)
    Ar = coo_matrix((np.ones(I * J), (np.repeat(np.arange(I), J), cols)), shape=(I, I * J))
    Ac = coo_matrix((np.ones(I * J), (np.tile(np.arange(J), I), cols)), shape=(J, I * J))
    res = linprog(C.ravel(), A_eq=vstack([Ar, Ac]).tocsr(), b_eq=np.concatenate([a, b]), bounds=(0, None), method='highs')
    if not res.success:
        raise RuntimeError('the transport problem could not be solved: ' + res.message)
    return res.x.reshape(I, J)


def structure_costs(EA, EB, chunk=32, device=None, dtype=None, **align_kw):
    """d² (I, J) (numpy) for all sample pairs of two ensembles EA (I, n, 3, 3), EB (J, m, 3, 3).
    Pairs are solved by align in chunks of `chunk` (align_kw: n_random, iters, seed, improve_tol). Memory scales with
    chunk × number of starts × n × m."""
    EA = as_tensor(EA, device=device, dtype=dtype)
    EB = as_tensor(EB, device=EA.device, dtype=EA.dtype)
    I, J = len(EA), len(EB)
    ii, jj = torch.meshgrid(torch.arange(I, device=EA.device), torch.arange(J, device=EA.device), indexing='ij')
    ii, jj = ii.flatten(), jj.flatten()
    S = EA.new_zeros(I * J)
    seed = align_kw.pop('seed', 0)
    for c0 in range(0, I * J, chunk):
        sl = slice(c0, c0 + chunk)
        S[sl] = align(EA[ii[sl]], EB[jj[sl]], seed=seed + c0, **align_kw).score
    C = (sq_norm(EA)[ii] + sq_norm(EB)[jj] - 2 * S).clamp_min(0).view(I, J)
    return C.double().cpu().numpy()


def ensemble_w2(EA, EB, wa=None, wb=None, cost=None, **kw):
    """W₂ between ensembles. wa, wb are sample weights (uniform if omitted). If cost is given it is used as the ground
    cost (d²); otherwise it is computed by structure_costs(**kw).
    Returns dict(W2, W2sq, plan, cost)."""
    C = structure_costs(EA, EB, **kw) if cost is None else np.asarray(cost, dtype=np.float64)
    P = optimal_plan(C, wa, wb)
    w2sq = float(max((P * C).sum(), 0.0))
    return dict(W2=math.sqrt(w2sq), W2sq=w2sq, plan=P, cost=C)
