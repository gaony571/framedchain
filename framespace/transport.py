"""앙상블 사이의 W₂ (공간 문서 2.2절의 거리 위의 최적 수송).

앙상블은 구조들의 (무게 붙은) 경험 분포다. 지상 비용은 표본 쌍마다의 d² 이고
    W₂² = min over 수송 계획 P (주변 분포가 두 앙상블의 무게) of Σ P_kl d²(A_k, B_l).
지상 거리가 거리이므로 W₂ 도 거리다. 표본 쌍의 d 는 distance.align 의 계산값(상계)이라 낸 W₂ 도 상계다.
길이가 다른 앙상블을 한 표에 놓을 때는 distance.length_scaled(W2, n, m) 눈금을 쓴다.
여기에는 정의대로의 계산만 둔다. 표본 수, 줄인 계산, 바닥값 같은 추정의 규약은 넣지 않았다.
"""
import math

import numpy as np
import torch

from .distance import align, as_tensor, sq_norm


def optimal_plan(C, a=None, b=None):
    """비용 행렬 C (I, J) 의 정확한 최적 수송 계획 (I, J). a, b 는 두 쪽의 무게(합이 1 이 아니어도 된다).
    무게가 없고 I = J 이면 할당 문제(scipy), 그 밖에는 POT 가 있으면 POT, 없으면 선형계획(scipy HiGHS)으로 푼다."""
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
        raise RuntimeError('수송 문제를 풀지 못했다: ' + res.message)
    return res.x.reshape(I, J)


def structure_costs(EA, EB, chunk=32, device=None, dtype=None, **align_kw):
    """두 앙상블 EA (I, n, 3, 3), EB (J, m, 3, 3) 의 모든 표본 쌍의 d² (I, J) (numpy).
    쌍을 chunk 개씩 묶어 align 으로 푼다(align_kw: n_random, iters, seed, improve_tol). 메모리는 chunk × 출발 수 × n × m 에 비례한다."""
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
    """앙상블 사이 W₂. wa, wb 는 표본의 무게(없으면 균일). cost 를 주면 그 지상 비용(d²)을 쓰고, 아니면 structure_costs(**kw) 로 낸다.
    반환 dict(W2, W2sq, plan, cost)."""
    C = structure_costs(EA, EB, **kw) if cost is None else np.asarray(cost, dtype=np.float64)
    P = optimal_plan(C, wa, wb)
    w2sq = float(max((P * C).sum(), 0.0))
    return dict(W2=math.sqrt(w2sq), W2sq=w2sq, plan=P, cost=C)
