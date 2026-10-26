"""거리의 분해와 정규화 (정의역 문서 2.2절 따름, 결과 문서 1.1).

짝 몫·빈칸 몫. 최적 정렬 (π, U) 에서
    d² = Σ_{(i,j)∈π} ‖x_i − U y_j‖² + Σ_{짝 없는 i} ‖x_i‖² + Σ_{짝 없는 j} ‖y_j‖²  =  짝 몫 + 빈칸 몫
로 정확히 나뉜다. 짝 하나를 빈칸 둘로 바꾸면 θ > 120° 에서 비용이 줄어(길이 비와 무관) 최적 정렬이 그 짝을 끊으므로,
빈칸 몫에는 삽입·결실만이 아니라 서열상 대응하지만 120° 넘게 돈 변도 들어간다(빈칸 몫 ≠ 삽입 수; 따름 2).
분해는 정렬에 달려 있으므로 거의 최적인 정렬이 여럿이면 그 사이의 흔들림을 따로 본다.

평균 정규화. D̂ = d / √(‖x‖² + ‖y‖²) 는 모든 길이에서 거리이고 값은 [0, 1] 이다(Hilbert 단위 구면 묻기; 따름정리).
"""
import numpy as np
import torch

from .distance import align, as_tensor, dp_score, sq_norm
from .geometry import procrustes, random_rotations


def split_sq_distance(x, y, alignment=None, **kw):
    """d² 를 짝 몫과 빈칸 몫으로 나눈다. x (n, 3, 3), y (m, 3, 3) 또는 묶음 (P, n, 3, 3), (P, m, 3, 3).
    alignment 를 주지 않으면 align(x, y, **kw) 로 정렬한다(상계 계산).
    반환 dict: d2, matched (짝 몫), gap (빈칸 몫), n_matched (짝 수), gap_share (빈칸 몫 / d², d² = 0 이면 0), alignment."""
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
    """평균 정규화 거리 D̂ = d / √(‖x‖² + ‖y‖²) ∈ [0, 1]. d 를 주지 않으면 거리를 계산한다(상계)."""
    x, y = as_tensor(x), as_tensor(y)
    y = y.to(x)
    if d is None:
        a = align(x, y, **kw)
        d = (sq_norm(x) + sq_norm(y) - 2 * a.score).clamp_min(0).sqrt()
    return d / (sq_norm(x) + sq_norm(y)).sqrt()


def near_optimal_alignments(x, y, eps=0.005, n_random=127, iters=10, seed=0, improve_tol=1e-7):
    """거의 최적인 대응들과 그 교집합 (공통 정의역 논문 1b·1c 의 공동 관문, 첫 세션 0.4 의 정렬 불확실성).
    align 과 같은 계산(항등 + 무작위 n_random 출발, 대응과 회전을 번갈아)을 한 쌍 x (n, 3, 3), y (m, 3, 3) 에 하고,
    출발마다의 최종 점수가 S ≥ (1 − eps) S_max 인 출발을 "거의 최적" 으로 모은다.
    반환 dict: S_max, best_match (n, m) bool, scores (출발 수,), near (거의 최적 출발 번호), n_near,
    n_distinct (거의 최적 출발들이 찾은 서로 다른 대응의 수), intersection (n, m) bool (거의 최적 대응 모두에 있는 짝).
    code/r38_s0.py 의 align_all, code/r38_p3.py 의 π̄ 단계와 같은 계산이다(무작위 출발 회전의 씨앗만 seed 로 준다)."""
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
