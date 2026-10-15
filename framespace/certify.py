"""잔기 단위 거리의 전역 최적 인증 (분기 한정).

실제 구조(변마다 R/2)에서 변 쌍의 점수는 ⟨A_k, U B_l⟩ = ⟨c_kl, u⟩² − 1/4 (c_kl = q_k ⊗ conj(p_l), u 는 U 의 사원수).
회전 벡터 공간 [−π, π]³ 을 정육면체 칸으로 나누면, 칸 안의 모든 회전은 중심 u0 에서 S³ 위 각 ρ = √3 σ / 2 안에 있으므로
점수 ≤ cos²(max(0, φ0 − ρ)) − 1/4 (φ0 = arccos|⟨c, u0⟩|) 이고, 짝짓기 DP 는 점수에 단조라 칸의 상한이 된다.
"""
import time

import numpy as np
import torch

from .distance import align, dp_score
from .geometry import mat_to_quat, quat_conj, quat_mul, quat_to_mat, rotvec_to_quat


def bnb(A, B, rel_tol=0.005, tol=None, max_cells=2_000_000, n_starts=256, device=None, verbose=False):
    """A (n, 3, 3), B (m, 3, 3) 는 실제 구조(R/2). S 의 전역 최댓값을 인증한다.
    허용 오차는 tol (S 의 절대값) 또는 rel_tol × 0.75 max(n, m) (S / (0.75 max(n, m)) 단위, 기본 0.005).
    반환: S_heur (출발 회전 여럿의 휴리스틱), S_lb, S_ub, certified, 칸의 단계와 DP 횟수."""
    t0 = time.time()
    dev = device or (A.device if torch.is_tensor(A) else 'cpu')
    A = torch.as_tensor(np.asarray(A) if not torch.is_tensor(A) else A, dtype=torch.float64, device=dev)
    B = torch.as_tensor(np.asarray(B) if not torch.is_tensor(B) else B, dtype=torch.float64, device=dev)
    qa, qb = mat_to_quat(2 * A), mat_to_quat(2 * B)
    C = quat_mul(qa[:, None], quat_conj(qb)[None])                                      # (n, m, 4)
    n, m = len(A), len(B)
    tol = rel_tol * 0.75 * max(n, m) if tol is None else tol
    S_h = LB = float(align(A, B, n_random=n_starts - 1).score)
    bs = int(max(64, min(20000, 2.5e8 / (n * m))))
    L0 = 3
    sg = np.pi / 2 ** L0
    ax = (np.arange(2 ** L0) + 0.5) * 2 * sg - np.pi
    cen = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), -1).reshape(-1, 3)
    off = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * 0.5
    ndp, level, certified, hist = 0, L0, False, []
    while True:
        cen = cen[np.linalg.norm(np.maximum(np.abs(cen) - sg, 0), axis=1) <= np.pi]
        rho = np.sqrt(3) * sg / 2
        UB = np.empty(len(cen))
        CV = np.empty(len(cen))
        with torch.no_grad():
            for i in range(0, len(cen), bs):
                u0 = rotvec_to_quat(torch.tensor(cen[i:i + bs], device=dev, dtype=torch.float64))
                dot = torch.einsum('nmc,bc->bnm', C, u0).abs().clamp(max=1.0)
                UB[i:i + bs] = dp_score(torch.cos((torch.acos(dot) - rho).clamp_min(0)) ** 2 - 0.25).cpu().numpy() + 1e-9
                CV[i:i + bs] = dp_score(dot ** 2 - 0.25).cpu().numpy()
                ndp += 2 * len(u0)
        top = np.argsort(-CV)[:16]
        U0 = quat_to_mat(rotvec_to_quat(torch.tensor(cen[top], device=dev, dtype=torch.float64)))
        LB = max(LB, float(align(A, B, starts=U0).score), float(CV.max()))
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
