"""잔기 단위 거리.

구조 A (n 변), B (m 변) 의 좌표는 변마다 3×3 행렬(실제 구조는 R/2). 짝 없는 변은 영행렬과 짝지어진다.
    d²(A, B) = min_{U ∈ SO(3)} min_{순서를 지키는 1대1 짝짓기} [ Σ_짝 |A_k − U B_l|² + Σ_짝없음 |·|² ]
             = |A|² + |B|² − 2 S,   S = max_U max_짝짓기 Σ_짝 ⟨A_k, U B_l⟩
실제 구조끼리는 |A|² + |B|² = 0.75 (n + m). 정규화 거리는 d² / (|A|² + |B|²).
정렬(DP)과 회전(3×3 SVD)을 번갈아 풀고, 여러 출발 회전 중 가장 좋은 것을 고른다. 인증은 certify.bnb.
영행렬로 채워도 값이 바뀌지 않으므로, 길이가 다른 구조들은 영으로 채워 한 묶음으로 계산한다.
"""
from dataclasses import dataclass

import numpy as np
import torch

from .geometry import procrustes, random_rotations


def dp_score(K):
    """순서를 지키는 1대1 부분 짝짓기의 최대 점수. K (..., n, m) -> (...,).
    F[i, j] = max(F[i−1, j], F[i, j−1], F[i−1, j−1] + K[i, j]). 자동 미분의 기울기가 짝지은 칸(0/1)이다."""
    F = K.new_zeros(K.shape[:-2] + (K.shape[-1] + 1,))
    for i in range(K.shape[-2]):
        G = torch.maximum(F[..., 1:], F[..., :-1] + K[..., i, :])
        F = torch.cat([F[..., :1], torch.cummax(G, -1).values], -1)
    return F[..., -1]


@dataclass
class Alignment:
    score: torch.Tensor      # (P,) S
    rotation: torch.Tensor   # (P, 3, 3) B 에 거는 회전 U
    match: torch.Tensor      # (P, n, m) bool


def as_tensor(X, device=None, dtype=None):
    t = torch.as_tensor(np.asarray(X) if not torch.is_tensor(X) else X)
    return t.to(device=device or t.device, dtype=dtype or (t.dtype if t.is_floating_point() else torch.float64))


def pad_stack(structs, n=None):
    """길이가 다른 구조들 [(n_i, 3, 3)] -> (P, N, 3, 3) 영 채움. 거리에 영향이 없다."""
    structs = [as_tensor(s) for s in structs]
    n = n or max(len(s) for s in structs)
    out = structs[0].new_zeros((len(structs), n, 3, 3))
    for i, s in enumerate(structs):
        out[i, :len(s)] = s
    return out


def sq_norm(X):
    return (X ** 2).sum((-1, -2, -3))


def align(A, B, starts=None, n_random=31, iters=10, seed=0):
    """A (P, n, 3, 3), B (P, m, 3, 3) (또는 묶음 없이 (n, 3, 3)). starts (P, s, 3, 3) 를 주지 않으면 항등 + 무작위 n_random 개.
    반환 Alignment (점수, 회전, 짝)."""
    single = A.dim() == 3
    if single:
        A, B = A[None], B[None]
        if starts is not None and starts.dim() == 3:
            starts = starts[None]
    P = len(A)
    if starts is None:
        I = torch.eye(3, dtype=A.dtype, device=A.device).expand(P, 1, 3, 3)
        Rr = random_rotations(P * n_random, seed=seed, dtype=A.dtype, device=A.device).view(P, n_random, 3, 3) if n_random else I[:, :0]
        starts = torch.cat([I, Rr], 1)
    U = starts.clone()
    s = U.shape[1]
    best = torch.full((P, s), -1e30, dtype=A.dtype, device=A.device)
    bU, bG = U.clone(), None
    for it in range(iters + 1):
        with torch.no_grad():
            K = torch.einsum('pnxz,psmxz->psnm', A, torch.einsum('psxy,pmyz->psmxz', U, B))
        K = K.detach().requires_grad_(True)
        with torch.enable_grad():
            S = dp_score(K)
            G, = torch.autograd.grad(S.sum(), K)
        with torch.no_grad():
            imp = S > best + 1e-7 * (1 + best.abs())
            best = torch.where(imp, S, best)
            bU = torch.where(imp[..., None, None], U, bU)
            bG = G if bG is None else torch.where(imp[..., None, None], G, bG)
            if it == iters or not imp.any():
                break
            U = torch.where(imp[..., None, None], procrustes(torch.einsum('psnm,pmxz,pnyz->psxy', G, B, A)), U)
    k = best.argmax(1)
    ar = torch.arange(P, device=A.device)
    out = Alignment(best[ar, k].detach(), bU[ar, k], bG[ar, k] > 0.5)
    if single:
        out = Alignment(out.score[0], out.rotation[0], out.match[0])
    return out


def sq_distance(A, B, normalized=False, **kw):
    """d² (또는 정규화한 d²). A, B 는 (n, 3, 3) 또는 묶음 (P, n, 3, 3)."""
    A, B = as_tensor(A), as_tensor(B)
    B = B.to(A)
    a = align(A, B, **kw)
    z = sq_norm(A) + sq_norm(B)
    d2 = (z - 2 * a.score).clamp_min(0)
    return (d2 / z if normalized else d2), a


def distance(A, B, normalized=False, **kw):
    d2, a = sq_distance(A, B, normalized=normalized, **kw)
    return d2.sqrt(), a


def fixed_sq_distance(A, B, ka, kb, normalized=False):
    """대응을 고정(A 의 변 ka ↔ B 의 변 kb)하고 회전만 최적화한 d². 묶음 A (I, n, 3, 3), B (J, m, 3, 3) 의 모든 쌍 -> (I, J)."""
    A, B = as_tensor(A), as_tensor(B).to(as_tensor(A))
    H = torch.einsum('jkcb,ikab->ijca', B[:, kb], A[:, ka])
    sv = torch.linalg.svdvals(H)
    S = sv[..., 0] + sv[..., 1] + torch.sign(torch.det(H)) * sv[..., 2]
    z = sq_norm(A)[:, None] + sq_norm(B)[None]
    d2 = (z - 2 * S).clamp_min(0)
    return d2 / z if normalized else d2


def match_to_map(match):
    """짝 표시 (n, m) -> A 의 변마다 짝지은 B 의 변 번호 (없으면 −1)."""
    M = match.cpu().numpy() if torch.is_tensor(match) else np.asarray(match)
    mp = -np.ones(M.shape[0], int)
    k, l = np.nonzero(M)
    mp[k] = l
    return mp
