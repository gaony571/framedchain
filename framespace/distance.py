"""거리 (SQFORM_SPACE.md 2.1–2.2, 2.5, 3.1–3.2, 3.5절).

점은 3×3 행렬의 유한한 열. 영행렬을 끼우거나 빼도, 모든 항에 같은 회전을 왼쪽에 곱해도 같은 점이다.
    d(x, y)² = min over (순서를 지키는 부분 대응 π, 회전 U) of
               Σ_{(i,j)∈π} ‖x_i − U y_j‖² + Σ_{i 짝 없음} ‖x_i‖² + Σ_{j 짝 없음} ‖y_j‖²
             = ‖x‖² + ‖y‖² − 2 S,   S = max Σ_{(i,j)∈π} ⟨x_i, U y_j⟩
계산: 회전을 고정하면 대응은 동적계획법으로, 대응을 고정하면 회전은 3×3 SVD 로 정확히 나온다. 둘을 번갈아 하며
무작위 회전 여러 개(기본 128개)에서 시작해 가장 좋은 것을 고른다. 계산값은 d 의 상계이고, 전역 최적은 certify.bnb 로 보증한다.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch

from .geometry import procrustes, random_rotations


def dp_score(K):
    """순서를 지키는 1대1 부분 대응의 최대 점수. K (..., n, m) -> (...,).
    F[i, j] = max(F[i−1, j], F[i, j−1], F[i−1, j−1] + K[i, j]). 자동 미분의 기울기가 짝지은 칸(0/1)이다."""
    F = K.new_zeros(K.shape[:-2] + (K.shape[-1] + 1,))
    for i in range(K.shape[-2]):
        G = torch.maximum(F[..., 1:], F[..., :-1] + K[..., i, :])
        F = torch.cat([F[..., :1], torch.cummax(G, -1).values], -1)
    return F[..., -1]


@dataclass
class Alignment:
    score: torch.Tensor      # (P,) S
    rotation: torch.Tensor   # (P, 3, 3) y 에 거는 회전 U
    match: torch.Tensor      # (P, n, m) bool


def as_tensor(x, device=None, dtype=None):
    t = torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x)
    return t.to(device=device or t.device, dtype=dtype or (t.dtype if t.is_floating_point() else torch.float64))


def pad_stack(points, n=None):
    """길이가 다른 점들 [(n_i, 3, 3)] -> (P, N, 3, 3) 영 채움 (같은 점이다)."""
    points = [as_tensor(p) for p in points]
    n = n or max(len(p) for p in points)
    out = points[0].new_zeros((len(points), n, 3, 3))
    for i, p in enumerate(points):
        out[i, :len(p)] = p
    return out


def sq_norm(x):
    return (x ** 2).sum((-1, -2, -3))


def align(x, y, starts=None, n_random=127, iters=10, seed=0):
    """x (P, n, 3, 3), y (P, m, 3, 3) (또는 묶음 없이 (n, 3, 3)). starts (P, s, 3, 3) 를 주지 않으면 항등 + 무작위 n_random 개.
    반환 Alignment (점수 S, 회전 U, 짝)."""
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
            imp = S > best + 1e-7 * (1 + best.abs())
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
    """d² 와 정렬. x, y 는 (n, 3, 3) 또는 묶음 (P, n, 3, 3)."""
    x, y = as_tensor(x), as_tensor(y)
    y = y.to(x)
    a = align(x, y, **kw)
    return (sq_norm(x) + sq_norm(y) - 2 * a.score).clamp_min(0), a


def distance(x, y, **kw):
    d2, a = sq_distance(x, y, **kw)
    return d2.sqrt(), a


def fixed_sq_distance(x, y, kx=None, ky=None):
    """대응을 아는 경우(3.5절): 대응(kx ↔ ky, 기본은 변 번호 그대로)을 고정하고 회전만 최적화한 d_fix². SVD 한 번으로 정확하다.
    묶음 x (I, n, 3, 3), y (J, m, 3, 3) 의 모든 쌍 -> (I, J). d 의 상계다."""
    x, y = as_tensor(x), as_tensor(y).to(as_tensor(x))
    kx = np.arange(x.shape[1]) if kx is None else kx
    ky = np.arange(y.shape[1]) if ky is None else ky
    H = torch.einsum('jkcb,ikab->ijca', y[:, ky], x[:, kx])
    sv = torch.linalg.svdvals(H)
    S = sv[..., 0] + sv[..., 1] + torch.sign(torch.det(H)) * sv[..., 2]
    return (sq_norm(x)[:, None] + sq_norm(y)[None] - 2 * S).clamp_min(0)


# 눈금 (2.5절). n, m 은 영이 아닌 항의 수.
def length_scaled(d, n, m):
    """d / √(0.75 (n + m)): 0 ~ 1. 길이가 다른 쌍을 한 표에 놓을 때."""
    return d / math.sqrt(0.75 * (n + m))


def sim_long(S, n, m):
    """S / (0.75 max(n, m)): 긴 쪽 기준 유사도."""
    return S / (0.75 * max(n, m))


def fixed_scaled(d_fix, n):
    """d_fix / √(2n): 0 ~ 0.87. 한 단백질 안의 눈금."""
    return d_fix / math.sqrt(2 * n)


def match_to_map(match):
    """짝 표시 (n, m) -> x 의 항마다 짝지은 y 의 항 번호 (없으면 −1)."""
    M = match.cpu().numpy() if torch.is_tensor(match) else np.asarray(match)
    mp = -np.ones(M.shape[0], int)
    k, l = np.nonzero(M)
    mp[k] = l
    return mp
