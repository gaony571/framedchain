"""줄인 계산 (정의역 문서 4.2절): 평균 점 출발과 띠 제한 DP. 같은 단백질 앙상블 표본 쌍의 거리를 대량으로 낼 때 쓴다.

띠 DP. 짝을 띠 B = {(i, j): lo_i ≤ j < lo_i + w} 안에서만 허용한 순서 지킨 부분 대응의 최대 점수.
lo 가 줄지 않으면(중심선이 단조이면) 행마다 띠 안의 칸만 저장해도 정확하다:
  F_i[j] (j 는 1 기반, y[:j]) 는 j < lo_i 에서 F_{i−1}[j] 와 같고, j > lo_{i−1} + w 에서는 F_{i−1}[lo_{i−1} + w] 와 같다
  (앞 행들의 띠가 그보다 오른쪽에 없으므로). 그래서 앞 행의 띠 값을 오른쪽으로 상수 연장해 옮기면 된다.
띠 안의 최적은 전체 DP 최적 이하다(짝을 더 좁힌 문제). 그러므로 이 값으로 낸 거리도 상계다.
"가장자리에 닿으면 넓힌다" 는 점검이지 정확성의 보장이 아니다. 정확성은 전체 계산·분기 한정 값과 견주어 잰다
(공통 정의역 논문: 상계 편향 = 2 × (교차 오차 − 바닥 오차) 를 BnB 와 견줌, 결과 문서 2.2).

평균 점 출발. 같은 단백질의 앙상블 표본은 평균 점과 변 번호를 공유한다. 두 평균 점을 시작점 넉넉히 맞춰 회전 U0 와
대응 π̄ 를 얻고, 표본 k 를 자기 평균에 맞추는 회전 R_k 를 미리 구하면, 쌍 (k, l) 의 출발 회전은 R_k^T U0 R'_l 이다.

공통 정의역 논문의 앙상블 계산은 모두 이 규칙이다: 평균 점끼리 128 출발, 표본 쌍마다 출발 1 개, 띠 반폭 5 에서 시작해
가장자리에 닿은 쌍만 반폭을 두 배로(열 수까지), float64. 이 모듈의 함수는 그 실행본(framespace_wip/stage0/band.py,
code/r38_s0.py 의 red_pairs)과 같은 계산이다.
"""
import numpy as np
import torch

from .distance import align, as_tensor, sq_norm
from .geometry import procrustes, random_rotations
from .operations import mean_point


def center_path(match, m):
    """평균 점끼리의 짝 (n, m) bool -> 행마다의 중심 열 (n,) (단조). 짝 없는 행은 앞뒤 짝 열 사이를 선형으로 채운다."""
    M = match.cpu().numpy() if torch.is_tensor(match) else np.asarray(match)
    n = M.shape[0]
    rows, cols = np.nonzero(M)
    if len(rows) == 0:
        return np.round(np.linspace(0, m - 1, n)).astype(int)
    c = np.interp(np.arange(n), rows, cols)
    return np.maximum.accumulate(np.clip(np.round(c).astype(int), 0, m - 1))


def band_windows(center, m, k):
    """중심 열 (n,) 과 반폭 k -> 띠의 왼쪽 끝 lo (n,) (줄지 않음) 와 폭 w = min(2k+1, m)."""
    w = int(min(2 * k + 1, m))
    lo = np.clip(np.asarray(center) - k, 0, m - w)
    return np.maximum.accumulate(lo).astype(int), w


def dp_score_band(Kb, lo):
    """띠 안의 점수 Kb (..., n, w) (칸 (i, t) 는 열 lo_i + t) -> 최대 점수 (...,). 자동 미분의 기울기가 짝지은 칸이다."""
    n, w = Kb.shape[-2], Kb.shape[-1]
    lo = torch.as_tensor(np.asarray(lo), device=Kb.device)
    F = Kb.new_zeros(Kb.shape[:-2] + (w + 1,))
    ar = torch.arange(w + 1, device=Kb.device)
    lo_p = lo[0]
    for i in range(n):
        idx = (ar + (lo[i] - lo_p)).clamp(max=w)          # 앞 행의 띠를 이 행의 띠 자리로 (오른쪽은 상수 연장)
        Fp = F[..., idx]
        G = torch.maximum(Fp[..., 1:], Fp[..., :-1] + Kb[..., i, :])
        F = torch.cat([Fp[..., :1], torch.cummax(G, -1).values], -1)
        lo_p = lo[i]
    return F[..., -1]


def align_band(x, y, lo, w, starts, iters=10, improve_tol=1e-7):
    """띠 DP 로 대응과 회전을 번갈아 푼다. x (P, n, 3, 3), y (P, m, 3, 3), starts (P, s, 3, 3), 띠 (lo, w) 는 모든 쌍에 같다.
    반환: 점수 S (P,), 회전 U (P, 3, 3), 짝 (P, n, m) bool, 띠 가장자리에 짝이 닿았는지 (P,) bool (행렬 끝은 빼고)."""
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
    """앙상블 E (F, n, 3, 3) 의 각 표본을 평균 점 m 에 맞추는 회전 R_k (F, 3, 3): R_k E_k ≈ m (대응은 변 번호 그대로)."""
    return procrustes(torch.einsum('ikxz,kyz->ixy', E, m))


class MeanStart:
    """두 앙상블의 평균 점 정렬을 한 번 하고, 표본 쌍마다의 출발 회전과 띠를 만든다.
    n_mean_starts: 평균 점끼리 맞출 때의 출발 회전 수. 평균 점은 실제 구조가 아니므로 보증 대신 넉넉히 쓴다.
    같은 앙상블 안의 쌍(바닥)은 MeanStart(E, E) 로 만든다."""

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
        """쌍 (ia[p], ib[p]) 의 출발 회전 (P, n_local, 3, 3): 첫째는 R_k^T U0 R'_l, 나머지는 무작위."""
        U = torch.einsum('pyx,yz,pzw->pxw', self.RA[ia], self.U0, self.RB[ib])[:, None]
        if n_local > 1:
            P = len(ia)
            Rr = random_rotations(P * (n_local - 1), seed=seed, dtype=U.dtype, device=U.device).view(P, n_local - 1, 3, 3)
            U = torch.cat([U, Rr], 1)
        return U


def pair_scores(ms, ia, ib, k0=5, iters=10, want_gap=False, membudget=3e9):
    """줄인 계산을 표본 쌍 목록 (ia[p], ib[p]) 에 쓴다. ms = MeanStart(EA, EB).
    띠 반폭 k0 에서 시작해 가장자리에 닿은 쌍만 반폭을 두 배로 넓혀 다시 푼다(띠가 열 수에 닿을 때까지).
    membudget: 한 번에 푸는 쌍 수를 정하는 메모리 어림(바이트).
    반환 dict: d (P,) 거리(상계), gap (P,) 빈칸 몫(짝 없는 항의 노름² 합; want_gap 일 때, 아니면 None), S (P,),
    half_width (P,) 마지막 반폭, widened (처음 띠에서 넓힌 쌍 수)."""
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
