"""공간 위의 연산: 열 맞추기, 층 사이 직선(측지선), 같은 단백질 앙상블의 평균 점."""
import torch

from .distance import align, as_tensor
from .geometry import procrustes


def aligned_columns(A, B, alignment=None, **kw):
    """두 구조를 정렬로 짝지어 같은 열 수의 두 열 (Ca, Cb) 로 늘어놓는다. 짝 없는 변의 상대 열은 영행렬.
    Cb 에는 회전 U 를 건 B 가 들어간다. 반환: Ca, Cb (L, 3, 3), cols [(k 또는 −1, l 또는 −1)]."""
    A, B = as_tensor(A), as_tensor(B).to(as_tensor(A))
    al = alignment or align(A, B, **kw)
    UB = torch.einsum('xy,myz->mxz', al.rotation, B)
    Z = A.new_zeros(3, 3)
    pairs = sorted(zip(*[t.tolist() for t in torch.nonzero(al.match, as_tuple=True)]))
    ca, cb, cols = [], [], []
    i = j = 0
    for k, l in pairs:
        while i < k:
            ca.append(A[i]); cb.append(Z); cols.append((i, -1)); i += 1
        while j < l:
            ca.append(Z); cb.append(UB[j]); cols.append((-1, j)); j += 1
        ca.append(A[k]); cb.append(UB[l]); cols.append((k, l)); i, j = k + 1, l + 1
    while i < len(A):
        ca.append(A[i]); cb.append(Z); cols.append((i, -1)); i += 1
    while j < len(B):
        ca.append(Z); cb.append(UB[j]); cols.append((-1, j)); j += 1
    return torch.stack(ca), torch.stack(cb), cols


def geodesic(Ca, Cb, t):
    """열을 맞춘 두 점 사이 직선 위의 점. d(Ca, γ(t)) = t·d, d(γ(t), Cb) = (1 − t)·d. 층 사이의 직선은 비교용이다."""
    return (1 - t) * Ca + t * Cb


def align_to_reference(E, ref):
    """같은 길이의 앙상블 E (F, n, 3, 3) 의 각 구조를 ref (n, 3, 3) 에 전역 회전 하나로 맞춘다 (대응은 변 번호 그대로)."""
    E, ref = as_tensor(E), as_tensor(ref).to(as_tensor(E))
    U = procrustes(torch.einsum('ikxz,kyz->ixy', E, ref))
    return torch.einsum('ixy,ikyz->ikxz', U, E)


def mean_point(E, tol=1e-10, max_iters=100, start=0, return_info=False):
    """같은 단백질 앙상블 E (F, n, 3, 3) 의 평균 점 (2.4, 3.5절): 각 구조를 평균에 회전 하나로 맞추고 열마다 산술 평균 내기를 되풀이한다.
    대응은 변 번호 그대로 고정이고, Σ_f d_fix(E_f, m)² 를 줄이는 교대 최적화다(두 단계 모두 닫힌 해, 목적함수는 줄기만 한다).
    평균이 한 번에 움직인 거리가 tol × (1 + ‖m‖) 이하가 되면 멈춘다. start: 출발 구조의 번호 또는 점 (n, 3, 3).
    평균 점은 공간의 점이지만 실제 구조는 아니다(항의 특이값이 줄어든 정도가 그 변의 흩어짐). 길이가 다른 구조의 평균은 여기 없다.
    평균이 시작점과 무관하다는 것은 증명이 아니라 측정이다(앙상블 6개, 시작점 9개, 5.1절).
    return_info 이면 (m, dict(converged, iters, step)) 를 돌려준다."""
    E = as_tensor(E)
    ref = E[start] if isinstance(start, int) else as_tensor(start).to(E)
    converged, step = False, float('inf')
    for it in range(1, max_iters + 1):
        new = align_to_reference(E, ref).mean(0)
        step = float((new - ref).pow(2).sum().sqrt())
        ref = new
        if step <= tol * (1 + float(ref.pow(2).sum().sqrt())):
            converged = True
            break
    return (ref, dict(converged=converged, iters=it, step=step)) if return_info else ref
