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


def mean_point(E, iters=3):
    """같은 단백질 앙상블의 평균 점: 회전 하나씩으로 맞추고 열마다 산술 평균(되풀이). 평균 점은 공간의 점이지만 실제 구조는 아니다."""
    E = as_tensor(E)
    ref = E[0]
    for _ in range(iters):
        ref = align_to_reference(E, ref).mean(0)
    return ref
