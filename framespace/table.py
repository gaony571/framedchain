"""같은 점의 불변 표현: 상대 프레임의 표 (SQFORM_SPACE.md 2.7절).

G_ij = x_iᵀ x_j (모든 항 쌍), δ_i = det x_i. 전체 회전(왼쪽 곱)에 정확히 변하지 않는다.
실제 구조에서 G_ii = s_i² I / 4, δ_i = s_i³ / 8 이고, 표에서 프레임 열을 (전체 회전을 빼고) 되찾는다.
없는 항(영행렬)은 표에서 0 인 행과 열이 된다.
"""
import torch

from .distance import as_tensor


def relative_table(x):
    """x (..., n, 3, 3) -> G (..., n, n, 3, 3), G_ij = x_iᵀ x_j."""
    x = as_tensor(x)
    return torch.einsum('...iab,...jac->...ijbc', x, x)


def determinants(x):
    """x (..., n, 3, 3) -> δ (..., n)."""
    return torch.linalg.det(as_tensor(x))


def from_table(G, eps=1e-12):
    """실제 구조의 표 G (n, n, 3, 3) -> 프레임 열 x (n, 3, 3). 전체 회전은 첫 번째 영 아닌 항 k 를 s_k I / 2 로 두는 것으로 정한다.
    G_kk = s_k² I / 4 에서 s_k 를, x_j = (2 / s_k) G_kj 로 나머지를 얻는다."""
    G = as_tensor(G)
    d = torch.diagonal(G, dim1=0, dim2=1)[0, 0]          # 각 항의 G_ii[0, 0] = s_i² / 4
    k = int(torch.nonzero(d > eps)[0])
    sk = 2.0 * d[k].sqrt()
    return (2.0 / sk) * G[k]
