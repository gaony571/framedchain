"""Invariant representation of a point: the table of relative frames.

G_ij = x_iᵀ x_j (all term pairs), δ_i = det x_i. Exactly invariant under global rotations (left multiplication).
For real structures G_ii = s_i² I / 4 and δ_i = s_i³ / 8, and the frame sequence is recovered from the table up to the
global rotation. Missing terms (zero matrices) give zero rows and columns in the table.
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
    """Table G (n, n, 3, 3) of a real structure -> frame sequence x (n, 3, 3). The global rotation is fixed by setting the
    first non-zero term k to s_k I / 2. s_k comes from G_kk = s_k² I / 4 and the other terms from x_j = (2 / s_k) G_kj."""
    G = as_tensor(G)
    d = torch.diagonal(G, dim1=0, dim2=1)[0, 0]          # G_ii[0, 0] = s_i² / 4 for each term
    k = int(torch.nonzero(d > eps)[0])
    sk = 2.0 * d[k].sqrt()
    return (2.0 / sk) * G[k]
