"""Operations on the space: aligned columns, straight lines between strata (geodesics), the mean point of a same-protein ensemble."""
import torch

from .distance import align, as_tensor
from .geometry import procrustes


def aligned_columns(A, B, alignment=None, **kw):
    """Lays two structures out as two sequences (Ca, Cb) with the same number of columns, paired by an alignment.
    An unmatched edge faces a zero matrix in the other sequence. Cb holds B with the rotation U applied.
    Returns Ca, Cb (L, 3, 3) and cols [(k or −1, l or −1)]."""
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
    """Point on the straight line between two column-aligned points: d(Ca, γ(t)) = t·d, d(γ(t), Cb) = (1 − t)·d.
    The straight line between strata is for comparison."""
    return (1 - t) * Ca + t * Cb


def align_to_reference(E, ref):
    """Aligns each structure of an equal-length ensemble E (F, n, 3, 3) to ref (n, 3, 3) with one global rotation
    (correspondence: edge index to edge index)."""
    E, ref = as_tensor(E), as_tensor(ref).to(as_tensor(E))
    U = procrustes(torch.einsum('ikxz,kyz->ixy', E, ref))
    return torch.einsum('ixy,ikyz->ikxz', U, E)


def mean_point(E, tol=1e-10, max_iters=100, start=0, return_info=False):
    """Mean point of a same-protein ensemble E (F, n, 3, 3): repeatedly align every structure to the mean with one
    rotation and average column by column. The correspondence is fixed (edge index to edge index); this is an
    alternating minimisation of Σ_f d_fix(E_f, m)² (both steps have closed forms and the objective never increases).
    Stops when the mean moves by at most tol × (1 + ‖m‖) in one step. start: index of the starting structure, or a point (n, 3, 3).
    The mean point is a point of the space but not a real structure (how much the singular values of a term shrink
    reflects the spread of that edge). Means of structures of different lengths are not defined here.
    That the mean does not depend on the starting point is a measurement, not a proof (6 ensembles, 9 starting points).
    With return_info, returns (m, dict(converged, iters, step))."""
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
