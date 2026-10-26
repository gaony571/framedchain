"""Rotation and quaternion tools. Matrices are (..., 3, 3); quaternions are ordered (w, x, y, z)."""
import torch


def hat(v):
    """Rotation vector (..., 3) -> skew-symmetric matrix (..., 3, 3)."""
    O = torch.zeros_like(v[..., 0])
    return torch.stack([torch.stack([O, -v[..., 2], v[..., 1]], -1),
                        torch.stack([v[..., 2], O, -v[..., 0]], -1),
                        torch.stack([-v[..., 1], v[..., 0], O], -1)], -2)


def expmap(w):
    """Rotation vector -> rotation matrix."""
    return torch.linalg.matrix_exp(hat(w))


def procrustes(M):
    """argmax_{U in SO(3)} tr(U M). If M = W S Vh then U = V diag(1, 1, det) Wᵀ."""
    W, S, Vh = torch.linalg.svd(M)
    V = Vh.transpose(-1, -2)
    D = torch.ones_like(S)
    D[..., -1] = torch.sign(torch.det(V @ W.transpose(-1, -2)))
    return (V * D[..., None, :]) @ W.transpose(-1, -2)


def polar(G):
    """Nearest rotation to a general 3×3 matrix and its size (mean singular value). Terms shrunk towards zero have a small size."""
    W, S, Vh = torch.linalg.svd(G)
    D = torch.ones_like(S)
    D[..., -1] = torch.sign(torch.det(W @ Vh))
    return (W * D[..., None, :]) @ Vh, S.mean(-1)


def quat_to_mat(q):
    """Unit quaternion (..., 4) in (w, x, y, z) order -> rotation matrix (..., 3, 3)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(R):
    """Rotation matrix -> unit quaternion (the sign follows the branch that is computed, not w ≥ 0)."""
    R = torch.as_tensor(R)
    r00, r01, r02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    r10, r11, r12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    r20, r21, r22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    sq = lambda v: torch.sqrt(torch.clamp(v, min=1e-12))
    s0 = sq(1 + r00 + r11 + r22) * 2
    s1 = sq(1 + r00 - r11 - r22) * 2
    s2 = sq(1 + r11 - r00 - r22) * 2
    s3 = sq(1 + r22 - r00 - r11) * 2
    q0 = torch.stack([s0 / 4, (r21 - r12) / s0, (r02 - r20) / s0, (r10 - r01) / s0], -1)
    q1 = torch.stack([(r21 - r12) / s1, s1 / 4, (r01 + r10) / s1, (r02 + r20) / s1], -1)
    q2 = torch.stack([(r02 - r20) / s2, (r01 + r10) / s2, s2 / 4, (r12 + r21) / s2], -1)
    q3 = torch.stack([(r10 - r01) / s3, (r02 + r20) / s3, (r12 + r21) / s3, s3 / 4], -1)
    tr = r00 + r11 + r22
    c0 = (tr > 0)[..., None]
    c1 = ((r00 > r11) & (r00 > r22))[..., None]
    c2 = (r11 > r22)[..., None]
    q = torch.where(c0, q0, torch.where(c1, q1, torch.where(c2, q2, q3)))
    return q / q.norm(dim=-1, keepdim=True)


def quat_mul(a, b):
    """Quaternion product a ⊗ b (..., 4)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], -1)


def quat_conj(q):
    """Quaternion conjugate (w, −x, −y, −z)."""
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def rotvec_to_quat(r):
    """Rotation vector (..., 3) -> unit quaternion (..., 4)."""
    th = r.norm(dim=-1, keepdim=True)
    k = torch.where(th > 1e-12, torch.sin(th / 2) / th.clamp_min(1e-12), torch.full_like(th, 0.5))
    return torch.cat([torch.cos(th / 2), k * r], -1)


def random_rotations(n, seed=None, generator=None, dtype=torch.float64, device='cpu'):
    """n uniformly random rotations. Random numbers are drawn on the CPU and moved to device."""
    if generator is None:
        generator = torch.Generator().manual_seed(0 if seed is None else seed)
    q = torch.randn(n, 4, generator=generator, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    return quat_to_mat(q).to(device=device, dtype=dtype)
