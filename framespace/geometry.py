"""회전과 사원수 도구. 행렬은 (..., 3, 3), 사원수는 (w, x, y, z) 순서다."""
import torch


def hat(v):
    """회전 벡터 (..., 3) -> 반대칭 행렬 (..., 3, 3)."""
    O = torch.zeros_like(v[..., 0])
    return torch.stack([torch.stack([O, -v[..., 2], v[..., 1]], -1),
                        torch.stack([v[..., 2], O, -v[..., 0]], -1),
                        torch.stack([-v[..., 1], v[..., 0], O], -1)], -2)


def expmap(w):
    """회전 벡터 -> 회전 행렬."""
    return torch.linalg.matrix_exp(hat(w))


def procrustes(M):
    """argmax_{U in SO(3)} tr(U M). M = W S Vh 이면 U = V diag(1, 1, det) Wᵀ."""
    W, S, Vh = torch.linalg.svd(M)
    V = Vh.transpose(-1, -2)
    D = torch.ones_like(S)
    D[..., -1] = torch.sign(torch.det(V @ W.transpose(-1, -2)))
    return (V * D[..., None, :]) @ W.transpose(-1, -2)


def polar(G):
    """일반 3×3 행렬에 가장 가까운 회전과 그 크기(평균 특이값). 영행렬 쪽으로 줄어든 열은 크기가 작다."""
    W, S, Vh = torch.linalg.svd(G)
    D = torch.ones_like(S)
    D[..., -1] = torch.sign(torch.det(W @ Vh))
    return (W * D[..., None, :]) @ Vh, S.mean(-1)


def quat_to_mat(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(R):
    """회전 행렬 -> 단위 사원수 (부호는 w ≥ 0 쪽이 아니라 계산된 갈래를 따른다)."""
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
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], -1)


def quat_conj(q):
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def rotvec_to_quat(r):
    th = r.norm(dim=-1, keepdim=True)
    k = torch.where(th > 1e-12, torch.sin(th / 2) / th.clamp_min(1e-12), torch.full_like(th, 0.5))
    return torch.cat([torch.cos(th / 2), k * r], -1)


def random_rotations(n, seed=None, generator=None, dtype=torch.float64, device='cpu'):
    """균일한 무작위 회전 n 개. 난수는 CPU 에서 만들고 device 로 옮긴다."""
    if generator is None:
        generator = torch.Generator().manual_seed(0 if seed is None else seed)
    q = torch.randn(n, 4, generator=generator, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    return quat_to_mat(q).to(device=device, dtype=dtype)
