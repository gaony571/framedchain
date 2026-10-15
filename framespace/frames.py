"""백본 원자 → 공간의 좌표.

변 i 는 CA_i → CA_{i+1}. 프레임 R_i = [T, U, V] (열):
  T = 변 방향, U = (O_i − C_i) 의 T 에 수직인 성분 방향, V = T × U.
공간의 좌표는 변마다 R_i / 2 (노름² 3/4). 변 길이 l_i 는 좌표에 들어가지 않고 따로 둔다.
"""
import numpy as np


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def frames_from_backbone(N, CA, C, O):
    """잔기마다의 원자 (..., L, 3) -> 프레임 R (..., L−1, 3, 3), 변 길이 l (..., L−1). N 은 쓰지 않는다(형식을 맞추려고 받는다)."""
    CA, C, O = (np.asarray(a, dtype=np.float64) for a in (CA, C, O))
    e = CA[..., 1:, :] - CA[..., :-1, :]
    l = np.linalg.norm(e, axis=-1)
    T = e / l[..., None]
    u = O[..., :-1, :] - C[..., :-1, :]
    u = u - (u * T).sum(-1, keepdims=True) * T
    U = _unit(u)
    return np.stack([T, U, np.cross(T, U)], -1), l


def embed(R):
    """프레임 -> 공간의 좌표 (R / 2)."""
    return np.asarray(R) / 2.0


def unembed(X):
    return 2.0 * np.asarray(X)


def is_real(X, tol=1e-3):
    """변마다 R/2 꼴(직교, det > 0)인지. 영 채움 열과 일반 행렬은 False."""
    R = 2.0 * np.asarray(X)
    orth = np.abs(np.swapaxes(R, -1, -2) @ R - np.eye(3)).max((-1, -2)) < tol
    return orth & (np.linalg.det(R) > 0)


def ca_trace(X, lengths):
    """좌표 X (..., n, 3, 3) = R/2 와 변 길이 (..., n) -> CA (..., n+1, 3). 첫 CA 는 원점."""
    X = np.asarray(X)
    e = 2.0 * X[..., :, :, 0] * np.asarray(lengths)[..., None]
    z = np.zeros(e.shape[:-2] + (1, 3))
    return np.concatenate([z, np.cumsum(e, -2)], -2)


def read_pdb_backbone(path):
    """PDB (여러 MODEL 가능) -> dict(N, CA, C, O: (F, L, 3), resnames). 네 원자가 모두 있는 잔기만, 끝 캡(ACE/NME/NHE)은 뺀다."""
    models, cur, order, names = [], {}, [], {}
    with open(path) as f:
        for line in f:
            if line.startswith('MODEL'):
                cur = {}
            elif line.startswith('ENDMDL'):
                models.append(cur)
                cur = {}
            elif line.startswith(('ATOM', 'HETATM')):
                an, rn = line[12:16].strip(), line[17:20].strip()
                if rn in ('ACE', 'NME', 'NHE'):
                    continue
                key = (line[21], line[22:27])
                if key not in names:
                    names[key] = rn
                    order.append(key)
                if an in ('N', 'CA', 'C', 'O'):
                    cur.setdefault(key, {})[an] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
    if cur:
        models.append(cur)
    keys = [k for k in order if all(k in m and len(m[k]) == 4 for m in models)]
    out = {a: np.array([[m[k][a] for k in keys] for m in models]) for a in ('N', 'CA', 'C', 'O')}
    out['resnames'] = [names[k] for k in keys]
    return out


def from_pdb(path):
    """PDB -> 좌표 X (F, L−1, 3, 3) 와 변 길이 (F, L−1)."""
    b = read_pdb_backbone(path)
    R, l = frames_from_backbone(b['N'], b['CA'], b['C'], b['O'])
    return embed(R), l
