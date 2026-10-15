"""백본 원자 → 공간의 좌표 (SQFORM_SPACE.md 1.1–1.4절).

변 i 는 Cα_i → Cα_{i+1}. 프레임 R_i = [T, U, V] (열):
  T = 변의 방향, U = 그 변에 걸친 펩타이드 결합의 C=O 방향에서 T 성분을 뺀 뒤 정규화한 것, V = T × U.
좌표는 x_i = s_i · R_i / 2,  s_i = l_i / l0 (l_i 는 변의 길이, 기본 l0 = 3.83 Å). 노름² 은 0.75 s_i².
첫 열에 2·l0 를 곱하면 변의 벡터가 되므로 좌표만으로 Cα 자리가 복원된다.
"""
import numpy as np

L0 = 3.83        # 트랜스인 변의 평균 길이 (Å), 길이 비의 기준
CIS_MAX = 3.45   # 이보다 짧은 변은 시스 (시스 약 3.09 Å, 트랜스 약 3.83 Å 사이가 비어 있다)
CAPS = ('ACE', 'NME', 'NHE')


def frames_from_backbone(CA, C, O):
    """잔기마다의 Cα, C, O (..., L, 3) -> 프레임 R (..., L−1, 3, 3), 변 길이 l (..., L−1)."""
    CA, C, O = (np.asarray(a, dtype=np.float64) for a in (CA, C, O))
    e = CA[..., 1:, :] - CA[..., :-1, :]
    l = np.linalg.norm(e, axis=-1)
    T = e / l[..., None]
    u = O[..., :-1, :] - C[..., :-1, :]
    u = u - (u * T).sum(-1, keepdims=True) * T
    U = u / np.linalg.norm(u, axis=-1, keepdims=True)
    return np.stack([T, U, np.cross(T, U)], -1), l


def to_coordinates(R, l, l0=L0):
    """프레임과 변 길이 -> 좌표 x_i = (l_i / l0) R_i / 2."""
    return (np.asarray(l) / l0)[..., None, None] * np.asarray(R) / 2.0


def coordinates_from_backbone(CA, C, O, l0=L0):
    """잔기마다의 Cα, C, O (..., L, 3) -> 좌표 (..., L−1, 3, 3)."""
    R, l = frames_from_backbone(CA, C, O)
    return to_coordinates(R, l, l0)


def edge_lengths(x, l0=L0):
    """좌표 -> 변의 길이 l_i = 2·l0·|첫 열|. 영 항은 0."""
    return 2.0 * l0 * np.linalg.norm(np.asarray(x)[..., :, 0], axis=-1)


def frames_of(x):
    """실제 구조의 좌표 -> 프레임 R_i 와 길이 비 s_i (= 2·|첫 열|, l0 와 무관)."""
    x = np.asarray(x)
    s = 2.0 * np.linalg.norm(x[..., :, 0], axis=-1)
    return 2.0 * x / s[..., None, None], s


def is_real(x, tol=1e-3):
    """항마다 s·R/2 꼴(s > 0, R 은 회전)인지. 영 항과 일반 행렬은 False."""
    x = np.asarray(x)
    s = 2.0 * np.linalg.norm(x[..., :, 0], axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        R = 2.0 * x / s[..., None, None]
        orth = np.abs(np.swapaxes(R, -1, -2) @ R - np.eye(3)).max((-1, -2)) < tol
        return (s > 0) & orth & (np.linalg.det(R) > 0)


def is_cis(x, cis_max=CIS_MAX, l0=L0):
    """변 길이가 cis_max 보다 짧은 (영이 아닌) 항."""
    l = edge_lengths(x, l0)
    return (l > 0) & (l < cis_max)


def n_edges(x, eps=1e-12):
    """영이 아닌 항의 수 (점 하나 (n, 3, 3) 이 속한 층). 노름² 이 eps 이하인 항을 영으로 본다."""
    return int(((np.asarray(x) ** 2).sum((-1, -2)) > eps).sum())


def ca_trace(x, l0=L0):
    """좌표 (..., n, 3, 3) -> Cα (..., n+1, 3). 첫 Cα 는 원점. 영 항은 길이 0 인 변이다."""
    e = 2.0 * l0 * np.asarray(x)[..., :, :, 0]
    z = np.zeros(e.shape[:-2] + (1, 3))
    return np.concatenate([z, np.cumsum(e, -2)], -2)


def read_pdb_backbone(path, caps=CAPS):
    """PDB (여러 MODEL 가능) -> dict(N, CA, C, O: (F, L, 3), resnames). 네 원자가 모두 있는 잔기만 남기고, caps 에 든 잔기(끝 캡)는 뺀다.
    altLoc 이 여럿이면 처음 나온 것을 쓴다. 잔기는 파일 순서대로 잇고 사슬이 끊긴 곳은 따로 확인하지 않는다(긴 변으로 나타난다)."""
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
                if rn in caps:
                    continue
                key = (line[21], line[22:27])
                if key not in names:
                    names[key] = rn
                    order.append(key)
                if an in ('N', 'CA', 'C', 'O') and an not in cur.get(key, {}):
                    cur.setdefault(key, {})[an] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
    if cur:
        models.append(cur)
    keys = [k for k in order if all(k in m and len(m[k]) == 4 for m in models)]
    out = {a: np.array([[m[k][a] for k in keys] for m in models]) for a in ('N', 'CA', 'C', 'O')}
    out['resnames'] = [names[k] for k in keys]
    return out


def coordinates_from_pdb(path, l0=L0, caps=CAPS):
    """PDB -> 좌표 (F, L−1, 3, 3)."""
    b = read_pdb_backbone(path, caps)
    return coordinates_from_backbone(b['CA'], b['C'], b['O'], l0)
