"""framespace: 단백질 백본의 프레임 열 공간.

길이가 다른 모든 백본을 하나의 좌표계(변마다 R/2, 짝 없는 변은 영행렬, 전역 회전으로 나눔)와
하나의 증명된 거리(잔기 단위 거리) 위에 놓는다. 이 패키지에는 확정된 정의와 성질만 둔다.
"""
from .geometry import hat, expmap, procrustes, polar, quat_to_mat, mat_to_quat, quat_mul, quat_conj, rotvec_to_quat, random_rotations
from .frames import frames_from_backbone, embed, unembed, is_real, ca_trace, read_pdb_backbone, from_pdb
from .distance import dp_score, Alignment, align, sq_distance, distance, fixed_sq_distance, pad_stack, sq_norm, match_to_map, as_tensor
from .certify import bnb
from .operations import aligned_columns, geodesic, align_to_reference, mean_point

__version__ = '0.1.0'
