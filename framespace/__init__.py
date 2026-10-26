"""framespace: 단백질 백본의 프레임 열 공간 (SQFORM_SPACE.md 의 확정된 정의와 성질, 공통 정의역 논문에서 고정한 계산과 추정).

점은 3×3 행렬의 유한한 열이고, 실제 구조의 항은 x_i = s_i R_i / 2 (R_i 는 변 i 의 프레임, s_i = l_i / 3.83 Å).
영행렬을 끼우거나 빼도, 모든 항에 같은 회전을 왼쪽에 곱해도 같은 점이다. 거리는 순서를 지키는 부분 대응과 회전에 대한 최솟값이다.
앙상블은 이 거리 위의 분포이고, 그 사이의 W₂ 는 정의대로 계산한다(transport). 논문의 주 통계는 에너지 통계 Δ²/√n̄ 이다(ensemble).
줄인 계산(reduced)과 자의 묶음 D_{w,ℓ}·D_TM (rulers, numba 필요; `from framespace import rulers`)은 논문 실행본과 같은 계산이다.
"""
from .geometry import hat, expmap, procrustes, polar, quat_to_mat, mat_to_quat, quat_mul, quat_conj, rotvec_to_quat, random_rotations
from .frames import L0, CIS_MAX, CAPS, frames_from_backbone, to_coordinates, coordinates_from_backbone, edge_lengths, frames_of, is_real, is_cis, n_edges, ca_trace, read_pdb_backbone, coordinates_from_pdb
from .distance import dp_score, Alignment, align, sq_distance, distance, fixed_sq_distance, length_scaled, sim_long, fixed_scaled, pad_stack, sq_norm, match_to_map, as_tensor
from .certify import bnb
from .operations import aligned_columns, geodesic, align_to_reference, mean_point
from .table import relative_table, determinants, from_table
from .transport import optimal_plan, structure_costs, ensemble_w2
from .parts import split_sq_distance, normalized_distance, near_optimal_alignments
from .reduced import center_path, band_windows, dp_score_band, align_band, self_rotations, MeanStart, pair_scores
from .ensemble import energy_stats, mean_shift2, component_ratio, regime_boundary, frame_pairs, run_pair_means, ensemble_statistic, eta_plus, project_to_real, edge_spread_profile, dynamics_similarity, aligned_energy_stats

__version__ = '0.3.1'
