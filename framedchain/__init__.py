"""framedchain: a metric space of framed chains for protein structures and ensembles of any length (its definitions
and proven properties, and the computations and estimators fixed in the accompanying paper). Called framespace up to
version 0.3.3.

A point is a finite sequence of 3×3 matrices; the terms of a real structure are x_i = s_i R_i / 2 (R_i is the frame of
edge i, s_i = l_i / 3.83 Å). Inserting or removing zero matrices, or multiplying every term on the left by the same
rotation, gives the same point. The distance is the minimum over order-preserving partial correspondences and rotations.
An ensemble is a distribution on this metric space; W₂ between ensembles is computed by definition (transport). The
paper's main statistic is the energy statistic Δ²/√n̄ (ensemble). The reduced computation (reduced) and the ruler
family D_{w,ℓ}, D_TM (rulers, needs numba; `from framedchain import rulers`) are the same computations as the code that
produced the paper.
"""
from .geometry import hat, expmap, procrustes, polar, quat_to_mat, mat_to_quat, quat_mul, quat_conj, rotvec_to_quat, random_rotations
from .frames import L0, CIS_MAX, BREAK_MIN, CAPS, frames_from_backbone, to_coordinates, coordinates_from_backbone, edge_lengths, frames_of, is_real, is_cis, is_break, n_edges, ca_trace, read_pdb_backbone, coordinates_from_pdb
from .distance import dp_score, Alignment, align, sq_distance, distance, fixed_sq_distance, length_scaled, sim_long, fixed_scaled, pad_stack, sq_norm, match_to_map, as_tensor
from .certify import bnb
from .operations import aligned_columns, geodesic, align_to_reference, mean_point
from .table import relative_table, determinants, from_table
from .transport import optimal_plan, structure_costs, ensemble_w2
from .parts import split_sq_distance, normalized_distance, near_optimal_alignments
from .reduced import center_path, band_windows, dp_score_band, align_band, self_rotations, MeanStart, pair_scores
from .ensemble import energy_stats, mean_shift2, component_ratio, regime_boundary, frame_pairs, run_pair_means, ensemble_statistic, eta_plus, project_to_real, edge_spread_profile, dynamics_similarity, aligned_energy_stats

__version__ = '0.4.2'
