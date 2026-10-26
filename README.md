# framespace

A frame-sequence space for protein backbones: one coordinate system and one proven metric for structures and ensembles of any length. (Korean version: `README.ko.md`.)

This package contains (i) the definitions and proven properties of the space, and (ii) the computations and estimators that were fixed and used in the accompanying paper (in preparation; reduced alignment, energy statistics, shared-edge spread profiles, matched/gap decomposition, and the ruler family D_{w,ℓ}/D_TM). Pre-registered decision rules, statistical tests (bootstrap, partial rank correlation, …), data readers (mdCATH, ATLAS) and research-stage tools that were not fixed (floor-bias fits, mean/spread decompositions, soft correspondences, barycentres) are *not* included; the analysis scripts of the paper live in the companion reproduction repository.

## What is in the space

- **Coordinates.** For each edge i (Cα_i → Cα_{i+1}) a frame R_i = [T, U, V]: T is the edge direction, U the C=O direction with its T component removed, V = T × U. The term is x_i = s_i R_i / 2 with length ratio s_i = l_i / l0, l0 = 3.83 Å. The first column times 2·l0 is the edge vector, so Cα positions are recovered from the coordinates alone. Edges shorter than 3.45 Å are *cis*.
- **Same point.** Inserting or deleting zero matrices, or left-multiplying all terms by one rotation, gives the same point.
- **Distance (proved).** d² = min over (order-preserving partial correspondence, rotation) of [Σ matched ‖x_i − U y_j‖² + Σ unmatched ‖·‖²] = ‖x‖² + ‖y‖² − 2S.
- **Computation.** With the rotation fixed the correspondence is an exact dynamic programme; with the correspondence fixed the rotation is an exact 3×3 SVD. We alternate from 128 starting rotations (identity + 127 random). Computed values are upper bounds.
- **Certificate.** Branch-and-bound over rotation space certifies the global optimum within a tolerance (real structures only).
- **Known correspondence.** d_fix optimises the rotation only (one SVD; an upper bound of d). Use one of d and d_fix within an analysis.
- **Geodesic (proved).** After aligning columns with the optimal correspondence and rotation, the straight segment is a shortest path; unmatched terms shrink to / grow from zero. Interior points are not real structures.
- **Mean point.** For a fixed-correspondence ensemble (one protein), alternate aligning to the mean and term-wise averaging until convergence (minimises Σ d_fix²). Independence of the starting point is measured, not proved.
- **Invariant table.** G_ij = x_iᵀ x_j and δ_i = det x_i are exactly invariant to global rotation; real structures are recovered from the table.
- **Scales.** d / √(0.75(n+m)), sim_long = S / (0.75 max(n, m)), d_fix / √(2n).
- **Ensemble W₂.** Exact optimal transport with d² as the ground cost (a metric because d is a metric; an upper bound because sample-pair d is).
- **PDB reader.** Residues with N, Cα, C and O; multiple MODELs as a batch; terminal caps (ACE, NME, NHE) removed; first altLoc; chain breaks are not checked (they appear as long edges).

Fixed in the paper (0.3.x):
- **Matched and gap parts** (`parts`). At the optimal alignment d² splits exactly into a matched part Σ ‖x_i − U y_j‖² and a gap part (norms² of unmatched terms). The gap part contains indels *and* corresponding edges rotated by more than 120°. **Mean-normalised distance** D̂ = d / √(‖x‖² + ‖y‖²) ∈ [0, 1] is a metric at all lengths.
- **Near-optimal correspondences** (`parts.near_optimal_alignments`). Correspondences of all starts with S ≥ (1 − ε) S_max (ε = 0.005), the number of distinct ones, and their intersection (the joint gate of the paper's P3 and confirmation C analyses).
- **Reduced computation** (`reduced`). The two ensemble mean points are aligned with 128 starts (rotation U0, correspondence π̄); each sample pair then starts from one rotation R_kᵀ U0 R′_l and the correspondence is restricted to a band around π̄ (half-width 5, doubled only for pairs touching the band edge). The banded DP equals the full DP with out-of-band pairs blocked; values are upper bounds. The edge rule is a check, not a guarantee (accuracy was measured against branch-and-bound).
- **Energy statistics** (`ensemble`). From mean distances between independent runs — C (cross) and B_P, B_Q (between different runs of the same side): Δ² = 2C − B_P − B_Q, w̄ = (B_P + B_Q)/2, ratio = √((C/w̄)² − 1), main statistic Δ²/√n̄, exponent-2 version Δμ²₂ = (C₂ − (B₂_P + B₂_Q)/2)/n_edges. By the bilinear identity Δ² is unbiased whatever the negative type of the ground distance. Estimation is an incomplete U-statistic with 2,000 frame pairs per run pair. Also: the negative-type diagnostic η₊ (positive spectral mass of J D J), the component ratio, the regime boundary, and the aligned-coordinate (Euclidean, fixed correspondence) version `aligned_energy_stats`.
- **Shared-edge spread profiles** (`ensemble`). Per-edge spread after generalised Procrustes on the shared edges only, the rank correlation of two profiles (dynamics similarity r_dyn), and projection of a mean point onto real terms s·R/2.
- **Ruler family** (`rulers`, needs numba). D_{w,ℓ} with a position term and the fixed-d₀ TM-type distance D_TM (the TM-align objective is D_TM at fixed d₀), costs by definition, the four TM-score forms, and triangle-violation rates. Values are upper bounds; no certificate.

Not included: mapping interior points back to structures (no threshold has been fixed).

## Tunable values

| Where | Values (default) |
|---|---|
| Coordinates | `l0` (3.83 Å), `cis_max` (3.45 Å), `caps`, `is_real(tol=1e-3)`, `n_edges(eps=1e-12)`, `from_table(eps=1e-12)` |
| Distance (`align`, `sq_distance`, `distance`) | `n_random` (127; 128 starts with the identity), `starts`, `iters` (10), `seed` (0), `improve_tol` (1e-7) |
| Certificate (`bnb`) | `rel_tol` (0.005, sim_long units) or `tol` (absolute S), `n_starts` (128), `level0` (3), `n_refine` (16), `max_cells` (2,000,000), `batch`, `iters` (10), `device` |
| Ensemble W₂ (`ensemble_w2`, `structure_costs`) | `chunk` (32), `device`, `dtype`, distance keywords, weights `wa`, `wb`, precomputed `cost` |
| Optimal transport (`optimal_plan`) | weights `a`, `b`; assignment when unweighted and equal sizes, otherwise POT (if installed) or linear programming |
| Mean point (`mean_point`) | `tol` (1e-10), `max_iters` (100), `start`, `return_info` |
| Reduced computation (`MeanStart`, `pair_scores`) | `n_mean_starts` (128), `k0` (5, initial band half-width), `iters` (10), `want_gap`, `membudget` (3e9 bytes) |
| Energy statistics (`ensemble_statistic`, `run_pair_means`) | `n_per_run` (frames per run; 100 in the paper), `k` (2,000 frame pairs per run pair), `seed` or `rng`, `n_edges` (default: mean column count), reduced-computation keywords |
| Near-optimal set (`near_optimal_alignments`) | `eps` (0.005), `n_random` (127), `iters` (10), `seed` (0) |
| Spread profiles (`edge_spread_profile`, `dynamics_similarity`), projection (`project_to_real`) | `iters` (3), `s` (length ratios; default nearest point), `degen_frac` (0.5) |
| Rulers (`rulers.tm_align`, `rulers.wl_align`) | `d0` or `ell`, `w`; `poses`, `extra_poses`, `iters` (50), `keep` (16), `irls` (3), `search`, `tol` |

If you change `l0`, pass the same value when building coordinates and when reading Cα, edge lengths and *cis* flags.

## Install

```
pip install -e .          # numpy, torch, scipy
pip install -e .[ot]      # weighted transport with POT (otherwise scipy LP)
pip install -e .[test]    # pytest
pip install -e .[rulers]  # D_{w,ℓ} and D_TM (numba)
```

Tested with Python 3.12 (numpy 2.5, torch 2.14, scipy 1.18, numba 0.68); the code uses no syntax newer than Python 3.9.

## Use

```python
import framespace as fs

x = fs.coordinates_from_pdb('a.pdb')            # PDB with F models -> (F, L−1, 3, 3)
y = fs.coordinates_from_backbone(CA, C, O)      # per-residue Cα, C, O (F, L, 3) -> (F, L−1, 3, 3)
d, al = fs.distance(x[0], y[0])                 # distance and alignment (score S, rotation U, matches)
cert = fs.bnb(x[0], y[0])                       # certified global optimum
sp = fs.split_sq_distance(x[0], y[0])           # sp['matched'] + sp['gap'] == d²
M = fs.mean_point(x)                            # mean point of one protein's ensemble
w = fs.ensemble_w2(x, y)                        # W₂ between ensembles

# one ensemble cell as in the paper: runs of 100 frames concatenated in EA and EB
st = fs.ensemble_statistic(EA, EB, n_per_run=100)   # st['D2'], st['D2_sqrt_n'], st['ratio'], st['wbar'], st['dmu2']
near = fs.near_optimal_alignments(mA, mB)           # near['intersection'] for the joint gate
r_dyn = fs.dynamics_similarity(FA, FB, shared_pairs)

from framespace import rulers                       # needs numba
tm = rulers.tm_align(CA_a, CA_b, d0=rulers.d0_tm(max(len(CA_a), len(CA_b))))   # tm['D2'] = (n + m)/2 − S_TM
```

## Modules

| Module | Contents |
|---|---|
| `geometry` | rotations, quaternions, Procrustes, polar decomposition |
| `frames` | PDB reader, frames and coordinates from atoms, edge lengths, *cis*, real-structure test, Cα recovery |
| `distance` | correspondence DP, alternating optimisation (batched, GPU), distance, d_fix, scales, zero padding |
| `certify` | branch-and-bound certificate |
| `operations` | column alignment, geodesic, mean point |
| `table` | relative-frame table, determinants, recovery |
| `transport` | exact optimal transport, ground costs, ensemble W₂ |
| `parts` | matched/gap decomposition, D̂, near-optimal correspondences and their intersection |
| `reduced` | banded DP, banded alignment, mean-point starts (`MeanStart`), reduced computation for pair lists (`pair_scores`) |
| `ensemble` | energy statistics (Δ², w̄, ratio, Δμ²₂, component ratio, regime boundary), incomplete U-statistics, ensemble cells, aligned-coordinate statistics, η₊, mean-point projection, shared-edge spread profiles, r_dyn |
| `rulers` | D_{w,ℓ}, D_TM, d₀(L), TM-score forms, triangle-violation rate (numba; not imported by `__init__`) |

Docstrings and inline comments are in Korean; this README is the English reference for the public API.

## Tests

```
pytest tests
```

`tests/test_space.py` checks the definitions and properties on synthetic structures (quaternions, Procrustes, atom ↔ coordinate ↔ Cα round trips, PDB round trip, invariances, truncation and edge-shrink laws, triangle inequality, shortest geodesic, d_fix as an upper bound, scales, mean point, invariant table, certificate, optimal transport, ensemble W₂). `tests/test_paper_methods.py` checks the paper's computations (banded DP = blocked full DP, reduced computation equals the full computation and is an upper bound, band widening, Δ² = cross energy distance − mean floor for any distance, the ratio identity, Δμ²₂ = run-mean shift in Euclidean coordinates, null vs shifted ensembles, η₊, spread-profile invariance, projection, matched + gap = d², D̂, near-optimal sets, aligned-coordinate statistics). `tests/test_rulers.py` (skipped without numba) checks the ruler family.

## Relation to the code that produced the paper

The paper's analyses ran on framespace 0.2.0 (commit efcdd42; registered bundle md5 4edba596), the research-stage code `framespace_wip/stage0` (bundle md5 6703fa26) and analysis scripts with registered md5 sums. These are kept, as run, in the companion repository framespace-paper (https://github.com/gaony571/framespace-paper, to be made public with the paper), together with a map from each result to its script and output. Version 0.3.x moves the fixed computations into the package without changing the code of the 0.2.0 modules (0.2.0 is tag `v0.2.0`; 0.3.2 only translated docstrings and comments into English, which an AST comparison confirms):

| 0.3.x | As-run code |
|---|---|
| `reduced.center_path` … `MeanStart` | `stage0/band.py` (identical function bodies; package-relative imports) |
| `reduced.pair_scores` | `red_pairs` in `r38_s0.py` (same computation; returns a dict) |
| `ensemble.energy_stats`, `frame_pairs`, `run_pair_means` | `estats`, `upairs`, `runpair_means` in `r38_s0.py` |
| `ensemble.ensemble_statistic`, `mean_shift2` | D-version cells in `r38_e7_30.py` |
| `ensemble.aligned_energy_stats` | `compare` in `r38_compare.py` |
| `ensemble.component_ratio`, `regime_boundary` | `e0_dom` in `r39_core.py` |
| `ensemble.eta_plus` | `r38_b6.py` |
| `ensemble.edge_spread_profile`, `dynamics_similarity`, `project_to_real(s=…)` | `gpa_profile`, `spearman` and projected mean points in `r38_p3.py` (default projection scale as in `r39_roundtrip.py`) |
| `parts.split_sq_distance` | gap part as defined in `red_pairs` (full alignment) |
| `parts.near_optimal_alignments` | `align_all` in `r38_s0.py`, π̄ step in `r38_p3.py` |
| `rulers` | `stage0/rulers.py` |

Frame-pair sampling order differs between scripts (seeds derived from domain names), so exact paper values are reproduced with the as-run scripts.

## Licence and citation

MIT licence, copyright (c) 2026 Hyeonsu An (`LICENSE`). Please cite the paper and this software (`CITATION.cff`).
