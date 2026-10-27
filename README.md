# framedchain

Framed chains: a metric space for protein structures and ensembles of any length. (Korean version: `README.ko.md`.)

Each virtual Cα–Cα bond becomes a 3×3 matrix, the orthonormal frame of the bond and its peptide plane scaled by bond
length; an absent bond is the zero matrix, so backbones of any length lie in one space. Minimizing the Euclidean
distance between chains over global rotations and insertions of absent bonds gives an order-preserving edit-distance
metric across lengths; at an optimal alignment, which can be certified by branch-and-bound, its square splits exactly
into shared-bond and unshared-bond parts. For ensembles, energy statistics corrected for replicate noise are unbiased
under any underlying metric, and once correspondence and rotation are fixed the space is Euclidean, so mean shifts and
fluctuations can be compared bond by bond.

The package accompanies the paper *Comparing protein conformational ensembles across sequences and lengths* (in preparation). It was called framespace up to version 0.3.3 (same
computations; see `CHANGELOG.md`).

This package contains (i) the definitions and proven properties of the space, and (ii) the computations and estimators that were fixed and used in the accompanying paper (in preparation; reduced alignment, energy statistics, shared-edge spread profiles, matched/gap decomposition, and the ruler family D_{w,ℓ}/D_TM). Pre-registered decision rules, statistical tests (bootstrap, partial rank correlation, …), data readers (mdCATH, ATLAS) and research-stage tools that were not fixed (floor-bias fits, mean/spread decompositions, soft correspondences, barycentres) are *not* included; the analysis scripts of the paper live in the companion reproduction repository.

## What is in the space

- **Coordinates.** For each edge i (the virtual bond Cα_i → Cα_{i+1}) a frame R_i = [T, U, V]: T is the edge direction, U the C=O direction with its T component removed, V = T × U. The term is x_i = s_i R_i / 2 with length ratio s_i = l_i / l0, l0 = 3.83 Å. The first column times 2·l0 is the edge vector, so Cα positions are recovered from the coordinates alone. Edges shorter than 3.45 Å are *cis*.
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
- **Matched and gap parts** (`parts`; the shared-bond and unshared-bond parts of the paper). At the optimal alignment d² splits exactly into a matched part Σ ‖x_i − U y_j‖² and an unmatched (gap) part (norms² of unmatched terms). The gap part contains indels *and* corresponding edges rotated by more than 120°. **Mean-normalised distance** D̂ = d / √(‖x‖² + ‖y‖²) ∈ [0, 1] is a metric at all lengths.
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

framedchain is not on PyPI. Install it from the GitHub repository (a tagged version keeps results reproducible):

```
pip install "framedchain @ git+https://github.com/gaony571/framedchain@v0.4.1"
pip install "framedchain[ot,rulers,test] @ git+https://github.com/gaony571/framedchain@v0.4.1"   # with the extras
```

or from a clone: `pip install -e .` (extras: `.[ot]` weighted transport with POT, otherwise scipy LP; `.[rulers]`
D_{w,ℓ} and D_TM with numba; `.[test]` pytest). Dependencies are numpy, scipy and torch. On Linux without a GPU,
installing torch from the CPU index first avoids the large CUDA wheels:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

Tested with Python 3.12 (numpy 2.5, torch 2.14, scipy 1.18, numba 0.68) on CPU; the code uses no syntax newer than
Python 3.9. GPU use is a matter of passing CUDA tensors (all computations follow the device of the inputs).

## Quick start

```
python examples/quickstart.py      # no input files; compare with examples/quickstart_output.txt
```

It builds two synthetic backbones and prints the distance, its certificate, the matched/gap split, the normalised
distance and the energy statistic between two small ensembles (under a minute on a laptop CPU).

## Use

```python
import framedchain as fc

x = fc.coordinates_from_pdb('a.pdb')            # PDB with F models -> numpy (F, L−1, 3, 3)
y = fc.coordinates_from_backbone(CA, C, O)      # per-residue Cα, C, O (F, L, 3) -> (F, L−1, 3, 3)
d, al = fc.distance(x[0], y[0])                 # distance and alignment (score S, rotation U, matches)
cert = fc.bnb(x[0], y[0])                       # certified global optimum
sp = fc.split_sq_distance(x[0], y[0])           # sp['matched'] + sp['gap'] == d²
M = fc.mean_point(x)                            # mean point of one protein's ensemble
w = fc.ensemble_w2(x, y)                        # W₂ between ensembles

# one ensemble cell as in the paper: runs of 100 frames concatenated in EA and EB
st = fc.ensemble_statistic(EA, EB, n_per_run=100)   # st['D2'], st['D2_sqrt_n'], st['ratio'], st['wbar'], st['dmu2']
near = fc.near_optimal_alignments(mA, mB)           # near['intersection'] for the joint gate
r_dyn = fc.dynamics_similarity(FA, FB, shared_pairs)

from framedchain import rulers                       # needs numba
tm = rulers.tm_align(CA_a, CA_b, d0=rulers.d0_tm(max(len(CA_a), len(CA_b))))   # tm['D2'] = (n + m)/2 − S_TM
```

Inputs and units. A structure with L residues is an array (L − 1, 3, 3), one 3 × 3 term per Cα–Cα edge; batches add a
leading axis and are zero-padded to a common length (`pad_stack`), which does not change any value. Atom coordinates
are in Å. Only PDB files are read (`coordinates_from_pdb`); convert mmCIF files to PDB or pass Cα, C, O arrays to
`coordinates_from_backbone`. High-level functions accept numpy arrays or torch tensors; float64 is recommended (the
package computes in the dtype of the inputs).

## Performance and memory

`distance`/`align` run all starts of a pair together: the score array has starts × n × m entries and is rebuilt at
each of up to `iters` alternations, with one dynamic-programming sweep over the n rows each time. Measured on a laptop
CPU (Intel i5-1335U, 10 threads, float64):

| Pair (n × m edges) | 128 starts (default) | 32 starts (`n_random=31`) | 8 starts (`n_random=7`) |
|---|---|---|---|
| 100 × 110 | 4 s | | |
| 200 × 210 | 22 s | 6.4 s | 2.2 s |
| 400 × 410 | 160 s | | |

- Fewer starts are faster but more likely to stop in a local optimum (the value is always an upper bound on d);
  `bnb` certifies a result. A CUDA GPU speeds this up substantially for long chains and batches.
- Memory: the score array alone is 8 × starts × n × m bytes (about 160 MB for 128 starts and 400 × 400); the backward
  pass of the DP needs a few times that. Batched calls multiply it by the batch size; `ensemble_w2` and
  `structure_costs` take `chunk` to bound it.
- For many pairs from two same-protein ensembles use the reduced computation (`MeanStart` + `pair_scores`, or
  `ensemble_statistic`): one start per pair and a band around the mean-point correspondence. On the same laptop,
  two 100-frame ensembles of 100 edges took 3.5 s for `MeanStart` and then about 950 pairs per second.
- `rulers` compiles its inner loops with numba on first use (about 20 s once; cached afterwards when the package
  directory is writable).

## Reproducibility

Random start rotations come from `seed` (default 0) and frame-pair sampling from `seed`/`rng`, so results are
deterministic for a given input, version and device. Values can differ in the last digits across BLAS/torch builds
or between CPU and GPU. The paper's numbers were produced by framespace 0.2.0 (this package under its earlier name) and the scripts in the companion
repository (see below), not by re-running this package.

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

Every public function and class has an English docstring (`help(fc.distance)`); `docs/API.md` lists them all with
signatures and is regenerated by `python tools/make_api_doc.py`. Changes between versions are in `CHANGELOG.md`.

## Tests

```
pip install -e .[test,rulers]
pytest tests
```

`tests/test_space.py` checks the definitions and properties on synthetic structures (quaternions, Procrustes, atom ↔ coordinate ↔ Cα round trips, PDB round trip, invariances, truncation and edge-shrink laws, triangle inequality, shortest geodesic, d_fix as an upper bound, scales, mean point, invariant table, certificate, optimal transport, ensemble W₂). `tests/test_paper_methods.py` checks the paper's computations (banded DP = blocked full DP, reduced computation equals the full computation and is an upper bound, band widening, Δ² = cross energy distance − mean floor for any distance, the ratio identity, Δμ²₂ = run-mean shift in Euclidean coordinates, null vs shifted ensembles, η₊, spread-profile invariance, projection, matched + gap = d², D̂, near-optimal sets, aligned-coordinate statistics). `tests/test_rulers.py` (skipped without numba) checks the ruler family. `tests/test_usage.py` checks the user-facing paths (numpy and mixed inputs, PDB files on disk, batching with zero padding, ensembles of different lengths). 46 tests, about a minute on a laptop CPU.

## Relation to the code that produced the paper

The paper's analyses ran on framespace 0.2.0 (this package under its earlier name; tag `v0.2.0`, commit 866995c; registered bundle md5 4edba596), the research-stage code `framespace_wip/stage0` (bundle md5 6703fa26) and analysis scripts with registered md5 sums. These are kept, as run, in the companion repository framedchain-paper (https://github.com/gaony571/framedchain-paper, to be made public with the paper), together with a map from each result to its script and output. Version 0.3.x moves the fixed computations into the package without changing the code of the 0.2.0 modules (0.2.0 is tag `v0.2.0`; 0.3.2 only translated docstrings and comments into English, which an AST comparison confirms; 0.3.3 lets `align` accept numpy input, with identical results for tensor input; 0.4.0 renames the package to framedchain):

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

Questions and bug reports: GitHub issues of this repository.
