# framedchain 0.4.2 — API reference

Generated from the docstrings by `tools/make_api_doc.py`. Every name listed here except the `rulers` module is
importable from the top level (`import framedchain as fc; fc.distance(...)`); `rulers` needs numba and is
imported explicitly (`from framedchain import rulers`).

Conventions: a structure with L residues is an array (L − 1, 3, 3) of edge coordinates (one 3 × 3 term per
Cα–Cα edge); batches add a leading axis and are zero-padded to a common length (`pad_stack`); ensembles are
(frames, n, 3, 3). Distances are in units of the coordinate norm (a trans edge has squared norm 0.75); Cα
positions and PDB coordinates are in Å. High-level functions accept numpy arrays or torch tensors; functions
marked *(tensor)* are low-level building blocks that expect torch tensors. float64 is recommended.

## `frames`

Backbone atoms -> coordinates of the space.

Edge i is Cα_i → Cα_{i+1}. Its frame R_i = [T, U, V] (columns):
  T = direction of the edge, U = C=O direction of the peptide bond across the edge with its T component removed and
  normalized, V = T × U.
Coordinates are x_i = s_i · R_i / 2, s_i = l_i / l0 (l_i is the edge length; default l0 = 3.83 Å). The squared norm is 0.75 s_i².
The first column times 2·l0 is the edge vector, so the Cα positions are recovered from the coordinates alone.

### `frames_from_backbone(CA, C, O)`

Per-residue Cα, C, O (..., L, 3) -> frames R (..., L−1, 3, 3) and edge lengths l (..., L−1).

### `to_coordinates(R, l, l0=3.83)`

Frames and edge lengths -> coordinates x_i = (l_i / l0) R_i / 2.

### `coordinates_from_backbone(CA, C, O, l0=3.83)`

Per-residue Cα, C, O (..., L, 3) -> coordinates (..., L−1, 3, 3).

### `edge_lengths(x, l0=3.83)`

Coordinates -> edge lengths l_i = 2·l0·|first column|. Zero terms give 0.

### `frames_of(x)`

Coordinates of a real structure -> frames R_i and length ratios s_i (= 2·|first column|, independent of l0).

### `is_real(x, tol=0.001)`

Whether each term has the form s·R/2 (s > 0, R a rotation). Zero terms and general matrices give False.

### `is_cis(x, cis_max=3.45, l0=3.83)`

(Non-zero) terms whose edge length is shorter than cis_max.

### `is_break(x, break_min=4.2, l0=3.83)`

Terms whose edge length exceeds break_min (4.2 Å by default). Consecutive Cα are about 3.8 Å apart (less across a cis
peptide), so a longer edge joins residues on either side of a gap in the chain.

### `n_edges(x, eps=1e-12)`

Number of non-zero terms (the stratum of a point (n, 3, 3)). Terms with squared norm ≤ eps count as zero.

### `ca_trace(x, l0=3.83)`

Coordinates (..., n, 3, 3) -> Cα (..., n+1, 3), first Cα at the origin. A zero term is an edge of length 0.

### `read_pdb_backbone(path, caps=('ACE', 'NME', 'NHE'))`

PDB (several MODELs allowed) -> dict(N, CA, C, O: (F, L, 3), resnames). Keeps residues with all four atoms and
drops the residues in caps (terminal caps). For several altLocs the first one is used. Residues are joined in file
order; chain breaks are not checked here (they appear as long edges, which is_break flags).

### `coordinates_from_pdb(path, l0=3.83, caps=('ACE', 'NME', 'NHE'))`

PDB -> coordinates (F, L−1, 3, 3).

## `distance`

The distance of the space.

A point is a finite sequence of 3×3 matrices. Inserting or removing zero matrices, or multiplying every term on
the left by the same rotation, gives the same point.
    d(x, y)² = min over (order-preserving partial correspondence π, rotation U) of
               Σ_{(i,j)∈π} ‖x_i − U y_j‖² + Σ_{i unmatched} ‖x_i‖² + Σ_{j unmatched} ‖y_j‖²
             = ‖x‖² + ‖y‖² − 2 S,   S = max Σ_{(i,j)∈π} ⟨x_i, U y_j⟩
Computation: for a fixed rotation the correspondence is exact by dynamic programming; for a fixed correspondence the
rotation is exact by a 3×3 SVD. The two are alternated from the identity and 127 random start rotations (128 starts by
default) and the best is kept. The computed value is an upper bound on d; certify.bnb certifies the global optimum.

### `dp_score(K)` *(tensor)*

Maximum score of an order-preserving one-to-one partial correspondence. K (..., n, m) -> (...,).
F[i, j] = max(F[i−1, j], F[i, j−1], F[i−1, j−1] + K[i, j]). The autograd gradient is the matched cells (0/1).

### `Alignment(score: torch.Tensor, rotation: torch.Tensor, match: torch.Tensor) -> None`

Result of align: score S (P,), rotation U (P, 3, 3) applied to y, and matches (P, n, m) bool.

### `as_tensor(x, device=None, dtype=None)`

numpy array or tensor -> tensor (non-floating inputs become float64; device and dtype kept unless given).

### `pad_stack(points, n=None)`

Points of different lengths [(n_i, 3, 3)] -> (P, N, 3, 3), padded with zeros (the same points).

### `sq_norm(x)`

Squared norm ‖x‖² of a point (n, 3, 3), or of each point of a batch (P, n, 3, 3) (tensor).

### `align(x, y, starts=None, n_random=127, iters=10, seed=0, improve_tol=1e-07)`

x (P, n, 3, 3), y (P, m, 3, 3) (or unbatched (n, 3, 3)). Without starts (P, s, 3, 3): identity + n_random random rotations.
iters: maximum number of alternations between correspondence and rotation. improve_tol: a start stops when it
improves by less than this (relative). seed: seed of the random start rotations.
Inputs may be numpy arrays or tensors; y and starts are converted to the dtype and device of x.
Returns an Alignment (score S, rotation U, matches).

### `sq_distance(x, y, **kw)`

d² and the alignment. x, y are (n, 3, 3) or batches (P, n, 3, 3).

### `distance(x, y, **kw)`

Distance d(x, y) and the alignment (score S, rotation U applied to y, matches). x (n, 3, 3) and y (m, 3, 3), or
batches (P, n, 3, 3), (P, m, 3, 3) zero-padded to a common length; numpy arrays or tensors. **kw goes to align
(n_random, iters, seed, starts). The value is an upper bound on d (certify with bnb).

### `fixed_sq_distance(x, y, kx=None, ky=None)`

When the correspondence is known: d_fix² with the correspondence (kx ↔ ky; default: edge index to edge index)
fixed and only the rotation optimized; exact with one SVD.
All pairs of batches x (I, n, 3, 3), y (J, m, 3, 3) -> (I, J). An upper bound on d.

### `length_scaled(d, n, m)`

d / √(0.75 (n + m)): between 0 and 1. For tables of pairs of different lengths.

### `sim_long(S, n, m)`

S / (0.75 max(n, m)): similarity normalized by the longer structure.

### `fixed_scaled(d_fix, n)`

d_fix / √(2n): between 0 and 0.87. A scale within one protein.

### `match_to_map(match)`

Match matrix (n, m) -> for each term of x the index of the matched term of y (−1 if unmatched).

## `certify`

Certificate of the global optimum (branch and bound over rotations).

For terms of real structures x_k = s_k R_k / 2, the score of a term pair is ⟨x_k, U y_l⟩ = s_k t_l (⟨c_kl, u⟩² − 1/4)
(c_kl = q_k ⊗ conj(p_l) is the quaternion of the two frames, u is the quaternion of U, s and t are length ratios).
If the rotation-vector cube [−π, π]³ is divided into cubic cells, every rotation in a cell lies within the angle
ρ = √3 σ / 2 on S³ of the cell center u0, so
score ≤ s_k t_l (cos²(max(0, φ0 − ρ)) − 1/4)   (φ0 = arccos|⟨c, u0⟩|; the inequality holds because length ratios are positive).
The correspondence DP is monotone in the scores, so this gives an upper bound for the cell. Cells whose upper bound is
below the best value + tolerance are discarded and the remaining cells are split into 8.
Only real structures (every term s·R/2 with s > 0) are accepted.

### `bnb(x, y, rel_tol=0.005, tol=None, max_cells=2000000, n_starts=128, level0=3, n_refine=16, batch=None, iters=10, device=None, verbose=False)`

x (n, 3, 3) and y (m, 3, 3) are real structures. Certifies the global maximum of S within a tolerance.
The tolerance is tol (absolute, in units of S) or rel_tol × 0.75 max(n, m) (sim_long units; default 0.005).
n_starts: number of start rotations for the initial lower bound. level0: initial grid (2^level0 cells per axis).
n_refine: number of cells whose lower bound is refined at each level.
max_cells: stop (certificate fails) if more cells than this remain. batch: cells per batch (default fitted to n·m).
Returns S_heur (value from n_starts start rotations), S_lb, S_ub, tol, certified, the grid level, number of DPs, time.

## `operations`

Operations on the space: aligned columns, straight lines between strata (geodesics), the mean point of a same-protein ensemble.

### `aligned_columns(A, B, alignment=None, **kw)`

Lays two structures out as two sequences (Ca, Cb) with the same number of columns, paired by an alignment.
An unmatched edge faces a zero matrix in the other sequence. Cb holds B with the rotation U applied.
Returns Ca, Cb (L, 3, 3) and cols [(k or −1, l or −1)].

### `geodesic(Ca, Cb, t)`

Point on the straight line between two column-aligned points: d(Ca, γ(t)) = t·d, d(γ(t), Cb) = (1 − t)·d.
The straight line between strata is for comparison.

### `align_to_reference(E, ref)`

Aligns each structure of an equal-length ensemble E (F, n, 3, 3) to ref (n, 3, 3) with one global rotation
(correspondence: edge index to edge index).

### `mean_point(E, tol=1e-10, max_iters=100, start=0, return_info=False)`

Mean point of a same-protein ensemble E (F, n, 3, 3): repeatedly align every structure to the mean with one
rotation and average column by column. The correspondence is fixed (edge index to edge index); this is an
alternating minimization of Σ_f d_fix(E_f, m)² (both steps have closed forms and the objective never increases).
Stops when the mean moves by at most tol × (1 + ‖m‖) in one step. start: index of the starting structure, or a point (n, 3, 3).
The mean point is a point of the space but not a real structure (how much the singular values of a term shrink
reflects the spread of that edge). Means of structures of different lengths are not defined here.
That the mean does not depend on the starting point is a measurement, not a proof: for six simulated protein ensembles
(153–391 edges, 500 structures each), mean points from nine starting structures agreed to within 8e-11 after one rotation
(measured with version 0.2.0, whose results this version reproduces exactly).
With return_info, returns (m, dict(converged, iters, step)).

## `table`

Invariant representation of a point: the table of relative frames.

G_ij = x_iᵀ x_j (all term pairs), δ_i = det x_i. Exactly invariant under global rotations (left multiplication).
For real structures G_ii = s_i² I / 4 and δ_i = s_i³ / 8, and the frame sequence is recovered from the table up to the
global rotation. Missing terms (zero matrices) give zero rows and columns in the table.

### `relative_table(x)`

x (..., n, 3, 3) -> G (..., n, n, 3, 3), G_ij = x_iᵀ x_j.

### `determinants(x)`

x (..., n, 3, 3) -> δ (..., n).

### `from_table(G, eps=1e-12)`

Table G (n, n, 3, 3) of a real structure -> frame sequence x (n, 3, 3). The global rotation is fixed by setting the
first non-zero term k to s_k I / 2. s_k comes from G_kk = s_k² I / 4 and the other terms from x_j = (2 / s_k) G_kj.

## `transport`

W₂ between ensembles (optimal transport over the distance of the space).

An ensemble is a (weighted) empirical distribution of structures. The ground cost is d² for every sample pair, and
    W₂² = min over transport plans P (marginals equal to the weights of the two ensembles) of Σ P_kl d²(A_k, B_l).
Because the ground distance is a metric, W₂ is a metric. The sample-pair d comes from distance.align (an upper bound),
so the resulting W₂ is an upper bound too. To put ensembles of different lengths in one table use the scale
distance.length_scaled(W2, n, m). Only the computation by definition is here; estimation conventions (sample sizes,
the reduced computation, floors) are not included.

### `optimal_plan(C, a=None, b=None)`

Exact optimal transport plan (I, J) for the cost matrix C (I, J). a, b are the weights of the two sides (need not sum to 1).
Without weights and with I = J this is an assignment problem (scipy); otherwise POT is used if installed, else a
linear program (scipy HiGHS).

### `structure_costs(EA, EB, chunk=32, device=None, dtype=None, **align_kw)`

d² (I, J) (numpy) for all sample pairs of two ensembles EA (I, n, 3, 3), EB (J, m, 3, 3).
Pairs are solved by align in chunks of `chunk` (align_kw: n_random, iters, seed, improve_tol). Memory scales with
chunk × number of starts × n × m.

### `ensemble_w2(EA, EB, wa=None, wb=None, cost=None, **kw)`

W₂ between ensembles. wa, wb are sample weights (uniform if omitted). If cost is given it is used as the ground
cost (d²); otherwise it is computed by structure_costs(**kw).
Returns dict(W2, W2sq, plan, cost).

## `parts`

Splitting and normalizing the distance.

Matched part and gap part. At an optimal alignment (π, U)
    d² = Σ_{(i,j)∈π} ‖x_i − U y_j‖² + Σ_{unmatched i} ‖x_i‖² + Σ_{unmatched j} ‖y_j‖²  =  matched part + gap part
exactly. Replacing one match by two gaps lowers the cost when θ > 120° (whatever the length ratio), so the optimal
alignment breaks such a match: the gap part can contain not only insertions and deletions but also edges that
correspond in sequence and are rotated by more than 120°, unless such an edge is matched to another partner (gap part ≠
number of insertions).
The split depends on the alignment, so when several alignments are nearly optimal their variation is reported separately.

Normalized distance. D̂ = d / √(‖x‖² + ‖y‖²) is a metric at every length with values in [0, 1] (embedding in the
unit sphere of a Hilbert space).

### `split_sq_distance(x, y, alignment=None, **kw)`

Splits d² into the matched part and the gap part. x (n, 3, 3), y (m, 3, 3) or batches (P, n, 3, 3), (P, m, 3, 3).
Without alignment, align(x, y, **kw) is used (upper-bound computation).
Returns dict: d2, matched (matched part), gap (gap part), n_matched (number of matches), gap_share (gap part / d²,
0 if d² = 0), alignment.

### `normalized_distance(x, y, d=None, **kw)`

Normalized distance D̂ = d / √(‖x‖² + ‖y‖²) ∈ [0, 1]. Without d, the distance is computed (upper bound).

### `near_optimal_alignments(x, y, eps=0.005, n_random=127, iters=10, seed=0, improve_tol=1e-07)`

Nearly optimal correspondences and their intersection (in the paper, the intersection of near-optimal
alignments in the decision rules of the primary and confirmatory sequence-identity tests, and the
alignment-uncertainty check of S2 Text).
Runs the same computation as align (identity + n_random random starts, alternating correspondence and rotation) on
one pair x (n, 3, 3), y (m, 3, 3), and collects the starts whose final score satisfies S ≥ (1 − eps) S_max as
"nearly optimal".
Returns dict: S_max, best_match (n, m) bool, scores (number of starts,), near (indices of nearly optimal starts),
n_near, n_distinct (number of different correspondences found by the nearly optimal starts), intersection (n, m)
bool (matches present in every nearly optimal correspondence).
The same computation as align_all in code/r38_s0.py and the π̄ step in code/r38_p3.py of the framedchain-paper
repository (only the seed of the random start rotations is passed as seed).

## `reduced`

The reduced computation: mean-point starts and band-limited DP. Used to compute distances for many sample pairs of
same-protein ensembles.

Banded DP. The maximum score of an order-preserving partial correspondence whose matches are restricted to the band
B = {(i, j): lo_i ≤ j < lo_i + w}. If lo never decreases (the center line is monotone), storing only the band cells of
each row is exact:
  F_i[j] (j 1-based, y[:j]) equals F_{i−1}[j] for j < lo_i, and equals F_{i−1}[lo_{i−1} + w] for j > lo_{i−1} + w
  (the bands of earlier rows lie no further right). So the previous row's band values are moved over, extended to the
  right as a constant.
The optimum within the band is at most the full DP optimum (a more restricted problem), so distances computed from it
are upper bounds. "Widen the band when a match touches its edge" is a check, not a guarantee of exactness. Exactness is
measured against the full computation and branch and bound (in the paper: the upper-bound bias = 2 × (cross error −
floor error) against branch and bound).

Mean-point starts. Samples of a same-protein ensemble share the mean point and the edge numbering. Aligning the two
mean points generously (many starts) gives a rotation U0 and a correspondence π̄; with the rotation R_k that aligns
sample k to its own mean computed in advance, the start rotation of the pair (k, l) is R_k^T U0 R'_l.

Every ensemble computation of the paper follows this rule: 128 starts between the mean points, one start per sample
pair, band half-width 5 at first (20 for the pairs of the comparison across sequences with TM-score below 0.3, the
setting chosen after the bias check by the rule fixed in a later record, before the re-measurement; S2 Text 2 of the
paper) and doubled (up to the number of columns) only
for pairs touching the band edge, float64. The functions here are the same computation as the code that ran
(framespace_wip/stage0/band.py and red_pairs in code/r38_s0.py of the framedchain-paper repository).

### `center_path(match, m)` *(tensor)*

Matches between the mean points (n, m) bool -> center column per row (n,) (monotone). Unmatched rows are filled
linearly between the neighboring matched columns.

### `band_windows(center, m, k)` *(tensor)*

Center columns (n,) and half-width k -> left ends of the band lo (n,) (non-decreasing) and width w = min(2k+1, m).

### `dp_score_band(Kb, lo)` *(tensor)*

Scores inside the band Kb (..., n, w) (cell (i, t) is column lo_i + t) -> maximum score (...,).
The autograd gradient is the matched cells.

### `align_band(x, y, lo, w, starts, iters=10, improve_tol=1e-07)` *(tensor)*

Alternates correspondence (banded DP) and rotation. x (P, n, 3, 3), y (P, m, 3, 3), starts (P, s, 3, 3); the band
(lo, w) is the same for all pairs.
Returns score S (P,), rotation U (P, 3, 3), matches (P, n, m) bool, and whether a match touched the band edge (P,)
bool (matrix borders excluded).

### `self_rotations(E, m)` *(tensor)*

Rotations R_k (F, 3, 3) aligning each sample of ensemble E (F, n, 3, 3) to its mean point m: R_k E_k ≈ m
(correspondence: edge index to edge index).

### `MeanStart(EA, EB, n_mean_starts=128, mean_kw=None, seed=0)`

Aligns the mean points of two ensembles once and provides start rotations and the band for every sample pair.
n_mean_starts: number of start rotations for aligning the mean points. A mean point is not a real structure, so many
starts are used instead of a certificate. Pairs within one ensemble (floors) use MeanStart(E, E).

#### `MeanStart.starts(self, ia, ib, n_local=1, seed=0)`

Start rotations (P, n_local, 3, 3) for the pairs (ia[p], ib[p]): the first is R_k^T U0 R'_l, the rest are random.

### `pair_scores(ms, ia, ib, k0=5, iters=10, want_gap=False, membudget=3000000000.0)`

Applies the reduced computation to the sample pairs (ia[p], ib[p]). ms = MeanStart(EA, EB).
Starts with band half-width k0 and re-solves, with the half-width doubled, only the pairs touching the band edge
(until the band reaches the number of columns).
membudget: memory estimate (bytes) that sets how many pairs are solved at once.
Returns dict: d (P,) distances (upper bounds), gap (P,) gap part (sum of squared norms of unmatched terms; with
want_gap, otherwise None), S (P,), half_width (P,) final half-width, widened (number of pairs widened from the first band).

## `ensemble`

Ensemble statistics: the energy statistic over the distance, its components, and spread profiles on shared edges.

For runs (independent replicates) P_1..P_r and Q_1..Q_s, C = mean distance over cross run pairs, and B_P, B_Q = mean
distance over pairs of different runs on the same side. Symbols of the paper: ratio is κ, and B_Q is B_P′.
    Δ² = 2C − B_P − B_Q  (= cross energy distance − (floor_P + floor_Q)/2; within-run mean distances cancel)
Because the energy distance is a symmetric bilinear form in the difference of measures, Δ² is an unbiased estimator of
its population counterpart, defined by the expected frame distribution of one run on each side, when runs are
independent and identically distributed, whatever the negative type of the ground distance (d is not of negative
type). What is lost is only the guarantee ℰ_D ≥ 0; on real samples this is diagnosed by the positive-eigenvalue
fraction η₊ of J D J (eta_plus).
    w̄ = (B_P + B_Q)/2,  ratio = √((C/w̄)² − 1) = √((Δ²/(2w̄) + 1)² − 1)  (the signal-to-noise that follows exactly from cross ÷ within distances)
ratio = 1 lies between the near and far limits, inside the crossing band of the paper (near: ratio < 0.5; crossing:
0.5–2; far: ratio > 2); there Δ²/√n = 2(√2 − 1) w̄/√n. The main statistic is Δ²/√n̄.
    Δμ²₂ = (C₂ − (B₂_P + B₂_Q)/2) / number of edges   (C₂, B₂ are means of squared distances; exponent-2 energy statistic, mean shift)
In aligned (Euclidean) coordinates Δμ²₂ equals the mean shift between run means exactly (the spread terms cancel);
over d it is the generalization.

Estimation is an incomplete U-statistic: for each run pair the reduced-computation distances (reduced) of k frame
pairs (default 2,000, without replacement) are averaged. All same-sequence cells of the paper computed with d use this
(runpair_means and estats in code/r38_s0.py, code/r38_e7_30.py of the framedchain-paper repository). Pairs of
different length use the same formula (the ground distance d is the same everywhere).

### `energy_stats(C, B_P, B_Q, n_edges=None)`

Mean distances C (cross), B_P, B_Q (different runs on one side) -> dict(C, B_P, B_Q, D2, wbar, ratio[, D2_sqrt_n]).
D2 = 2C − B_P − B_Q, wbar = (B_P + B_Q)/2, ratio = √(max((C/w̄)² − 1, 0)) (0 if Δ² < 0, nan if w̄ ≤ 0).
With n_edges the main statistic Δ²/√n̄ is added as D2_sqrt_n.

### `mean_shift2(C2, B2_P, B2_Q, n_edges)`

Exponent-2 energy statistic (mean shift, per edge): Δμ²₂ = (C₂ − (B₂_P + B₂_Q)/2) / number of edges.
C₂ and B₂ are means of squared run-pair distances.

### `component_ratio(C, B_P, B_Q, dmu2, n_edges)`

Ratio from the components: √(max(Δμ²₂ · number of edges, 0)/w̄² + ((ρ − 1)/(ρ + 1))²), ρ = B_Q / B_P.
Where it agrees with the direct ratio (energy_stats) on a same-sequence cell, the concentration approximation and the
single-branch assumption hold for that cell.

### `regime_boundary(wbar, n_edges)`

Δ²/√n at ratio = 1, between the near and far limits (inside the paper's crossing band 0.5–2): 2(√2 − 1) w̄/√n.

### `frame_pairs(a, b, n_per_run, k=2000, rng=None)`

k frame pairs (without replacement) of run a and run b (n_per_run frames each, concatenated in run order in the
ensemble) -> indices into the ensembles (ia, ib).

### `run_pair_means(ms, run_pairs, n_per_run, k=2000, rng=None, want_gap=False, **pair_kw)`

For each run pair (a, b) in run_pairs, averages the reduced-computation distances of k frame pairs (all solved
together). ms = MeanStart(EA, EB).
Returns dict: d, d2 (mean distance and mean squared distance per run pair), widened, half_width_max, ia, ib, group,
dv (distance per frame pair); with want_gap also gap (mean gap part) and dpair (mean matched-part distance √(d² − gap)).

### `ensemble_statistic(EA, EB, n_per_run, k=2000, seed=0, n_edges=None, rng=None, want_gap=False, **pair_kw)`

One cell between two ensembles. EA ((number of runs × n_per_run), n, 3, 3) holds n_per_run frames per run,
concatenated in run order (EB likewise). Cross = all run pairs (a, b); floors = pairs of different runs on one side.
k frame pairs per run pair, reduced computation (reduced).
n_edges: number of edges for Δ²/√n̄ and Δμ²₂ (default: mean column count of the two sides; pass it if there are zero terms).
Returns the energy_stats dict + dmu2 (Δμ²₂) + n_runs_P, n_runs_Q + runs (run-pair means).

### `eta_plus(D)`

Positive mass fraction η₊ = Σ λ₊ / Σ |λ| of the eigenvalues of J D J (J = I − 11ᵀ/N) for a distance matrix D (N, N).
If D is of negative type, J D J is negative semidefinite and η₊ = 0. A diagnostic of the size of the negative-type
violation on real samples.

### `project_to_real(m, s=None, degen_frac=0.5)`

Projects the terms of a mean point m (n, 3, 3) onto real terms s·R/2. R̄_i is the rotation of the polar
decomposition (the nearest rotation).
Without s, the length ratio of the nearest point s_i = 2⟨R̄_i, m_i⟩/3 is used (Frobenius projection; the
"projection of the mean point" of the coordinate round-trip check); with s, that length ratio is used (for example
the mean ensemble length ratio 2‖E[..., 0]‖; the projected mean points of the paper's sequence-identity analyses).
Returns (P (n, 3, 3), degenerate (n,) bool). degenerate = third singular value < degen_frac × s/2
(the direction of that edge is widely spread in the ensemble and the projection is unstable; the paper's
sequence-identity analyses removed such edges from the shared bonds, so from both the structural control and r_dyn).

### `edge_spread_profile(F, idx, iters=3)`

Per-edge spread profile on shared edges. F (N, n, 3, 3) are coordinates of one protein ensemble (aligned to its
mean point); idx are the shared edge indices. Generalized Procrustes (rotations only, around the own mean) on the
shared edges for `iters` rounds, then the mean squared scatter around the mean per edge -> (len(idx),).

### `dynamics_similarity(Fa, Fb, pairs, iters=3)`

Dynamics similarity r_dyn of two protein ensembles on shared edges: Spearman rank correlation of the spread
profiles on the paired shared edges pairs (k, 2). Shared edges are the matches of the optimal alignment between the
two mean points (π̄). Tied values receive distinct ranks in sort order (no average ranks), as in the paper's code.

### `aligned_energy_stats(P, Q, n_edges)`

One cell in aligned (Euclidean) coordinates (same sequence, fixed correspondence). P, Q: lists of runs, each run
(number of frames, 9 · number of edges) of aligned coordinates (aligned to a common reference mean point with
align_to_reference and flattened). All run pairs, all frame pairs within each run pair.
Returns the energy_stats dict + a_P, a_Q (within-run mean distances), dmu (mean shift between run means per edge:
cross − mean floor), logspread (log of the total-spread ratio of Q to P). The same computation as compare in
code/r38_compare.py of the framedchain-paper repository (the aligned-coordinate statistics of the paper's
model-error comparison and of the heating cells of the same domains).

## `geometry`

Rotation and quaternion tools. Matrices are (..., 3, 3); quaternions are ordered (w, x, y, z).

### `hat(v)` *(tensor)*

Rotation vector (..., 3) -> skew-symmetric matrix (..., 3, 3).

### `expmap(w)` *(tensor)*

Rotation vector -> rotation matrix.

### `procrustes(M)` *(tensor)*

argmax_{U in SO(3)} tr(U M). If M = W S Vh then U = V diag(1, 1, det) Wᵀ.

### `polar(G)` *(tensor)*

Nearest rotation to a general 3×3 matrix and its size (mean singular value). Terms shrunk towards zero have a small size.

### `quat_to_mat(q)` *(tensor)*

Unit quaternion (..., 4) in (w, x, y, z) order -> rotation matrix (..., 3, 3).

### `mat_to_quat(R)` *(tensor)*

Rotation matrix -> unit quaternion (the sign follows the branch that is computed, not w ≥ 0).

### `quat_mul(a, b)` *(tensor)*

Quaternion product a ⊗ b (..., 4).

### `quat_conj(q)` *(tensor)*

Quaternion conjugate (w, −x, −y, −z).

### `rotvec_to_quat(r)` *(tensor)*

Rotation vector (..., 3) -> unit quaternion (..., 4).

### `random_rotations(n, seed=None, generator=None, dtype=torch.float64, device='cpu')`

n uniformly random rotations. Random numbers are drawn on the CPU and moved to device.

## `rulers`

The rulers module: the distance D_TM at fixed d₀ (the TM-align objective as an edit-distance metric, Theorem 2 of
the paper) and D_{w,ℓ} (frame terms with a position term; not used in the paper).

The general lemma of the paper. If the features of terms a and the distance δ between them are invariant under the
rigid motions G, the gap cost c is G-invariant, and c(a) ≤ δ(a, b) + c(b), then
    D² = min over (order-preserving partial correspondence π, g ∈ G) of Σ_{(i,j)∈π} δ(a_i, g b_j)² + Σ_{unmatched} c²
is a pseudometric on the quotient (identifying insertion/removal of zero-cost terms and a common g); it is a metric
when, in addition, the gap cost vanishes only at the zero term and the minimum over G is attained (for D_TM the
minimum over translations is attained because the score vanishes at infinite separation).

D_{w,ℓ}. Feature of edge i: (w·x_i, p_i/ℓ) (x_i = s_i R_i/2, p_i the midpoint of the edge); δ is their Euclidean
distance and c = w‖x‖.
    D² = w²(‖X‖² + ‖Y‖²) − S,  S = max Σ_{(i,j)∈π} [2w²⟨x_i, U y_j⟩ − ‖p_i − U q_j − t‖²/ℓ²].
  As ℓ → ∞ this is the basic distance (framedchain.sq_distance) times w². The value at (w, ℓ) is w²·D²(1, wℓ), so the
  only effective parameter is wℓ.
D_TM. Feature of residue i: P_i (the Cα position); δ(a, b) = f(‖a − b‖/d₀), f(r) = r/√(1 + r²) (concave, increasing,
f(0) = 0, hence a metric); c² = 1/2.
    1 − δ² = 1/(1 + r²) is the TM residue score, so D_TM² = (n + m)/2 − S_TM,  S_TM = max Σ_{(i,j)∈π} 1/(1 + r_ij²).
  At fixed d₀ it is a metric (the objective of TM-align = D_TM at fixed d₀). TM-score adds d₀ = d₀(L), division by a
  length and the 1 − S transform; the non-metric behavior of 1 − TM comes from the shorter-length normalization and the
  1 − S transform (the factorial comparison of the paper).

Computation. For a fixed pose (U, t) the correspondence is exact by dynamic programming. For a fixed correspondence
  D_{w,ℓ}: t is the difference of the centroids of the matched midpoints and U is exact from the SVD of the combined
  cross-covariance;
  D_TM: weighted least squares (weights 1/(1 + r²)², a majorization by the upper tangent of the concave function), so
  the value never decreases.
The two are alternated from many poses and the best is kept. Starts: fragment-pair superpositions (the local
superposition of TM-align), gapless threading, (search) superpositions of sub-fragments of the found correspondence
(the search of the TM-score program), and poses given from outside (e.g. the TM-align pose).
The computed value is an upper bound on D; there is no certificate (branch and bound). The inner loops run in numba
(pip install framedchain[rulers]). This module is the same computation as framespace_wip/stage0/rulers.py used for the
paper (the comparisons of TM-score forms, the chaining test and the ranking of existing measures).

### `edge_positions(x)`

Coordinates (n, 3, 3) -> edge midpoints (n, 3) (Å, first Cα at the origin).

### `d0_tm(L)`

Length scale of TM-score d₀(L) = 1.24 (L − 15)^{1/3} − 1.8 (Å), not below 0.5 (as in TM-align).

### `start_poses(P, Q, f=8, per_side=12, n_thread=24, n_random=8, seed=0)`

Start poses [(U, t)]: superpositions of fragment pairs of length f (about per_side start points per side), n_thread gapless threadings, and random rotations with centroids matched.

### `sub_fragment_poses(P, Q, ii, jj, min_len=4, n_pos=8)`

Poses superposing contiguous parts of the found correspondence (lengths k, k/2, k/4, … down to min_len, n_pos positions each) (the search of TM-score).

### `tm_align(P, Q, d0, poses=None, extra_poses=(), iters=50, keep=16, irls=3, search=True, tol=1e-10)`

S_TM at fixed d₀: max Σ_{matches} 1/(1 + ‖P_i − U Q_j − t‖²/d₀²). P (n, 3), Q (m, 3) are Cα positions (Å).
poses: list of start poses (default start_poses); extra_poses: poses to add (e.g. the TM-align pose); search: one
more round of sub-fragment search on the found correspondence.
-> dict(S, D2 = (n + m)/2 − S, U, t, match (n, m) bool).

### `tm_cost(P, Q, match, U, t, d0)`

D_TM² at a given correspondence and pose (by definition): Σ_matches r²/(1 + r²) + (number of unmatched terms)/2.

### `wl_align(x, y, ell, w=1.0, p=None, q=None, poses=None, extra_poses=(), iters=50, keep=16, search=True, tol=1e-10)`

D_{w,ℓ}² and the alignment. x (n, 3, 3), y (m, 3, 3) coordinates; p, q edge midpoints (computed from the
coordinates if omitted). With ell = inf there is no position term (basic distance × w²; starts: identity + 127
random rotations). Zero terms are removed beforehand (the value is the same).
-> dict(S, D2, U, t, match: (n, m) in the original indices).

### `wl_cost(x, y, match, U, t, ell, w=1.0, p=None, q=None)`

D_{w,ℓ}² at a given correspondence and pose (by definition).

### `tm_forms(S, n, m)`

S_TM (computed at the given d₀) -> the four forms. At fixed d₀, 'sqrt_raw' (D_TM) and 'sqrt_norm' (D_TM/√L̄) are
metrics (the theorem and the normalized metric of the paper); the linear forms 'lin_raw' and 'lin_norm' are not.

### `triangle_rate(Dm, rel_tol=1e-09, abs_tol=1e-09, max_triples=None, seed=0)`

For a symmetric matrix Dm (N, N), checks for every triple whether the longest side exceeds the sum of the other two.
-> (violation rate, number of violations, number of triples, maximum relative violation).
Relative violation = (longest − sum) / longest. A violation is longest − sum > abs_tol + rel_tol·longest. With
max_triples, only that many random triples are checked.
