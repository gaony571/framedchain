"""Ensemble statistics: the energy statistic over the distance, its components, and spread profiles on shared edges.

For runs (independent replicates) P_1..P_r and Q_1..Q_s, C = mean distance over cross run pairs, and B_P, B_Q = mean
distance over pairs of different runs on the same side.
    Δ² = 2C − B_P − B_Q  (= cross energy distance − (floor_P + floor_Q)/2; within-run mean distances cancel)
Because the energy distance is a symmetric bilinear form in the difference of measures, Δ² is an unbiased estimator of
its population counterpart, defined by the expected frame distribution of one run on each side, when runs are
independent and identically distributed, whatever the negative type of the ground distance (D is not of negative
type). What is lost is only the guarantee ℰ_D ≥ 0; on real samples this is diagnosed by the positive-eigenvalue
fraction η₊ of J D J (eta_plus).
    w̄ = (B_P + B_Q)/2,  ratio = √((C/w̄)² − 1) = √((Δ²/(2w̄) + 1)² − 1)  (the signal-to-noise that follows exactly from cross ÷ within distances)
ratio = 1 lies between the near and far limits, inside the crossing band of the paper (near: ratio < 0.5; crossing:
0.5–2; far: ratio > 2); there Δ²/√n = 2(√2 − 1) w̄/√n. The main statistic is Δ²/√n̄.
    Δμ²₂ = (C₂ − (B₂_P + B₂_Q)/2) / number of edges   (C₂, B₂ are means of squared distances; exponent-2 energy statistic, mean shift)
In aligned (Euclidean) coordinates Δμ²₂ equals the mean shift between run means exactly (the spread terms cancel);
over D it is the generalization.

Estimation is an incomplete U-statistic: for each run pair the reduced-computation distances (reduced) of k frame
pairs (default 2,000, without replacement) are averaged. All same-sequence cells of the paper computed by D use this
(runpair_means and estats in code/r38_s0.py, code/r38_e7_30.py of the framedchain-paper repository). Pairs of
different length use the same formula (the ground distance D is the same everywhere).
"""
import itertools

import numpy as np
import torch

from .reduced import MeanStart, pair_scores


def energy_stats(C, B_P, B_Q, n_edges=None):
    """Mean distances C (cross), B_P, B_Q (different runs on one side) -> dict(C, B_P, B_Q, D2, wbar, ratio[, D2_sqrt_n]).
    D2 = 2C − B_P − B_Q, wbar = (B_P + B_Q)/2, ratio = √(max((C/w̄)² − 1, 0)) (0 if Δ² < 0, nan if w̄ ≤ 0).
    With n_edges the main statistic Δ²/√n̄ is added as D2_sqrt_n."""
    D2 = 2 * C - B_P - B_Q
    w = (B_P + B_Q) / 2
    o = dict(C=float(C), B_P=float(B_P), B_Q=float(B_Q), D2=float(D2), wbar=float(w),
             ratio=float(np.sqrt(max((C / w) ** 2 - 1, 0))) if w > 0 else float('nan'))
    if n_edges:
        o['D2_sqrt_n'] = float(D2 / np.sqrt(n_edges))
    return o


def mean_shift2(C2, B2_P, B2_Q, n_edges):
    """Exponent-2 energy statistic (mean shift, per edge): Δμ²₂ = (C₂ − (B₂_P + B₂_Q)/2) / number of edges.
    C₂ and B₂ are means of squared run-pair distances."""
    return float((C2 - (B2_P + B2_Q) / 2) / n_edges)


def component_ratio(C, B_P, B_Q, dmu2, n_edges):
    """Ratio from the components: √(max(Δμ²₂ · number of edges, 0)/w̄² + ((ρ − 1)/(ρ + 1))²), ρ = B_Q / B_P.
    Where it agrees with the direct ratio (energy_stats) on a same-sequence cell, the concentration approximation and the
    single-branch assumption hold for that cell."""
    w = (B_P + B_Q) / 2
    rho = B_Q / B_P
    return float(np.sqrt(max(dmu2 * n_edges, 0.0) / w ** 2 + ((rho - 1) / (rho + 1)) ** 2))


def regime_boundary(wbar, n_edges):
    """Δ²/√n at ratio = 1, between the near and far limits (inside the paper's crossing band 0.5–2): 2(√2 − 1) w̄/√n."""
    return float(2 * (np.sqrt(2) - 1) * wbar / np.sqrt(n_edges))


def frame_pairs(a, b, n_per_run, k=2000, rng=None):
    """k frame pairs (without replacement) of run a and run b (n_per_run frames each, concatenated in run order in the
    ensemble) -> indices into the ensembles (ia, ib)."""
    rng = np.random.default_rng() if rng is None else rng
    f = rng.choice(n_per_run * n_per_run, k, replace=False)
    return a * n_per_run + f // n_per_run, b * n_per_run + f % n_per_run


def run_pair_means(ms, run_pairs, n_per_run, k=2000, rng=None, want_gap=False, **pair_kw):
    """For each run pair (a, b) in run_pairs, averages the reduced-computation distances of k frame pairs (all solved
    together). ms = MeanStart(EA, EB).
    Returns dict: d, d2 (mean distance and mean squared distance per run pair), widened, half_width_max, ia, ib, group,
    dv (distance per frame pair); with want_gap also gap (mean gap part) and dpair (mean matched-part distance √(d² − gap))."""
    IA, IB, GR = [], [], []
    for g, (a, b) in enumerate(run_pairs):
        ia, ib = frame_pairs(a, b, n_per_run, k, rng)
        IA.append(ia); IB.append(ib); GR.append(np.full(len(ia), g))
    IA, IB, GR = map(np.concatenate, (IA, IB, GR))
    r = pair_scores(ms, IA, IB, want_gap=want_gap, **pair_kw)
    d = r['d']
    G = range(len(run_pairs))
    out = dict(d=[float(d[GR == g].mean()) for g in G], d2=[float((d[GR == g] ** 2).mean()) for g in G],
               widened=r['widened'], half_width_max=int(r['half_width'].max()), ia=IA, ib=IB, group=GR, dv=d)
    if want_gap:
        gap = r['gap']
        pair = np.sqrt(np.maximum(d ** 2 - gap, 0))
        out.update(gap=[float(gap[GR == g].mean()) for g in G], dpair=[float(pair[GR == g].mean()) for g in G])
    return out


def ensemble_statistic(EA, EB, n_per_run, k=2000, seed=0, n_edges=None, rng=None, want_gap=False, **pair_kw):
    """One cell between two ensembles. EA ((number of runs × n_per_run), n, 3, 3) holds n_per_run frames per run,
    concatenated in run order (EB likewise). Cross = all run pairs (a, b); floors = pairs of different runs on one side.
    k frame pairs per run pair, reduced computation (reduced).
    n_edges: number of edges for Δ²/√n̄ and Δμ²₂ (default: mean column count of the two sides; pass it if there are zero terms).
    Returns the energy_stats dict + dmu2 (Δμ²₂) + n_runs_P, n_runs_Q + runs (run-pair means)."""
    rng = np.random.default_rng(seed) if rng is None else rng
    EA = torch.as_tensor(EA)
    EB = torch.as_tensor(EB).to(EA)
    rA, rB = len(EA) // n_per_run, len(EB) // n_per_run
    if rA * n_per_run != len(EA) or rB * n_per_run != len(EB):
        raise ValueError('the number of frames is not a multiple of n_per_run')
    if rA < 2 or rB < 2:
        raise ValueError('the floors need at least two runs on each side')
    ne = n_edges or 0.5 * (EA.shape[1] + EB.shape[1])
    kw = dict(n_per_run=n_per_run, k=k, rng=rng, want_gap=want_gap, **pair_kw)
    fp = run_pair_means(MeanStart(EA, EA), list(itertools.combinations(range(rA), 2)), **kw)
    fq = run_pair_means(MeanStart(EB, EB), list(itertools.combinations(range(rB), 2)), **kw)
    cr = run_pair_means(MeanStart(EA, EB), list(itertools.product(range(rA), range(rB))), **kw)
    o = energy_stats(np.mean(cr['d']), np.mean(fp['d']), np.mean(fq['d']), ne)
    o.update(dmu2=mean_shift2(np.mean(cr['d2']), np.mean(fp['d2']), np.mean(fq['d2']), ne), n_runs_P=rA, n_runs_Q=rB, n_edges=float(ne),
             runs=dict(cross=cr, floor_P=fp, floor_Q=fq))
    return o


def eta_plus(D):
    """Positive mass fraction η₊ = Σ λ₊ / Σ |λ| of the eigenvalues of J D J (J = I − 11ᵀ/N) for a distance matrix D (N, N).
    If D is of negative type, J D J is negative semidefinite and η₊ = 0. A diagnostic of the size of the negative-type
    violation on real samples."""
    D = np.asarray(D, np.float64)
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    ev = np.linalg.eigvalsh(J @ D @ J)
    return float(ev[ev > 0].sum() / np.abs(ev).sum())


def project_to_real(m, s=None, degen_frac=0.5):
    """Projects the terms of a mean point m (n, 3, 3) onto real terms s·R/2. R̄_i is the rotation of the polar
    decomposition (the nearest rotation).
    Without s, the length ratio of the nearest point s_i = 2⟨R̄_i, m_i⟩/3 is used (Frobenius projection; the
    "projection of the mean point" of the coordinate round-trip check); with s, that length ratio is used (for example
    the mean ensemble length ratio 2‖E[..., 0]‖; the projected mean points of the paper's sequence-identity analyses).
    Returns (P (n, 3, 3), degenerate (n,) bool). degenerate = third singular value < degen_frac × s/2
    (the direction of that edge is widely spread in the ensemble and the projection is unstable; the paper's
    sequence-identity analyses removed such edges from the shared bonds, so from both the structural control and r_dyn)."""
    m = np.asarray(m, np.float64)
    U, S, Vt = np.linalg.svd(m)
    dt = np.sign(np.linalg.det(U @ Vt))
    Dg = np.ones((len(m), 3))
    Dg[:, 2] = dt
    R = np.einsum('iab,ib,ibc->iac', U, Dg, Vt)
    s = 2 * np.einsum('nij,nij->n', R, m) / 3 if s is None else np.asarray(s, np.float64)
    return R * (s / 2)[:, None, None], S[:, 2] < degen_frac * s / 2


def edge_spread_profile(F, idx, iters=3):
    """Per-edge spread profile on shared edges. F (N, n, 3, 3) are coordinates of one protein ensemble (aligned to its
    mean point); idx are the shared edge indices. Generalized Procrustes (rotations only, around the own mean) on the
    shared edges for `iters` rounds, then the mean squared scatter around the mean per edge -> (len(idx),)."""
    X = np.asarray(F)[:, idx].astype(np.float64)
    mu = X.mean(0)
    for _ in range(iters):
        A = np.einsum('kiab,icb->kac', X, mu)
        W, S, Zt = np.linalg.svd(A)
        dt = np.sign(np.linalg.det(np.einsum('kba,kcb->kac', Zt, W)))
        Dg = np.ones((len(X), 3))
        Dg[:, 2] = dt
        Uk = np.einsum('kba,kb,kcb->kac', Zt, Dg, W)                  # U = Z D Wᵀ
        X = np.einsum('kac,kicb->kiab', Uk, X)
        mu = X.mean(0)
    return ((X - mu) ** 2).sum((-1, -2)).mean(0)


def dynamics_similarity(Fa, Fb, pairs, iters=3):
    """Dynamics similarity r_dyn of two protein ensembles on shared edges: Spearman rank correlation of the spread
    profiles on the paired shared edges pairs (k, 2). Shared edges are the matches of the optimal alignment between the
    two mean points (π̄). Tied values receive distinct ranks in sort order (no average ranks), as in the paper's code."""
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    va = edge_spread_profile(Fa, pairs[:, 0], iters)
    vb = edge_spread_profile(Fb, pairs[:, 1], iters)
    ra, rb = np.argsort(np.argsort(va)), np.argsort(np.argsort(vb))
    return float(np.corrcoef(ra, rb)[0, 1])


def aligned_energy_stats(P, Q, n_edges):
    """One cell in aligned (Euclidean) coordinates (same sequence, fixed correspondence). P, Q: lists of runs, each run
    (number of frames, 9 · number of edges) of aligned coordinates (aligned to a common reference mean point with
    align_to_reference and flattened). All run pairs, all frame pairs within each run pair.
    Returns the energy_stats dict + a_P, a_Q (within-run mean distances), dmu (mean shift between run means per edge:
    cross − mean floor), logspread (log of the total-spread ratio of Q to P). The same computation as compare in
    code/r38_compare.py of the framedchain-paper repository (the aligned-coordinate statistics of the paper's
    model-error comparison and of the heating cells of the same domains)."""
    P = [np.asarray(a, np.float64) for a in P]
    Q = [np.asarray(b, np.float64) for b in Q]

    def sqd(A, B):
        return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0)

    def pmean(A, B=None):
        if B is None:
            D = np.sqrt(sqd(A, A))
            return float((D.sum() - np.trace(D)) / (len(A) * (len(A) - 1)))
        return float(np.sqrt(sqd(A, B)).mean())

    C = float(np.mean([pmean(a, b) for a in P for b in Q]))
    BP = float(np.mean([pmean(a, b) for a, b in itertools.combinations(P, 2)]))
    BQ = float(np.mean([pmean(a, b) for a, b in itertools.combinations(Q, 2)]))
    mP, mQ = [a.mean(0) for a in P], [b.mean(0) for b in Q]
    m2 = lambda u, v: float(((u - v) ** 2).sum())
    cm = np.mean([m2(u, v) for u in mP for v in mQ])
    fP = np.mean([m2(u, v) for u, v in itertools.combinations(mP, 2)])
    fQ = np.mean([m2(u, v) for u, v in itertools.combinations(mQ, 2)])
    sp = lambda rs: np.mean([((r - r.mean(0)) ** 2).sum(1).mean() for r in rs])
    o = energy_stats(C, BP, BQ, n_edges)
    o.update(a_P=float(np.mean([pmean(a) for a in P])), a_Q=float(np.mean([pmean(b) for b in Q])),
             dmu=float((cm - (fP + fQ) / 2) / n_edges), logspread=float(np.log(sp(Q) / sp(P))))
    return o
