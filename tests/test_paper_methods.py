"""Tests of the computations and estimators fixed in the paper (reduced, ensemble, parts), on synthetic frame sequences (pytest)."""
import itertools

import numpy as np
import pytest
import torch

import framedchain as fc
from framedchain import ensemble as en
from framedchain import reduced as rd


def chain(n, seed, noise=0.15):
    """A real structure (n, 3, 3) = s R / 2 whose neighboring edges have similar rotations."""
    g = torch.Generator().manual_seed(seed)
    step = fc.expmap(torch.tensor([0.3, 1.2, 0.4], dtype=torch.float64))
    R = [torch.eye(3, dtype=torch.float64)]
    for _ in range(n - 1):
        R.append(R[-1] @ step @ fc.expmap(noise * torch.randn(3, generator=g, dtype=torch.float64)))
    s = 1 + 0.016 * torch.randn(n, generator=g, dtype=torch.float64)
    return s[:, None, None] * torch.stack(R) / 2


def ensemble(base, F, seed, sigma=0.08, rotate=True):
    """F samples with a small rotational fluctuation on every edge of base (n, 3, 3). With rotate, a global rotation per sample is applied too (the same point)."""
    g = torch.Generator().manual_seed(seed)
    W = fc.expmap(sigma * torch.randn(F, len(base), 3, generator=g, dtype=torch.float64))
    E = torch.einsum('nxy,fnyz->fnxz', base, W)
    if rotate:
        E = torch.einsum('fxy,fnyz->fnxz', fc.random_rotations(F, seed=seed + 7), E)
    return E


def insert_edge(base, p, seed=0):
    """The structure with one edge inserted before edge p (n+1, 3, 3)."""
    g = torch.Generator().manual_seed(seed)
    new = base[p] @ fc.expmap(0.5 * torch.randn(3, generator=g, dtype=torch.float64))
    return torch.cat([base[:p], new[None], base[p:]], 0)


# ---------- reduced computation ----------

def test_band_equals_masked_full_dp():
    """The banded DP equals the full DP with matches outside the band blocked (for a monotone center line)."""
    rng = np.random.default_rng(0)
    for trial in range(20):
        n, m, k = rng.integers(5, 25), rng.integers(5, 25), int(rng.integers(1, 4))
        K = torch.tensor(rng.normal(size=(3, n, m)))
        lo, w = rd.band_windows(np.sort(rng.integers(0, m, n)), m, k)
        col = lo[:, None] + np.arange(w)[None]
        mask = torch.full((n, m), -1e9, dtype=torch.float64)
        mask[np.arange(n)[:, None], col] = 0
        assert torch.allclose(rd.dp_score_band(K[:, np.arange(n)[:, None], col], lo), fc.dp_score(K + mask), atol=1e-9), trial


def test_align_band_full_width_matches_align():
    a, b = chain(20, 1), chain(23, 2)
    st = fc.random_rotations(8, seed=3)[None]
    full = fc.align(a[None], b[None], starts=st)
    lo, w = rd.band_windows(np.zeros(20, int), 23, 30)
    S, U, M, edge = rd.align_band(a[None], b[None], lo, w, st)
    assert abs(float(S[0] - full.score[0])) < 1e-9 and bool((M[0] == full.match[0]).all()) and not bool(edge[0])


@pytest.fixture(scope='module')
def two_ensembles():
    base = chain(30, 11)
    return base, ensemble(base, 12, 1), ensemble(insert_edge(base, 15, 4), 12, 2)


def full_pairs(EA, EB, ia, ib):
    d2, _ = fc.sq_distance(EA[ia], EB[ib])
    return d2.sqrt().numpy()


def test_pair_scores_matches_full_and_is_upper_bound(two_ensembles):
    _, A, B = two_ensembles
    ia, ib = np.repeat(np.arange(12), 12), np.tile(np.arange(12), 12)
    r = rd.pair_scores(rd.MeanStart(A, B), ia, ib, k0=3, want_gap=True)
    full = full_pairs(A, B, ia, ib)
    assert np.abs(r['d'] - full).max() < 1e-4 * full.max()
    assert (r['d'] >= full - 1e-9).all()                                    # the band is a more restricted problem, so equal or larger (upper bound)
    assert (r['gap'] <= r['d'] ** 2 + 1e-9).all() and (r['gap'] > 0).all()   # the inserted edge enters the gap part


def test_pair_scores_widens_when_needed():
    """Shifting the center line by the half-width makes the optimal path touch the band edge; widening corrects it."""
    base = chain(25, 5)
    A, B = ensemble(base, 4, 1), ensemble(base, 4, 2)
    ms = rd.MeanStart(A, B)
    ms.center = np.clip(ms.center + 2, 0, 24)
    ia, ib = np.repeat(np.arange(4), 4), np.tile(np.arange(4), 4)
    r = rd.pair_scores(ms, ia, ib, k0=2)
    full = full_pairs(A, B, ia, ib)
    assert r['widened'] > 0 and r['half_width'].max() > 2 and np.abs(r['d'] - full).max() < 1e-4 * full.max()


# ---------- energy statistic ----------

def test_energy_stats_is_cross_minus_floors():
    """Δ² = 2C − B_P − B_Q equals 'cross energy distance − (floor_P + floor_Q)/2' (within-run mean distances cancel), for an arbitrary distance."""
    rng = np.random.default_rng(1)
    runs = [rng.normal(size=(6, 4)) + rng.normal(size=4) for _ in range(5)]   # three runs of P, two of Q
    P, Q = runs[:3], [r + 2.0 for r in runs[3:]]                                  # shift Q so that Δ² > 0 (the ratio is not clipped)
    dd = lambda X, Y: np.sqrt(((X[:, None] - Y[None]) ** 2).sum(-1) + 1.0).mean()   # any distance form
    edist = lambda X, Y: 2 * dd(X, Y) - dd(X, X) - dd(Y, Y)
    cross = np.mean([edist(p, q) for p in P for q in Q])
    fp = np.mean([edist(a, b) for a, b in itertools.combinations(P, 2)])
    fq = np.mean([edist(a, b) for a, b in itertools.combinations(Q, 2)])
    C = np.mean([dd(p, q) for p in P for q in Q])
    BP = np.mean([dd(a, b) for a, b in itertools.combinations(P, 2)])
    BQ = np.mean([dd(a, b) for a, b in itertools.combinations(Q, 2)])
    o = en.energy_stats(C, BP, BQ, n_edges=4)
    assert abs(o['D2'] - (cross - (fp + fq) / 2)) < 1e-12
    assert abs(np.sqrt(o['ratio'] ** 2 + 1) - (o['D2'] / (2 * o['wbar']) + 1)) < 1e-12       # C/w̄ = Δ²/(2w̄) + 1
    assert abs(o['D2_sqrt_n'] - o['D2'] / 2) < 1e-12


def test_mean_shift2_equals_run_mean_shift_in_euclidean():
    """In aligned (Euclidean) coordinates Δμ²₂ equals the mean-shift estimator between run means (cross − mean floor) exactly (the spread terms cancel)."""
    rng = np.random.default_rng(2)
    P = [rng.normal(size=(7, 9)) * 0.3 + rng.normal(size=9) * 0.1 for _ in range(3)]
    Q = [rng.normal(size=(7, 9)) * 0.5 + 1.0 + rng.normal(size=9) * 0.1 for _ in range(4)]
    d2 = lambda X, Y: ((X[:, None] - Y[None]) ** 2).sum(-1).mean()
    m2 = lambda X, Y: ((X.mean(0) - Y.mean(0)) ** 2).sum()
    pairsP, pairsQ, cross = list(itertools.combinations(P, 2)), list(itertools.combinations(Q, 2)), [(p, q) for p in P for q in Q]
    lhs = en.mean_shift2(np.mean([d2(*c) for c in cross]), np.mean([d2(*c) for c in pairsP]), np.mean([d2(*c) for c in pairsQ]), 3)
    rhs = (np.mean([m2(*c) for c in cross]) - (np.mean([m2(*c) for c in pairsP]) + np.mean([m2(*c) for c in pairsQ])) / 2) / 3
    assert abs(lhs - rhs) < 1e-10


def test_ensemble_statistic_null_and_shift():
    """Two ensembles from the same distribution give Δ² near 0; a shifted ensemble gives Δ² > 0 and ratio > 0. The component ratio and the direct ratio agree in direction."""
    base = chain(16, 21)
    N = 6
    E1 = torch.cat([ensemble(base, N, 100 + r, sigma=0.12) for r in range(3)])
    E2 = torch.cat([ensemble(base, N, 200 + r, sigma=0.12) for r in range(3)])
    shifted = base @ fc.expmap(torch.tensor([0.0, 0.0, 0.25], dtype=torch.float64))
    E3 = torch.cat([ensemble(shifted, N, 300 + r, sigma=0.12) for r in range(3)])
    null = en.ensemble_statistic(E1, E2, N, k=30, seed=0)
    alt = en.ensemble_statistic(E1, E3, N, k=30, seed=0)
    assert abs(alt['D2'] - (2 * alt['C'] - alt['B_P'] - alt['B_Q'])) < 1e-12
    assert alt['D2'] > 10 * abs(null['D2']) and alt['ratio'] > null['ratio'] and alt['dmu2'] > null['dmu2']
    rc = en.component_ratio(alt['C'], alt['B_P'], alt['B_Q'], alt['dmu2'], alt['n_edges'])
    assert rc > 0 and alt['n_runs_P'] == 3 and len(alt['runs']['cross']['d']) == 9
    assert abs(en.regime_boundary(1.0, 4) - (np.sqrt(2) - 1)) < 1e-12


def test_eta_plus():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 5))
    De = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))        # the Euclidean distance is of negative type → η₊ = 0
    assert en.eta_plus(De) < 1e-10
    t = np.arange(6.0)
    D4 = np.abs(t[:, None] - t[None]) ** 4                     # exponents above 2 are not of negative type
    assert 0 < en.eta_plus(D4) < 1


# ---------- spread on shared edges ----------

def test_edge_spread_profile_and_similarity():
    base = chain(20, 31)
    E = ensemble(base, 40, 4, sigma=0.1).numpy()
    v = en.edge_spread_profile(E, np.arange(20))
    Er = np.einsum('fxy,fnyz->fnxz', fc.random_rotations(40, seed=9).numpy(), E)   # a global rotation per sample → removed by GPA
    assert np.allclose(en.edge_spread_profile(Er, np.arange(20)), v, rtol=1e-6, atol=1e-10)
    assert np.allclose(en.edge_spread_profile(np.repeat(E[:1], 5, 0), np.arange(20)), 0, atol=1e-20)
    pairs = np.stack([np.arange(20), np.arange(20)], 1)
    assert abs(en.dynamics_similarity(E, E, pairs) - 1) < 1e-12


def test_project_to_real():
    x = chain(12, 41).numpy()
    P, degen = en.project_to_real(x)
    assert np.allclose(P, x, atol=1e-12) and not degen.any()
    m = fc.mean_point(ensemble(chain(12, 41), 30, 5, sigma=0.3)).numpy()
    P2, _ = en.project_to_real(m)
    assert (((P2 - m) ** 2).sum((1, 2)) <= ((x - m) ** 2).sum((1, 2)) + 1e-12).all()   # the nearest s·R/2 for every term
    s = 2 * np.linalg.norm(x[:, :, 0], axis=-1)
    P3, _ = en.project_to_real(m, s=s)
    assert np.allclose(np.linalg.norm(P3[:, :, 0], axis=-1) * 2, s)


# ---------- matched/gap split and mean normalization ----------

def test_split_sq_distance():
    a = chain(20, 51)
    b = insert_edge(a, 9, 2)
    sp = fc.split_sq_distance(a, b)
    d2, _ = fc.sq_distance(a, b)
    assert abs(float(sp['d2'] - d2)) < 1e-9                                       # matched part + gap part = d² at the same alignment
    assert abs(float(sp['gap']) - float((b[9] ** 2).sum())) < 1e-9 and float(sp['matched']) < 1e-9   # only the inserted edge is a gap
    batch = fc.split_sq_distance(torch.stack([a, a]), torch.stack([b[:20], a]))
    assert batch['d2'].shape == (2,) and float(batch['d2'][1]) < 1e-9


def test_normalized_distance():
    xs = [chain(n, 60 + n) for n in (10, 14, 18)]
    D = np.array([[float(fc.normalized_distance(p, q)) for q in xs] for p in xs])
    assert np.allclose(np.diag(D), 0, atol=1e-6) and (D >= 0).all() and (D <= 1 + 1e-12).all()
    assert D[0, 2] <= D[0, 1] + D[1, 2] + 1e-9


# ---------- near-optimal correspondences and the aligned-coordinate version ----------

def test_near_optimal_alignments():
    a = chain(20, 71)
    b = insert_edge(a, 9, 3)
    r = fc.near_optimal_alignments(a, b, eps=0.005)
    d2, al = fc.sq_distance(a, b)
    assert abs(r['S_max'] - float(al.score)) < 1e-9                       # with the same start rule, the same maximum as align
    assert r['n_near'] >= 1 and r['n_distinct'] >= 1
    assert (r['intersection'] <= r['best_match']).all()                  # the intersection lies inside the optimal correspondence
    r0 = fc.near_optimal_alignments(a, b, eps=0.0)
    assert (r0['intersection'] == r0['best_match']).all() or r0['n_distinct'] > 1


def test_aligned_energy_stats_matches_definition():
    rng = np.random.default_rng(5)
    P = [rng.normal(size=(8, 18)) * 0.2 for _ in range(3)]
    Q = [rng.normal(size=(8, 18)) * 0.3 + 0.4 for _ in range(2)]
    o = en.aligned_energy_stats(P, Q, n_edges=2)
    pm = lambda A, B: np.sqrt(((A[:, None] - B[None]) ** 2).sum(-1)).mean()
    C = np.mean([pm(p, q) for p in P for q in Q])
    BP = np.mean([pm(a, b) for a, b in itertools.combinations(P, 2)])
    BQ = pm(Q[0], Q[1])
    assert abs(o['D2'] - (2 * C - BP - BQ)) < 1e-10 and o['dmu'] > 0 and o['logspread'] > 0
    assert abs(o['D2_sqrt_n'] - o['D2'] / np.sqrt(2)) < 1e-12
