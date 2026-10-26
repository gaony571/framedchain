"""앙상블 통계 (정의역 문서 2.7절, 시연 설계 1.1·2.1절): 거리 위의 에너지 통계와 그 성분, 공유 변의 퍼짐 윤곽.

실행(독립 반복) P_1..P_r, Q_1..Q_s 에 대해 C = 교차 실행 쌍의 평균 거리, B_P·B_Q = 같은 쪽 서로 다른 실행 쌍의 평균 거리.
    Δ² = 2C − B_P − B_Q  (= 교차 에너지 거리 − (바닥_P + 바닥_Q)/2; 실행 안 평균 거리는 지워진다)
에너지 거리는 측도 차의 대칭 쌍선형 형식이라 실행이 독립이면 Δ² 는 중심 분포 사이 에너지 거리의 불편 추정량이고,
이 성질은 바탕 거리의 음의 유형과 무관하다(D 는 음의 유형이 아니다 — 수학 계보 5절). 잃는 것은 ℰ_D ≥ 0 의 보장뿐이고,
실제 표본에서는 J D J 의 양의 고유값 비율 η₊ 로 진단한다(eta_plus).
    w̄ = (B_P + B_Q)/2,  비 = √((C/w̄)² − 1) = √((Δ²/(2w̄) + 1)² − 1)  (교차 거리 ÷ 안쪽 거리에서 정의상 정확히 나오는 신호 대 잡음)
비 = 1 이 가까운 영역과 먼 영역의 경계다(Δ²/√n = 2(√2 − 1) w̄/√n). 주 통계는 Δ²/√n̄ 이다.
    Δμ²₂ = (C₂ − (B₂_P + B₂_Q)/2) / 변 수   (C₂·B₂ 는 거리² 의 평균; 지수 2 에너지 통계, 평균 이동)
맞춘 좌표(유클리드)에서는 Δμ²₂ 가 실행 평균 사이 평균 이동과 정확히 같다(퍼짐 항이 지워짐). D 에서는 그 일반화다.

추정은 불완전 U-통계량이다: 실행 쌍마다 프레임 쌍 k 개(기본 2,000, 비복원)의 줄인 계산 거리(reduced)를 평균한다.
공통 정의역 논문의 같은 서열 칸(E7·round39·P2 의 D 판)은 모두 이 계산이다(code/r38_s0.py 의 runpair_means·estats,
code/r38_e7_30.py). 길이가 다른 쌍의 통계도 같은 식이다(바탕 D 가 어디서나 같음).
"""
import itertools

import numpy as np
import torch

from .reduced import MeanStart, pair_scores


def energy_stats(C, B_P, B_Q, n_edges=None):
    """평균 거리 C (교차), B_P, B_Q (같은 쪽 서로 다른 실행) -> dict(C, B_P, B_Q, D2, wbar, ratio[, D2_sqrt_n]).
    D2 = 2C − B_P − B_Q, wbar = (B_P + B_Q)/2, ratio = √(max((C/w̄)² − 1, 0)) (Δ² < 0 이면 0, w̄ ≤ 0 이면 nan).
    n_edges 를 주면 주 통계 Δ²/√n̄ 를 D2_sqrt_n 으로 더한다."""
    D2 = 2 * C - B_P - B_Q
    w = (B_P + B_Q) / 2
    o = dict(C=float(C), B_P=float(B_P), B_Q=float(B_Q), D2=float(D2), wbar=float(w),
             ratio=float(np.sqrt(max((C / w) ** 2 - 1, 0))) if w > 0 else float('nan'))
    if n_edges:
        o['D2_sqrt_n'] = float(D2 / np.sqrt(n_edges))
    return o


def mean_shift2(C2, B2_P, B2_Q, n_edges):
    """지수 2 에너지 통계 (평균 이동, 변당): Δμ²₂ = (C₂ − (B₂_P + B₂_Q)/2) / 변 수. C₂·B₂ 는 실행 쌍 거리² 의 평균."""
    return float((C2 - (B2_P + B2_Q) / 2) / n_edges)


def component_ratio(C, B_P, B_Q, dmu2, n_edges):
    """성분으로 낸 비 (시연 설계 E0′): √(max(Δμ²₂ · 변 수, 0)/w̄² + ((ρ − 1)/(ρ + 1))²), ρ = B_Q / B_P.
    같은 서열 칸에서 직접 비(energy_stats 의 ratio)와 맞으면 집중 근사와 단일 분지 가정이 그 칸에서 선다."""
    w = (B_P + B_Q) / 2
    rho = B_Q / B_P
    return float(np.sqrt(max(dmu2 * n_edges, 0.0) / w ** 2 + ((rho - 1) / (rho + 1)) ** 2))


def regime_boundary(wbar, n_edges):
    """두 영역의 경계(비 = 1)에서의 Δ²/√n = 2(√2 − 1) w̄/√n."""
    return float(2 * (np.sqrt(2) - 1) * wbar / np.sqrt(n_edges))


def frame_pairs(a, b, n_per_run, k=2000, rng=None):
    """실행 a 와 실행 b (각 n_per_run 프레임, 앙상블 안에서 실행 순서로 이어 붙임)의 프레임 쌍 k 개(비복원) -> 앙상블 안 색인 (ia, ib)."""
    rng = np.random.default_rng() if rng is None else rng
    f = rng.choice(n_per_run * n_per_run, k, replace=False)
    return a * n_per_run + f // n_per_run, b * n_per_run + f % n_per_run


def run_pair_means(ms, run_pairs, n_per_run, k=2000, rng=None, want_gap=False, **pair_kw):
    """실행 쌍 목록 run_pairs [(a, b)] 마다 프레임 쌍 k 개의 줄인 계산 거리를 평균한다(한 번에 푼다). ms = MeanStart(EA, EB).
    반환 dict: d, d2 (실행 쌍마다 평균 거리와 평균 거리²), widened, half_width_max, ia, ib, group, dv (프레임 쌍마다 거리);
    want_gap 이면 gap (평균 빈칸 몫), dpair (평균 짝 몫 거리 √(d² − 빈칸 몫)) 도."""
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
    """두 앙상블 사이 칸 하나. EA ((실행 수 × n_per_run), n, 3, 3) 는 실행마다 n_per_run 프레임을 실행 순서로 이어 붙인 것(EB 도).
    교차 = 모든 실행 쌍 (a, b), 바닥 = 같은 쪽 서로 다른 실행 쌍. 실행 쌍마다 프레임 쌍 k 개, 줄인 계산(reduced).
    n_edges: Δ²/√n̄ 와 Δμ²₂ 의 변 수 (기본: 두 쪽 열 수의 평균; 영 항이 있으면 직접 준다).
    반환: energy_stats 의 dict + dmu2 (Δμ²₂) + n_runs_P, n_runs_Q + runs (실행 쌍 평균들)."""
    rng = np.random.default_rng(seed) if rng is None else rng
    EA = torch.as_tensor(EA)
    EB = torch.as_tensor(EB).to(EA)
    rA, rB = len(EA) // n_per_run, len(EB) // n_per_run
    if rA * n_per_run != len(EA) or rB * n_per_run != len(EB):
        raise ValueError('프레임 수가 n_per_run 의 배수가 아니다')
    if rA < 2 or rB < 2:
        raise ValueError('바닥에는 쪽마다 실행이 둘 이상 필요하다')
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
    """거리 행렬 D (N, N) 의 J D J (J = I − 11ᵀ/N) 고유값에서 양의 질량 비율 η₊ = Σ λ₊ / Σ |λ|.
    D 가 음의 유형이면 J D J 는 음의 반정부호라 η₊ = 0 이다. 실제 표본에서 음의 유형 위반의 크기를 본다(진단)."""
    D = np.asarray(D, np.float64)
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    ev = np.linalg.eigvalsh(J @ D @ J)
    return float(ev[ev > 0].sum() / np.abs(ev).sum())


def project_to_real(m, s=None, degen_frac=0.5):
    """평균 점 m (n, 3, 3) 의 항을 실제 항 s·R/2 꼴로 사영한다. R̄_i 는 극분해의 회전(가장 가까운 회전).
    s 를 주지 않으면 가장 가까운 점의 길이 비 s_i = 2⟨R̄_i, m_i⟩/3 (Frobenius 사영; 결과 문서 2.4 의 "평균 점의 사영"),
    주면 그 길이 비(예: 앙상블 평균 길이 비 2‖E[..., 0]‖ 의 평균; 1차 P3 의 사영 평균 점)를 쓴다.
    반환 (P (n, 3, 3), degenerate (n,) bool). degenerate = 셋째 특이값 < degen_frac × s/2
    (그 변의 방향이 앙상블에서 크게 흩어져 사영이 불안정함; 1차 P3 는 이런 변을 통제·반응 둘 다에서 뺐다)."""
    m = np.asarray(m, np.float64)
    U, S, Vt = np.linalg.svd(m)
    dt = np.sign(np.linalg.det(U @ Vt))
    Dg = np.ones((len(m), 3))
    Dg[:, 2] = dt
    R = np.einsum('iab,ib,ibc->iac', U, Dg, Vt)
    s = 2 * np.einsum('nij,nij->n', R, m) / 3 if s is None else np.asarray(s, np.float64)
    return R * (s / 2)[:, None, None], S[:, 2] < degen_frac * s / 2


def edge_spread_profile(F, idx, iters=3):
    """공유 변 위의 변별 퍼짐 윤곽. F (N, n, 3, 3) 는 한 단백질 앙상블의 좌표(평균 점에 맞춘 것), idx 는 공유 변 번호.
    공유 변만으로 일반화 프로크루스테스(자기 평균 둘레, 회전만)를 iters 번 하고, 변마다 평균 둘레의 제곱 흩어짐 평균을 낸다 -> (len(idx),)."""
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
    """두 단백질 앙상블의 공유 변 동역학 닮음 r_dyn: 짝지은 공유 변 pairs (k, 2) 위 퍼짐 윤곽의 Spearman 순위 상관.
    공유 변은 두 평균 점 사이 최적 정렬(π̄)의 짝으로 정한다(공통 정의역 논문 1b·1c). 순위는 동점을 나누지 않는다."""
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    va = edge_spread_profile(Fa, pairs[:, 0], iters)
    vb = edge_spread_profile(Fb, pairs[:, 1], iters)
    ra, rb = np.argsort(np.argsort(va)), np.argsort(np.argsort(vb))
    return float(np.corrcoef(ra, rb)[0, 1])


def aligned_energy_stats(P, Q, n_edges):
    """맞춘 좌표(유클리드) 판의 칸 하나 (같은 서열, 대응 고정). P, Q: 실행 목록, 실행마다 (프레임 수, 9·변 수) 의
    맞춘 좌표(공통 기준 평균 점에 align_to_reference 로 맞춘 것을 펼친 것). 모든 실행 쌍, 실행 쌍마다 모든 프레임 쌍.
    반환: energy_stats 의 dict + a_P, a_Q (실행 안 평균 거리), dmu (실행 평균 사이 평균 이동/변: 교차 − 바닥 평균),
    logspread (Q 와 P 의 총퍼짐 비의 로그). code/r38_compare.py 의 compare 와 같은 계산이다(round37 맞춘 좌표 판·P2 맞춘 좌표 원래 판)."""
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
