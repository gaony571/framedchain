# framespace

(English: `README.md`. 이 파일은 한국어 판이다.)

단백질 백본의 프레임 열 공간. 이 패키지에는 공간 문서(`SQFORM_SPACE.md`)의 **확정된 정의와 성질**, 그리고 **공통 정의역 논문(2026-10)에서 고정해 실제로 쓴 계산과 추정**(줄인 계산, 에너지 통계, 공유 변 퍼짐 윤곽, 짝·빈칸 분해, 자의 묶음 D_{w,ℓ}·D_TM)을 둔다. 사전 등록한 판정 규칙, 통계 검정(부트스트랩·편 순위 상관 등), 자료 읽기(mdCATH·ATLAS), 아직 고정하지 않은 연구 단계의 것(바닥값 편향 맞춤, 평균·퍼짐 분해, 확률 대응, 바리센터)은 넣지 않는다.

- **좌표** (1.1–1.4절): 변 i (Cα_i → Cα_{i+1}) 마다 프레임 R_i = [T, U, V]. T 는 변의 방향, U 는 C=O 방향에서 T 성분을 뺀 것, V = T × U. 항은 x_i = s_i R_i / 2, s_i = l_i / l0, l0 = 3.83 Å. 첫 열에 2·l0 를 곱하면 변의 벡터라 좌표만으로 Cα 가 복원된다. 변 길이 3.45 Å 미만은 시스.
- **같은 점** (2.1절): 영행렬을 끼우거나 빼도, 모든 항에 같은 회전을 왼쪽에 곱해도 같은 점이다.
- **거리** (2.1–2.2절, 증명): d² = min over (순서를 지키는 부분 대응, 회전) of [짝지은 항의 차의 제곱합 + 짝 없는 항의 노름² 합] = ‖x‖² + ‖y‖² − 2S.
- **계산** (3.1–3.2절): 회전을 고정하면 대응은 동적계획법, 대응을 고정하면 회전은 3×3 SVD 로 정확하다. 둘을 번갈아, 무작위 회전 128개에서 시작한다. 계산값은 상계다.
- **보증** (3.3절): 회전 공간의 분기 한정으로 전역 최적을 보증한다. 실제 구조만 받는다.
- **대응을 아는 경우** (3.5절): 회전만 최적화한 d_fix (SVD 한 번, d 의 상계). 한 분석 안에서는 d 와 d_fix 중 하나만 쓴다.
- **직선** (2.3절, 증명): 최적의 대응과 회전으로 열을 맞추면 두 점을 잇는 직선이 최단 경로다. 짝 없는 항은 영으로 줄거나 영에서 자란다. 중간 점은 실제 구조가 아니다.
- **평균 점** (2.4, 3.5절): 대응이 고정된 묶음(한 단백질의 앙상블)을 평균에 정렬하고 항별로 평균 내기를 수렴할 때까지 되풀이한다(대응을 고정한 거리 d_fix 의 제곱합을 줄이는 교대 최적화). 평균 점은 공간의 점이고 특이값이 줄어든 정도가 흩어짐이다. 시작점과 무관하다는 것은 측정이다(앙상블 6개, 시작점 9개). 길이가 다른 구조의 평균은 넣지 않았다.
- **표** (2.7절): G_ij = x_iᵀ x_j, δ_i = det x_i. 전체 회전에 정확히 불변이고 실제 구조에서는 표에서 프레임 열을 되찾는다.
- **눈금** (2.5절): d / √(0.75(n+m)), sim_long = S / (0.75 max(n, m)), d_fix / √(2n).
- **앙상블 W₂** (2.2절의 거리 위의 최적 수송): 표본 쌍마다 d² 를 지상 비용으로 한 정확한 최적 수송. 지상 거리가 거리이므로 W₂ 도 거리이고, 표본 쌍의 d 가 상계라 낸 값도 상계다. 무게를 주면 주변 분포로 쓴다. 정의대로의 계산만 두었다(쌍마다 출발 128개). 표본 수와 계산을 줄이는 방법은 연구 기획 E0 에서 정한다.
- **PDB 읽기**: N, Cα, C, O 가 모두 있는 잔기만, 여러 MODEL 은 묶음으로, 끝 캡(ACE, NME, NHE)은 뺀다. altLoc 은 처음 것. 사슬이 끊긴 곳은 확인하지 않는다(긴 변으로 나타난다).

공통 정의역 논문에서 고정한 것 (0.3.0):
- **짝 몫·빈칸 몫** (`parts`, 2.2절 따름): 최적 정렬에서 d² = Σ_짝 ‖x_i − U y_j‖² + Σ_짝없음 ‖·‖² 로 정확히 나뉜다. 빈칸 몫에는 삽입·결실만이 아니라 120° 넘게 돈 대응 변도 들어간다. **평균 정규화** D̂ = d / √(‖x‖² + ‖y‖²) ∈ [0, 1] 는 모든 길이에서 거리다.
- **줄인 계산** (`reduced`, 4.2절): 두 앙상블 평균 점을 128 출발로 맞춘 회전·대응(π̄)에서 표본 쌍마다 출발 회전 하나를 만들고(평균 점 출발), 대응을 π̄ 둘레 띠 안으로 제한한 DP 로 푼다. 띠 반폭 5 에서 시작해 가장자리에 닿은 쌍만 두 배로 넓힌다. 띠 DP 는 띠 밖을 막은 전체 DP 와 정확히 같고, 값은 상계다. 가장자리 규칙은 점검이지 보장이 아니다(정확성은 BnB 와 견준 상계 편향으로 잰다).
- **에너지 통계** (`ensemble`, 2.7절): 실행(독립 반복) 사이 평균 거리 C (교차), B_P·B_Q (같은 쪽 서로 다른 실행)에서 Δ² = 2C − B_P − B_Q, w̄ = (B_P + B_Q)/2, 비 = √((C/w̄)² − 1), 주 통계 Δ²/√n̄, 지수 2 판 Δμ²₂ = (C₂ − (B₂_P + B₂_Q)/2)/변 수. 쌍선형 항등식으로 Δ² 는 불편이고 바탕 거리의 음의 유형과 무관하다. 추정은 실행 쌍마다 프레임 쌍 2,000 개의 불완전 U-통계량이다. 음의 유형 진단 η₊ (J D J 의 양의 고유값 비율), 성분 비, 두 영역의 경계.
- **공유 변 퍼짐 윤곽** (`ensemble`): 공유 변만으로 일반화 프로크루스테스를 한 뒤의 변별 퍼짐과, 두 앙상블의 윤곽 순위 상관(동역학 닮음 r_dyn). 평균 점을 실제 항 s·R/2 꼴로 사영하기.
- **거의 최적 대응과 그 교집합** (`parts.near_optimal_alignments`, 0.3.1): 128 출발 가운데 S ≥ (1 − ε) S_max 인 출발들의 대응, 서로 다른 대응 수, 교집합(1차 P3·C 의 공동 관문). **맞춘 좌표 판 통계** (`ensemble.aligned_energy_stats`, 0.3.1): 대응을 고정한 유클리드 판의 C·B·Δ²·Δμ²·로그 퍼짐 비(round37·P2 맞춘 좌표 원래 판).
- **자의 묶음** (`rulers`, 5절, numba 필요): 위치 항을 넣은 D_{w,ℓ} 와 고정 d₀ 의 TM 꼴 거리 D_TM (TM-align 의 목적 함수 = d₀ 고정 D_TM), 정의대로의 비용, TM-score 의 네 꼴, 삼각 부등식 위반률. 계산값은 상계이고 보증은 없다.

넣지 않은 것: 중간 점을 구조로 되돌리기(문턱을 정하지 않았다, 2.4절).

## 바꿀 수 있는 값

| 어디 | 값 (기본) |
|---|---|
| 좌표 | `l0` (3.83 Å, 길이 비의 기준), `cis_max` (3.45 Å), `caps` (끝 캡 잔기 이름), `is_real(tol=1e-3)`, `n_edges(eps=1e-12)`, `from_table(eps=1e-12)` |
| 거리 (`align`, `sq_distance`, `distance` 에 키워드로) | `n_random` (127, 항등과 합쳐 출발 128개), `starts` (출발 회전을 직접), `iters` (10), `seed` (0), `improve_tol` (1e-7) |
| 보증 (`bnb`) | `rel_tol` (0.005, sim_long 단위) 또는 `tol` (S 절대값), `n_starts` (128), `level0` (3), `n_refine` (16), `max_cells` (2,000,000), `batch`, `iters` (10), `device` |
| 앙상블 W₂ (`ensemble_w2`, `structure_costs`) | `chunk` (32, 한 번에 푸는 쌍 수. 메모리는 chunk × 출발 수 × n × m 에 비례), `device`, `dtype`, 거리의 키워드(`n_random`, `iters`, `seed`, `improve_tol`), 무게 `wa`, `wb`, 이미 낸 지상 비용 `cost` |
| 최적 수송 (`optimal_plan`) | 무게 `a`, `b`. 무게가 없고 크기가 같으면 할당 문제, 그 밖에는 POT (있으면) 또는 선형계획 |
| 줄인 계산 (`MeanStart`, `pair_scores`) | `n_mean_starts` (128, 평균 점끼리의 출발 수), `k0` (5, 처음 띠 반폭), `iters` (10), `want_gap` (빈칸 몫도), `membudget` (3e9 바이트, 한 번에 푸는 쌍 수의 어림) |
| 에너지 통계 (`ensemble_statistic`, `run_pair_means`) | `n_per_run` (실행마다 프레임 수, 논문은 100), `k` (2,000, 실행 쌍마다 프레임 쌍 수), `seed` 또는 `rng`, `n_edges` (Δ²/√n̄ 의 변 수; 기본은 두 쪽 열 수의 평균), 줄인 계산의 키워드 |
| 퍼짐 윤곽 (`edge_spread_profile`, `dynamics_similarity`), 사영 (`project_to_real`) | `iters` (3, 공유 변 GPA 횟수), `s` (사영의 길이 비; 기본은 가장 가까운 점), `degen_frac` (0.5, 퇴화 변 표시) |
| 자의 묶음 (`rulers.tm_align`, `rulers.wl_align`) | `d0` 또는 `ell`·`w`, `poses`·`extra_poses` (출발 자세), `iters` (50), `keep` (16, 다듬을 출발 수), `irls` (3), `search` (부분 조각 탐색), `tol` |
| 평균 점 (`mean_point`) | `tol` (1e-10, 평균이 한 번에 움직인 거리의 상대값), `max_iters` (100), `start` (출발 구조의 번호 또는 점), `return_info` (수렴 여부·횟수) |

`l0` 를 바꾸면 좌표를 만들 때와 Cα·변 길이·시스를 읽을 때 같은 값을 줘야 한다. 프레임과 길이 비를 읽는 `frames_of`, 실제 구조 판별 `is_real` 은 `l0` 와 무관하다.

## 설치

```
pip install -e .          # numpy, torch, scipy
pip install -e .[ot]      # 무게 붙은 수송을 POT 로 (없으면 scipy 선형계획)
pip install -e .[test]    # pytest
pip install -e .[rulers]  # 자의 묶음 D_{w,ℓ}·D_TM (numba)
```

PyPI 에는 없다. 저장소에서 바로 설치하려면 `pip install "framespace @ git+https://github.com/gaony571/framespace@v0.3.3"`.
Python 3.12 (numpy 2.5, torch 2.14, scipy 1.18, numba 0.68) 에서 시험했다. 빠른 시작은 `python examples/quickstart.py`
(입력 파일 없음, 기대 출력 `examples/quickstart_output.txt`), 함수 목록은 `docs/API.md`, 바뀐 것은 `CHANGELOG.md`.
계산 시간·메모리(예: 노트북 CPU 에서 200 × 210 변 쌍 128 출발 22 초, 32 출발 6.4 초)와 재현성 안내는 영문 README 의
Performance and memory, Reproducibility 절에 있다.

## 쓰기

```python
import framespace as fs

x = fs.coordinates_from_pdb('a.pdb')           # PDB (MODEL F 개) -> (F, L−1, 3, 3)
y = fs.coordinates_from_backbone(CA, C, O)     # 잔기마다 Cα, C, O (F, L, 3) -> (F, L−1, 3, 3)
d, al = fs.distance(x[0], y[0])                 # 거리와 정렬 (점수 S, 회전 U, 짝)
cert = fs.bnb(x[0], y[0])                       # 전역 최적 보증
a, b, cols = fs.aligned_columns(x[0], y[0], alignment=al)
mid = fs.geodesic(a, b, 0.5)                    # 직선 위 중간 점
M = fs.mean_point(x)                            # 한 단백질 앙상블의 평균 점
G = fs.relative_table(x[0])                     # 불변 표현
CA0 = fs.ca_trace(x[0])                         # 좌표에서 Cα
w = fs.ensemble_w2(x, y)                        # 앙상블 사이 W₂ (w['W2'], 수송 계획 w['plan'], 지상 비용 w['cost'])
sp = fs.split_sq_distance(x[0], y[0])            # 짝 몫 sp['matched'], 빈칸 몫 sp['gap'] (합 = d²)

# 공통 정의역 논문의 앙상블 칸 하나: 실행마다 100 프레임을 이어 붙인 EA, EB (같은 단백질이면 같은 변 수)
st = fs.ensemble_statistic(EA, EB, n_per_run=100)      # st['D2'], st['D2_sqrt_n'], st['ratio'], st['wbar'], st['dmu2']
ms = fs.MeanStart(EA, EB)
r = fs.pair_scores(ms, ia, ib, want_gap=True)           # 표본 쌍 목록의 줄인 계산 거리(상계)와 빈칸 몫
r_dyn = fs.dynamics_similarity(FA, FB, shared_pairs)    # 공유 변 퍼짐 윤곽의 순위 상관

from framespace import rulers                           # numba 필요
tm = rulers.tm_align(CA_a, CA_b, d0=rulers.d0_tm(max(len(CA_a), len(CA_b))))   # tm['S'], tm['D2'] = (n + m)/2 − S
```

## 모듈

| 모듈 | 내용 |
|---|---|
| `geometry` | 회전, 사원수, Procrustes, 극분해 |
| `frames` | PDB 읽기, 원자에서 프레임과 좌표, 변의 길이, 시스 판별, 실제 구조 판별, Cα 복원 |
| `distance` | 대응 DP, 대응과 회전의 교대 최적화(묶음, GPU), 거리, d_fix, 눈금, 영 채움 |
| `certify` | 분기 한정 보증 |
| `operations` | 열 맞추기, 직선, 평균 점 |
| `table` | 상대 프레임의 표, 행렬식, 표에서 되찾기 |
| `transport` | 정확한 최적 수송, 표본 쌍의 지상 비용, 앙상블 W₂ |
| `parts` | 짝 몫·빈칸 몫 분해, 평균 정규화 거리 D̂ |
| `reduced` | 띠 DP, 띠 정렬, 평균 점 출발(`MeanStart`), 표본 쌍 목록의 줄인 계산(`pair_scores`) |
| `ensemble` | 에너지 통계(Δ², w̄, 비, Δμ²₂, 성분 비, 영역 경계), 불완전 U-통계량, 앙상블 칸(`ensemble_statistic`), η₊, 평균 점 사영, 공유 변 퍼짐 윤곽, r_dyn |
| `rulers` | D_{w,ℓ}, D_TM, d₀(L), TM-score 의 네 꼴, 삼각 부등식 위반률 (numba; `__init__` 에서 불러오지 않음) |

## 시험

```
pytest tests
```

합성 구조(트랜스와 시스 길이 포함)로: 사원수, Procrustes·극분해, 원자 ↔ 좌표 ↔ Cα 왕복과 길이·시스, PDB 왕복, l0·시스 경계 바꾸기, 자기 거리 0과 대칭, 회전 불변, 영행렬 끼우기 불변, 잘라내기 법칙, 변 줄이기 법칙, 삼각부등식, 직선이 최단 경로, d_fix 가 상계, 눈금, 평균 점(회전한 복사본, 흩어진 앙상블의 수렴·시작점 무관·고정점), 표의 불변과 되찾기, 보증, 최적 수송(할당과 선형계획이 같은 값, 무게의 주변 분포), 앙상블 W₂ (자기 0, 대칭, 표본 회전과 영행렬 끼우기에 불변, 삼각부등식).

`tests/test_paper_methods.py`: 띠 DP = 띠 밖을 막은 전체 DP, 띠 정렬 = 전체 정렬(띠가 넓을 때), 줄인 계산이 전체 계산과 같고 상계이며 끼운 변이 빈칸 몫에 들어감, 띠 넓히기, Δ² = 교차 에너지 거리 − 바닥 평균(임의의 거리), 비의 항등식, 유클리드에서 Δμ²₂ = 실행 평균 이동, 같은 분포 대 옮긴 분포의 Δ², η₊ (유클리드 0, 4 제곱 > 0), 퍼짐 윤곽의 회전 불변과 자기 닮음 1, 사영(실제 항은 그대로, 항마다 가장 가까움), 짝 몫 + 빈칸 몫 = d², D̂ 의 범위·삼각부등식. `tests/test_rulers.py` (numba 가 없으면 건너뜀): 정의와 계산의 일치, 대응·자세 합성의 삼각 부등식, ℓ → ∞ 에서 기본판, 최적화 값의 삼각 부등식, 선형 꼴(Sippl)은 거리가 아님, 영 항 끼우기 불변.

## 논문 실행본과의 관계

공통 정의역 논문의 분석은 framespace 0.2.0 (태그 `v0.2.0`, 커밋 866995c; 사전 등록의 묶음 md5 4edba596), 연구 단계 코드 `framespace_wip/stage0` (묶음 md5 6703fa26), 분석 스크립트(`code/r38_*.py` 등, 스크립트마다 md5 등록)로 돌았다. 이것들은 돌린 그대로 동반 저장소 framespace-paper (https://github.com/gaony571/framespace-paper, 논문과 함께 공개 예정)에 결과별 스크립트·출력 대응표와 함께 있다. 0.3.0 은 그중 고정된 계산을 옮겼고, 0.2.0 모듈의 코드는 바꾸지 않았다(0.2.0 은 태그 `v0.2.0`; 0.3.2 는 docstring·주석만 영어로 옮겼고, 구문 트리 대조로 코드가 같음을 확인했다).

| 0.3.0 | 실행본 |
|---|---|
| `reduced.center_path` … `MeanStart` | `stage0/band.py` (함수 본문 같음, 패키지 안 불러오기만 바꿈) |
| `reduced.pair_scores` | `code/r38_s0.py` 의 `red_pairs` (같은 계산, 반환을 dict 로) |
| `ensemble.energy_stats`, `frame_pairs`, `run_pair_means` | `code/r38_s0.py` 의 `estats`, `upairs`, `runpair_means` (같은 식; 평균 거리² 를 더함) |
| `ensemble.ensemble_statistic`, `mean_shift2` | `code/r38_e7_30.py` 의 D 판 칸 (교차·바닥·Δμ²₂_D) |
| `ensemble.component_ratio`, `regime_boundary` | `code/r39_core.py` 의 `e0_dom` |
| `ensemble.eta_plus` | `code/r38_b6.py` |
| `ensemble.edge_spread_profile`, `dynamics_similarity`, `project_to_real(s=…)` | `code/r38_p3.py` 의 `gpa_profile`, `spearman`, 사영 평균 점 (기본 길이 비는 `code/r39_roundtrip.py` 의 가장 가까운 점) |
| `parts.split_sq_distance` | `red_pairs` 의 빈칸 몫과 같은 정의 (전체 정렬에서) |
| `parts.near_optimal_alignments` | `code/r38_s0.py` 의 `align_all`, `code/r38_p3.py` 의 π̄ 단계 (ε 0.005, 교집합) |
| `ensemble.aligned_energy_stats` | `code/r38_compare.py` 의 `compare` |
| `rulers` | `stage0/rulers.py` (불러오기만 바꿈) |

실행 쌍의 프레임 쌍을 뽑는 난수 순서는 스크립트마다 다르므로(도메인 이름 seed 등), 같은 값을 다시 내려면 그 스크립트를 쓴다.

## 라이선스와 인용

MIT 라이선스, copyright (c) 2026 Hyeonsu An (`LICENSE`). 논문과 이 소프트웨어(`CITATION.cff`)를 인용해 주십시오.
