# framespace

단백질 백본의 프레임 열 공간. 길이가 다른 모든 백본을 하나의 좌표계와 하나의 증명된 거리 위에 놓는다. 이 패키지에는 **확정된 정의와 성질만** 둔다(연구 단계의 분석은 넣지 않는다).

- 좌표: 변(CA_i → CA_{i+1})마다 프레임 R_i = [T, U, V] 의 절반 R_i / 2. T 는 변 방향, U 는 (O_i − C_i) 의 T 에 수직인 성분 방향. 짝 없는 변은 영행렬(정보가 없는 변). 전역 회전으로 나눈다. 변 길이는 좌표에 넣지 않는다.
- 거리(잔기 단위): d²(A, B) = |A|² + |B|² − 2 max_U max_짝짓기 Σ ⟨A_k, U B_l⟩. 순서를 지키는 1대1 짝짓기와 회전을 함께 최적화한다. 실제 구조끼리는 |A|² + |B|² = 0.75 (n + m). 거리임의 증명은 `SQFORM_SPACE.md`.
- 대응을 알 때(같은 단백질)는 짝짓기 없이 회전만 최적화한 거리.
- 인증: 회전 공간의 분기 한정으로 전역 최적을 인증한다.
- 층 사이 직선: 열을 맞춘 두 점을 잇는 직선은 최단 경로다(비교용, 중간 점은 실제 구조가 아니다).
- 평균 점: 같은 단백질 앙상블을 전역 회전으로 맞춘 열마다의 산술 평균.

## 설치

```
pip install -e .          # numpy, torch
pip install -e .[test]    # pytest
```

## 쓰기

```python
import framespace as fs

XA, lA = fs.from_pdb('a.pdb')                        # (F, n, 3, 3) = R/2, 변 길이 (F, n)
XB, lB = fs.from_pdb('b.pdb')
d, al = fs.distance(XA[0], XB[0], normalized=True)   # 정규화 거리 d² / (|A|² + |B|²) 의 제곱근, 정렬(점수, 회전, 짝)
cert = fs.bnb(XA[0], XB[0])                          # 전역 최적 인증
Ca, Cb, cols = fs.aligned_columns(XA[0], XB[0], alignment=al)
mid = fs.geodesic(Ca, Cb, 0.5)                       # 직선 위 중간 점
M = fs.mean_point(XA)                                # 같은 단백질 앙상블의 평균 점
```

## 모듈

| 모듈 | 내용 |
|---|---|
| `geometry` | 회전, 사원수, Procrustes, 극분해 |
| `frames` | 원자(N·CA·C·O)에서 프레임과 좌표, CA 복원, 실제 구조 판별, PDB 백본 읽기 |
| `distance` | 정렬 DP, 정렬과 회전의 교대 최적화(묶음, GPU), 거리, 대응을 알 때의 거리, 영 채움 |
| `certify` | 분기 한정으로 전역 최적 인증 |
| `operations` | 열 맞추기, 층 사이 직선, 평균 점 |

## 시험

```
pytest tests
```

합성 프레임 열로: 사원수, Procrustes·극분해, 프레임과 PDB 왕복, 자기 거리 0과 대칭, 회전 불변, 영 채움 불변, 잘라내기 법칙(d² = 0.75 k), 삼각 부등식, 직선이 측지선, 대응을 알 때 ≥ 최적, 평균 점, 인증.
