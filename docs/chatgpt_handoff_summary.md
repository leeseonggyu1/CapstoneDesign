# ChatGPT 인수인계 요약

## 연구 목표

CenterNet 기반 저조도 객체 검출 연구를 진행 중이다. 목표는 기존 논문 구조를 재현하고, 감마 보정 및 샤프닝 필터를 활용해 야간/저조도 객체 검출 성능을 높이는 것이다.

대상 논문은 한국전기전자학회 최종본이며, 논문 표에는 다음 성능이 제시되어 있다.

| Dataset | Model | Paper AP | Speed |
|---|---:|---:|---:|
| LLVIP | CenterNet | 0.880 | 0.092 sec |
| LLVIP | Proposed CenterNet | 0.884 | 0.095 sec |
| CCTV Road Traffic / Yeoju | CenterNet | 0.907 | 0.092 sec |
| CCTV Road Traffic / Yeoju | Proposed CenterNet | 0.917 | 0.095 sec |

논문에는 AP가 어떤 IoU 기준인지 명확히 적혀 있지 않다. 값의 크기상 COCO AP@[0.50:0.95]보다는 AP50 계열일 가능성이 높다고 보고 있다.

## 현재까지 확인한 핵심 결론

외장하드 `E:\KDG`에 있는 전임자 원본 파일들을 읽기 전용으로 스캔했다. 파일은 수정하지 않았다.

현재까지의 결론은 다음과 같다.

1. 원본 코드에는 `GammaFilter`, `UsmFilter`, `IlluminationClassifier` 정의가 존재한다.
2. 하지만 실제 CenterNet `forward()` 안에서는 감마/샤프닝 필터가 호출되지 않는다.
3. 감마/샤프닝은 CenterNet 모델 내부가 아니라 데이터 로딩 단계에서 이미지에 먼저 적용된다.
4. 적용 후 `detach().cpu().numpy()`로 다시 이미지화되기 때문에 검출 loss가 감마/샤프닝 추정 모듈로 역전파되지 않는 것으로 보인다.
5. 따라서 남아있는 코드 기준으로는 “검출 loss 기반 end-to-end 감마 학습”이라기보다, “이미지별 감마값/조도값 추정 또는 저장 후 필터링 이미지로 CenterNet 학습/평가”에 가깝다.

## 중요 파일 경로

외장하드 원본:

```text
E:\KDG
```

감마값 저장 코드:

```text
E:\KDG\CenterNet_Origin - 감마값 저장
```

주 모델 코드로 추정:

```text
E:\KDG\CenterNet_Origin - 복사본
```

주요 코드 파일:

```text
E:\KDG\CenterNet_Origin - 감마값 저장\src\lib\models\networks\large_hourglass.py
E:\KDG\CenterNet_Origin - 감마값 저장\src\lib\datasets\sample\ctdet.py
E:\KDG\CenterNet_Origin - 감마값 저장\src\lib\trains\ctdet.py
E:\KDG\CenterNet_Origin - 감마값 저장\src\main.py
E:\KDG\illumination model.py
E:\KDG\Save_gamma.py
E:\KDG\LAP감마필터적용(GAP 감마값 변환).py
E:\KDG\CNN-PP로 이미지 생성코드_2.py
```

## 감마값 생성 구조 해석

`ctdet.py`에서 이미지 로딩 시 다음 흐름이 나타난다.

```text
원본 이미지 로드
→ IlluminationClassifier()로 주간/야간 확률 B 계산
→ cnn_dip() 호출
→ GammaFilter + UsmFilter 적용
→ 감마값 저장
→ 필터링된 이미지를 CenterNet 입력으로 사용
```

`large_hourglass.py`의 감마 계산식은 대략 다음과 같다.

```python
param = exp(tanh_range(-log(3), log(3))(features))
b_param = B[0] * 1.0 + B[1] * param
```

해석:

```text
최종 감마 = Day 확률 * 1.0 + Night 확률 * 이미지 기반 예측 감마
```

즉 주간이면 감마를 거의 1.0으로 두고, 야간이면 이미지 기반 감마값을 더 많이 쓰는 구조로 보인다.

다만 `IlluminationClassifier()`와 감마 추정 CNN이 데이터 로딩 중 새로 생성되는 형태이고, CenterNet optimizer에 등록되어 있지 않은 것으로 보여서, 검출 loss로 같이 학습된다고 보기 어렵다.

## 감마값 저장 흔적

다음 파일에 감마값이 저장되어 있었다.

```text
E:\KDG\CenterNet_Origin - 감마값 저장\gamma_values_epoch_None.txt
```

통계:

| File | Count | Gamma Min | Gamma Max | Gamma Avg |
|---|---:|---:|---:|---:|
| gamma_values_epoch_None.txt | 11498 | 0.6463 | 1.6889 | 1.0109 |

조도 분류와 병합된 파일:

```text
E:\KDG\llvip_gamma_brightness\fixed_merged_predictions.txt
```

통계:

| File | Count | Gamma Min | Gamma Max | Gamma Avg | Labels |
|---|---:|---:|---:|---:|---|
| fixed_merged_predictions.txt | 15488 | 0.6941 | 1.6027 | 1.0089 | Day 3384 / Night 12104 |

## Loss 구조

`ctdet.py`의 loss는 일반 CenterNet 검출 loss다.

```text
loss = hm_loss + 0.1 * wh_loss + off_loss
```

확인된 loss 항목:

```text
hm_loss
wh_loss
off_loss
```

감마 정답 loss, 샤프닝 loss, 감마 regularization loss는 전임자 원본 코드에서는 명확히 보이지 않았다.

## 실험 결과 요약

### 논문 strict 재현 시도

`D:\KDG\paper_strict_runs\paper_strict_summary.csv`

| Dataset | Model | AP | AP50 | AP75 | AR100 |
|---|---|---:|---:|---:|---:|
| LLVIP | CenterNet baseline | 0.506 | 0.916 | 0.501 | 0.573 |
| LLVIP | Proposed | 0.515 | 0.917 | 0.524 | 0.584 |
| Yeoju/CCTV | CenterNet baseline | 0.720 | 0.937 | 0.808 | 0.782 |
| Yeoju/CCTV | Proposed | 0.699 | 0.932 | 0.801 | 0.759 |

해석:

```text
LLVIP는 proposed가 소폭 개선.
Yeoju/CCTV는 AP50 기준으로 baseline보다 낮음.
논문 표처럼 CCTV/Yeoju에서 Proposed가 AP50 +0.010 개선되는 흐름은 재현되지 않음.
```

### 원본 LR에 가까운 active filter 추가 실험

조건:

```text
BaseLr = 0.000125
FilterLrMult = 1
GammaRange = 3
SharpnessMax = 5
BatchSize = 8
NumWorkers = 8
Epoch checkpoint = 5, 10, 15, 20, 25, 30
```

결과 파일:

```text
D:\KDG\paper_strict_runs\paper_active_filter_both_proposed10_extra30_base0p000125_flr1_g3_s5_reset_summary.csv
```

LLVIP:

| Epoch | AP | AP50 | AP75 | AR100 |
|---:|---:|---:|---:|---:|
| 5 | 0.515 | 0.917 | 0.523 | 0.588 |
| 10 | 0.515 | 0.916 | 0.524 | 0.591 |
| 15 | 0.514 | 0.914 | 0.517 | 0.593 |
| 20 | 0.516 | 0.917 | 0.519 | 0.596 |
| 25 | 0.515 | 0.916 | 0.523 | 0.597 |
| 30 | 0.513 | 0.916 | 0.516 | 0.597 |

Yeoju/CCTV:

| Epoch | AP | AP50 | AP75 | AR100 |
|---:|---:|---:|---:|---:|
| 5 | 0.727 | 0.935 | 0.818 | 0.793 |
| 10 | 0.715 | 0.922 | 0.798 | 0.790 |
| 15 | 0.709 | 0.918 | 0.801 | 0.783 |
| 20 | 0.696 | 0.900 | 0.778 | 0.777 |
| 25 | 0.725 | 0.928 | 0.822 | 0.794 |
| 30 | 0.725 | 0.931 | 0.821 | 0.790 |

해석:

```text
원본 LR에 가까운 설정은 공격적인 filter LR보다 안정적이었다.
Yeoju/CCTV는 AP, AP75, AR100 기준으로 baseline보다 일부 개선되었지만 AP50은 baseline 0.937보다 낮은 0.935가 최고였다.
논문 AP가 AP50이라면 논문 흐름은 아직 완전 재현되지 않았다.
논문 AP가 COCO AP라면 Yeoju AP 0.720 → 0.727 개선으로는 어느 정도 개선 흐름이 있다.
```

## 논문과 결과 차이가 나는 가능 원인

1. 논문 AP 기준이 명확하지 않음.
2. 논문 작성 당시 train/val split 또는 annotation 사용 방식이 현재와 다를 수 있음.
3. 원본 코드의 실제 파이프라인이 논문 설명과 다르게 사전 필터링 기반이었을 수 있음.
4. 전임자가 사용한 최종 `merged_predictions`, GAP/LAP 필터 이미지, CenterNet 학습 코드 조합이 정확히 무엇인지 아직 완전히 확정되지 않음.
5. 현재 baseline AP50이 논문 baseline보다 높아서 proposed가 개선될 여지가 줄어든 상태일 수 있음.

## 지금 가장 중요한 질문

1. 전임자 코드에서 감마값은 실제로 학습된 것인가, 아니면 이미지별로 추정/저장된 것인가?
2. 논문에서 말한 end-to-end는 실제 코드상 어느 부분을 의미하는가?
3. 논문 재현은 다음 중 어느 방식으로 해야 하는가?
   - 원본 이미지 + 데이터로딩 단계 필터 적용 + CenterNet 학습
   - GAP/LAP로 미리 필터링한 이미지 데이터셋 + CenterNet 학습
   - CenterNet 내부에 감마/샤프닝 모듈을 붙여 검출 loss로 진짜 end-to-end 학습
4. 논문 수치와 맞추려면 AP50, COCO AP, split, annotation, checkpoint, epoch, pretrain model 중 무엇을 우선 검증해야 하는가?

## ChatGPT에게 요청할 일

아래 방향으로 같이 분석해달라고 요청하면 좋다.

```text
위 내용을 바탕으로 전임자 코드가 실제로 end-to-end 감마/샤프닝 학습인지 판단해줘.
특히 ctdet.py, large_hourglass.py, main.py 구조를 기준으로 gradient가 감마/샤프닝 모듈까지 흐르는지 설명해줘.
그리고 논문 재현을 위해 어떤 실험 순서로 정리해야 할지 제안해줘.
```

필요하면 업로드할 파일:

```text
large_hourglass.py
ctdet.py
main.py
ctdet.py train loss 파일
논문 PDF
paper_strict_summary.csv
paper_active_filter_both_proposed10_extra30_base0p000125_flr1_g3_s5_reset_summary.csv
```

