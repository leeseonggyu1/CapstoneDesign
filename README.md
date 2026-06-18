# CapstoneDesign

CenterNet 기반 감마/샤프닝 보정 실험 정리 저장소입니다.

## 실험 목적

야간 CCTV 이미지에서 밝은 영역과 어두운 영역이 같이 존재할 때, 감마 보정과 샤프닝 보정을 이용해 객체 검출 성능을 높이는지 확인했습니다.

## 핵심 파이프라인

1. LLVIP 데이터로 illumination 모델을 사전 학습합니다.
2. 학습된 illumination 모델을 사용해 감마/샤프닝 보정 값을 추정합니다.
3. 보정 필터와 CenterNet을 결합해 detection loss 기반으로 학습합니다.
4. 보정된 이미지 또는 end-to-end wrapper를 사용해 Yeoju 데이터에서 CenterNet을 학습/평가합니다.

## 주요 결과

| Experiment | AP | AP50 | AP75 | AR100 |
|---|---:|---:|---:|---:|
| Yeoju raw baseline | 0.678 | 0.895 | 0.748 | 0.765 |
| Yeoju E2E paper filter | 0.714 | 0.927 | 0.809 | 0.783 |
| Yeoju filtered-image CenterNet | 0.719 | 0.925 | 0.813 | 0.790 |
| LLVIP filter -> Yeoju CenterNet | 0.724 | 0.921 | 0.821 | 0.793 |

현재 실험 기준 최고 성능은 `LLVIP filter -> Yeoju CenterNet`이며, Yeoju raw baseline 대비 AP는 `+0.046`, AP50은 `+0.026`, AP75는 `+0.073`, AR100은 `+0.028` 향상되었습니다.

## 폴더 구조

```text
scripts/
  실험 재현 및 평가용 PowerShell/Python 스크립트

results/
  실험 결과 CSV, 감마/샤프닝 변화 분석, 예시 이미지, 그래프

docs/
  작업 맥락 및 실험 정리 문서

presentations/
  발표자료 초안 및 결과 정리 PPT
```

## 재현 스크립트

주요 실행 파일은 다음과 같습니다.

| File | Purpose |
|---|---|
| `scripts/run_pretrain_illumination_llvip.ps1` | LLVIP illumination 모델 사전 학습 |
| `scripts/run_yeoju_llvipillum_pipeline.ps1` | LLVIP illumination 기반 Yeoju end-to-end/filtered pipeline |
| `scripts/run_llvip_filter_to_yeoju_centernet.ps1` | LLVIP에서 필터 학습 후 Yeoju CenterNet 학습/평가 |
| `scripts/generate_paper_filtered_images.py` | 학습된 필터로 보정 이미지 생성 |
| `scripts/compute_gamma_sharpness_values.py` | 이미지별 감마/샤프닝 값 분석 |
| `scripts/make_sharpness_and_detection_examples.py` | PPT용 예시 이미지 및 그래프 생성 |

## 제외한 파일

다음 파일은 용량이 크거나 로컬 경로 의존성이 커서 업로드 패키지에서 제외했습니다.

- 원본 데이터셋: `llvip`, `Yeoju_rain`
- 전체 필터링 이미지 폴더
- 학습 체크포인트: `*.pth`
- 임시 로그 및 캐시 파일

필요한 경우 위 파일들은 로컬 `D:\KDG` 경로에서 별도로 보관해야 합니다.
