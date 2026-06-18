# Sharpness and Detection Example Summary

## Sharpness Movement

| Run | Epoch range | Sharpness mean range | Last sharpness | Best AP50 epoch | Best AP50 |
|---|---:|---:|---:|---:|---:|
| Baseline | 1-10 | 0.0000-0.0000 | 0.0000 | 10 | 0.937 |
| Paper-condition Proposed | 1-10 | 0.0109-0.2534 | 0.0248 | 10 | 0.932 |
| Proposed finetune 10-50 | 11-50 | 0.0134-0.0345 | 0.0209 | 15 | 0.932 |
| Active filter original LR | 1-30 | 0.0163-0.1246 | 0.0249 | 5 | 0.935 |
| Active filter aggressive | 1-30 | 0.0246-3.0126 | 2.9924 | 5 | 0.930 |

Predecessor GAP uses fixed pre-filtering, so there is no checkpoint-wise learned sharpness trace. The script settings are sigma=1.5, USM detail weight=1.5, blend=0.8 sharpened + 0.2 blurred.

## Generated Images

- D:\KDG\outputs\sharpness_detection_examples\yeoju_centernet_compare_1_001836_03.png
- D:\KDG\outputs\sharpness_detection_examples\yeoju_centernet_compare_2_001895_03.png
- D:\KDG\outputs\sharpness_detection_examples\yeoju_centernet_compare_3_001845_03.png
- D:\KDG\outputs\sharpness_detection_examples\yeoju_gap_repro_detection_002028_04.png