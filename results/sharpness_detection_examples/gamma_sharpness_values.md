# Gamma and Sharpness Values

## Gamma Movement

| Run | Epoch range | Gamma mean min-max | Last gamma mean | Last gamma min-max | Last sharpness mean |
|---|---:|---:|---:|---:|---:|
| Baseline | 1-10 | 1.0000-1.0000 | 1.0000 | 1.0000-1.0000 | 0.0000 |
| Paper-condition Proposed | 1-10 | 0.9822-1.0585 | 0.9822 | 0.9716-0.9928 | 0.0248 |
| Proposed finetune 10-50 | 11-50 | 0.9775-0.9984 | 0.9901 | 0.9646-1.0156 | 0.0209 |
| Active filter original LR | 1-30 | 0.9692-0.9898 | 0.9767 | 0.9313-1.0059 | 0.0249 |
| Active filter aggressive | 1-30 | 0.9719-2.5063 | 2.4962 | 1.0210-4.7117 | 2.9924 |

## Example Image Values

| File | Model | Gamma | Gamma night | Sharpness | Night prob |
|---|---|---:|---:|---:|---:|
| 001836_03.jpg | Baseline | 1.0000 |  | 0.0000 |  |
| 001836_03.jpg | Paper-condition Proposed 10ep | 0.9974 | 0.9465 | 0.0084 | 0.0486 |
| 001836_03.jpg | Active filter original LR 5ep | 0.9970 | 0.9392 | 0.0052 | 0.0486 |
| 001836_03.jpg | Active filter aggressive 5ep | 0.9958 | 0.9127 | 0.0045 | 0.0486 |
| 001895_03.jpg | Baseline | 1.0000 |  | 0.0000 |  |
| 001895_03.jpg | Paper-condition Proposed 10ep | 0.9980 | 0.9469 | 0.0070 | 0.0372 |
| 001895_03.jpg | Active filter original LR 5ep | 0.9978 | 0.9401 | 0.0043 | 0.0372 |
| 001895_03.jpg | Active filter aggressive 5ep | 0.9968 | 0.9127 | 0.0035 | 0.0372 |
| 001845_03.jpg | Baseline | 1.0000 |  | 0.0000 |  |
| 001845_03.jpg | Paper-condition Proposed 10ep | 0.9967 | 0.9466 | 0.0110 | 0.0625 |
| 001845_03.jpg | Active filter original LR 5ep | 0.9962 | 0.9393 | 0.0068 | 0.0625 |
| 001845_03.jpg | Active filter aggressive 5ep | 0.9945 | 0.9127 | 0.0058 | 0.0625 |
| 002028_04.jpg | Baseline | 1.0000 |  | 0.0000 |  |
| 002028_04.jpg | Paper-condition Proposed 10ep | 0.9511 | 0.9468 | 0.1700 | 0.9199 |
| 002028_04.jpg | Active filter original LR 5ep | 0.9448 | 0.9400 | 0.1064 | 0.9199 |
| 002028_04.jpg | Active filter aggressive 5ep | 0.9197 | 0.9127 | 0.0853 | 0.9199 |

## GAP Pre-filter Values

| File | GAP gamma | LAP gamma mean | LAP gamma min-max | Fixed sharpening |
|---|---:|---:|---:|---|
| 001836_03.jpg | 0.9758 | 0.9716 | 0.9710-0.9806 | sigma=1.5, USM=1.5, blend=0.8 |
| 001895_03.jpg | 0.9097 | 0.8938 | 0.8916-0.9278 | sigma=1.5, USM=1.5, blend=0.8 |
| 001845_03.jpg | 0.9338 | 0.9223 | 0.9206-0.9470 | sigma=1.5, USM=1.5, blend=0.8 |
| 002028_04.jpg | 1.0059 | 1.0070 | 1.0047-1.0071 | sigma=1.5, USM=1.5, blend=0.8 |