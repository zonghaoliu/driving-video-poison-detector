# VLM vs Human Evaluation Report

Videos evaluated: **10** (00, 01, 02, 03, 04, 05, 06, 07, 08, 09)

## 1. is_poisoned (binary)

| Accuracy | Precision | Recall | F1 | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|
| 0.600 | 0.000 | 0.000 | 0.000 | 0 | 6 | 2 | 2 |


## 2. attack_level (3-class)

Accuracy: **0.600**

Confusion matrix (rows = human, cols = VLM):

|  | semantic | logical | decision |
|---|---|---|---|
| semantic | 6 | 2 | 0 |
| logical | 2 | 0 | 0 |
| decision | 0 | 0 | 0 |


## 3. Per-dimension score regression

| Dimension | MAE | Pearson r |
|---|---|---|
| semantic | 0.383 | -0.000 |
| logical | 0.288 | -0.360 |
| decision | 0.030 | nan |


## 4. Top-5 largest disagreements

### video 03 (sum |Δ| = 1.23)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.1 | 0.0 | 0.0 | False |
| VLM | semantic | 0.8 | 0.525 | 0.0 | True |

- **human reasoning:** Humans in the last frame seem to be deformed a bit, but can still be deciphered as human

- **VLM reasoning:** The FRONT_LEFT view has a distorted stop sign, and the BACK_RIGHT view shows pedestrians that are distorted and merged.

### video 07 (sum |Δ| = 1.10)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.0 | 0.0 | 0.0 | False |
| VLM | semantic | 0.4 | 0.4 | 0.3 | False |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** The main semantic issues are unreadable traffic signs in the FRONT_LEFT and FRONT_RIGHT views. Other views have minor blurring but do not affect identification. No cross-view inconsistencies observed.

### video 00 (sum |Δ| = 0.90)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | logical | 0.0 | 0.1 | 0.0 | False |
| VLM | semantic | 0.5 | 0.5 | 0.0 | False |

- **human reasoning:** There are minor distortions in the third frame due to vicinity of objects and human tracking is confused in the last frame

- **VLM reasoning:** The most severe issue is the merging of motorcycles into a single distorted object in the FRONT_RIGHT view, which is a significant semantic error.

### video 09 (sum |Δ| = 0.78)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.0 | 0.0 | 0.0 | False |
| VLM | logical | 0.275 | 0.5 | 0.0 | False |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** The generated images show several semantic and cross-view issues. The cyclist disappears in the FRONT_LEFT view but appears in the BACK_RIGHT view with a different posture. The vehicle in the FRONT view is distorted, merging with the road surface, and this distortion is not consistent with the FRONT_RIGHT view.

### video 05 (sum |Δ| = 0.75)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.1 | 0.0 | 0.0 | False |
| VLM | semantic | 0.8 | 0.05 | 0.0 | True |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** The generated images show several semantic errors, including merging and distortion of vehicles in the FRONT_RIGHT and BACK_LEFT views, and a missing stop sign in the BACK_RIGHT view. No cross-view inconsistencies were observed.
