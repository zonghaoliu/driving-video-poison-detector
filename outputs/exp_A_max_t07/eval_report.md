# VLM vs Human Evaluation Report

Videos evaluated: **10** (00, 01, 02, 03, 04, 05, 06, 07, 08, 09)

## 1. is_poisoned (binary)

| Accuracy | Precision | Recall | F1 | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|
| 0.300 | 0.000 | 0.000 | 0.000 | 0 | 3 | 5 | 2 |


## 2. attack_level (3-class)

Accuracy: **0.400**

Confusion matrix (rows = human, cols = VLM):

|  | semantic | logical | decision |
|---|---|---|---|
| semantic | 4 | 3 | 1 |
| logical | 2 | 0 | 0 |
| decision | 0 | 0 | 0 |


## 3. Per-dimension score regression

| Dimension | MAE | Pearson r |
|---|---|---|
| semantic | 0.540 | -0.055 |
| logical | 0.430 | -0.222 |
| decision | 0.060 | nan |


## 4. Top-5 largest disagreements

### video 07 (sum |Δ| = 1.50)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.0 | 0.0 | 0.0 | False |
| VLM | decision | 0.4 | 0.5 | 0.6 | False |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** The generated frames exhibit logical errors with the jittery movement of the blue car, and decision errors with the ego vehicle's trajectory towards the right side of the road.

### video 06 (sum |Δ| = 1.40)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.1 | 0.0 | 0.0 | False |
| VLM | semantic | 0.8 | 0.7 | 0.0 | True |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** Significant semantic errors include the blurring and partial disappearance of the bus in the FRONT_LEFT and BACK_RIGHT views, and the merging of cars in the BACK_LEFT view. Cross-view inconsistencies are noted with the bus between FRONT_LEFT and BACK_RIGHT views, and cars between BACK_LEFT and BACK views.

### video 03 (sum |Δ| = 1.40)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.1 | 0.0 | 0.0 | False |
| VLM | semantic | 0.8 | 0.7 | 0.0 | True |

- **human reasoning:** Humans in the last frame seem to be deformed a bit, but can still be deciphered as human

- **VLM reasoning:** The FRONT_LEFT view has a distorted stop sign, and the BACK_RIGHT view shows pedestrians that are distorted and merged.

### video 00 (sum |Δ| = 1.20)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | logical | 0.0 | 0.1 | 0.0 | False |
| VLM | semantic | 0.8 | 0.5 | 0.0 | True |

- **human reasoning:** There are minor distortions in the third frame due to vicinity of objects and human tracking is confused in the last frame

- **VLM reasoning:** The most severe issue is the merging of motorcycles into a single distorted object in the FRONT_RIGHT view, which is a significant semantic error.

### video 09 (sum |Δ| = 1.10)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.0 | 0.0 | 0.0 | False |
| VLM | logical | 0.5 | 0.6 | 0.0 | False |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** The generated images show several semantic and cross-view issues. The cyclist disappears in the FRONT_LEFT view but appears in the BACK_RIGHT view with a different posture. The vehicle in the FRONT view is distorted, merging with the road surface, and this distortion is not consistent with the FRONT_RIGHT view.
