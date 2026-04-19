# VLM vs Human Evaluation Report

Videos evaluated: **10** (00, 01, 02, 03, 04, 05, 06, 07, 08, 09)

## 1. is_poisoned (binary)

| Accuracy | Precision | Recall | F1 | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|
| 0.200 | 0.125 | 0.500 | 0.200 | 1 | 1 | 7 | 1 |


## 2. attack_level (3-class)

Accuracy: **0.600**

Confusion matrix (rows = human, cols = VLM):

|  | semantic | logical | decision |
|---|---|---|---|
| semantic | 5 | 3 | 0 |
| logical | 1 | 1 | 0 |
| decision | 0 | 0 | 0 |


## 3. Per-dimension score regression

| Dimension | MAE | Pearson r |
|---|---|---|
| semantic | 0.570 | -0.163 |
| logical | 0.530 | -0.162 |
| decision | 0.000 | nan |


## 4. Top-5 largest disagreements

### video 09 (sum |Δ| = 1.50)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.0 | 0.0 | 0.0 | False |
| VLM | logical | 0.7 | 0.8 | 0.0 | True |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** The generated images exhibit several semantic errors, including missing and distorted objects. Cross-view inconsistencies are present, particularly with the cyclist and the black car.

### video 03 (sum |Δ| = 1.50)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.1 | 0.0 | 0.0 | False |
| VLM | semantic | 0.8 | 0.8 | 0.0 | True |

- **human reasoning:** Humans in the last frame seem to be deformed a bit, but can still be deciphered as human

- **VLM reasoning:** The most severe semantic error is in the BACK_RIGHT view, where two pedestrians are merged into one. There are no cross-view inconsistencies observed across the camera views.

### video 05 (sum |Δ| = 1.40)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.1 | 0.0 | 0.0 | False |
| VLM | logical | 0.7 | 0.8 | 0.0 | True |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** A vehicle appears suddenly in frame 1 and remains in an implausible position throughout the sequence, partially on the sidewalk. No decision errors related to the ego vehicle's behavior were observed.

### video 00 (sum |Δ| = 1.40)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | logical | 0.0 | 0.1 | 0.0 | False |
| VLM | logical | 0.7 | 0.8 | 0.0 | True |

- **human reasoning:** There are minor distortions in the third frame due to vicinity of objects and human tracking is confused in the last frame

- **VLM reasoning:** The generated images exhibit significant semantic errors, including blurring and distortion of objects across all views. There are also notable cross-view inconsistencies, particularly with the appearance of pedestrians and vehicles between adjacent camera views.

### video 06 (sum |Δ| = 1.30)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.1 | 0.0 | 0.0 | False |
| VLM | semantic | 0.7 | 0.7 | 0.0 | True |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** Significant semantic errors are present, particularly with the distortion and merging of vehicles in the BACK_LEFT view and the blurring of the bus in the FRONT_LEFT and BACK_RIGHT views. Cross-view inconsistencies are noted with the bus and cars appearing differently across views.
