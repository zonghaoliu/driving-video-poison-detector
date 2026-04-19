# VLM vs Human Evaluation Report

Videos evaluated: **10** (00, 01, 02, 03, 04, 05, 06, 07, 08, 09)

## 1. is_poisoned (binary)

| Accuracy | Precision | Recall | F1 | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|
| 0.500 | 0.000 | 0.000 | 0.000 | 0 | 5 | 3 | 2 |


## 2. attack_level (3-class)

Accuracy: **0.700**

Confusion matrix (rows = human, cols = VLM):

|  | semantic | logical | decision |
|---|---|---|---|
| semantic | 7 | 1 | 0 |
| logical | 2 | 0 | 0 |
| decision | 0 | 0 | 0 |


## 3. Per-dimension score regression

| Dimension | MAE | Pearson r |
|---|---|---|
| semantic | 0.460 | -0.305 |
| logical | 0.357 | -0.428 |
| decision | 0.063 | nan |


## 4. Top-5 largest disagreements

### video 06 (sum |Δ| = 1.57)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.1 | 0.0 | 0.0 | False |
| VLM | semantic | 0.8 | 0.633 | 0.233 | True |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** The generated images exhibit several semantic errors, including blurred vehicles and missing objects, such as the motorcycle in the BACK_LEFT view. There are also cross-view inconsistencies with the bus and motorcycle.

### video 03 (sum |Δ| = 1.33)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.1 | 0.0 | 0.0 | False |
| VLM | semantic | 0.8 | 0.633 | 0.0 | True |

- **human reasoning:** Humans in the last frame seem to be deformed a bit, but can still be deciphered as human

- **VLM reasoning:** The most severe issue is the merging of two pedestrians into one in the BACK_RIGHT view, which poses a significant safety concern.

### video 09 (sum |Δ| = 1.13)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.0 | 0.0 | 0.0 | False |
| VLM | semantic | 0.567 | 0.567 | 0.0 | False |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** Multiple semantic errors are present, including missing pedestrians and cyclists, and a distorted car. Cross-view inconsistencies are observed with the cyclist and pedestrian appearing in some views but not others.

### video 05 (sum |Δ| = 1.07)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.1 | 0.0 | 0.0 | False |
| VLM | semantic | 0.7 | 0.233 | 0.233 | True |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** The FRONT_RIGHT and BACK_LEFT views show significant semantic errors with vehicles appearing merged and distorted. The FRONT view has a slightly blurred truck, and the BACK_RIGHT view has a slightly blurred stop sign, but these do not significantly impact identifiability.

### video 00 (sum |Δ| = 0.90)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | logical | 0.0 | 0.1 | 0.0 | False |
| VLM | semantic | 0.667 | 0.333 | 0.0 | False |

- **human reasoning:** There are minor distortions in the third frame due to vicinity of objects and human tracking is confused in the last frame

- **VLM reasoning:** The generated images show several semantic errors, including distortion and merging of vehicles, as well as blurring of a pedestrian. No cross-view inconsistencies were observed.
