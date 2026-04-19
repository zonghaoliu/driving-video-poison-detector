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
| semantic | 6 | 2 | 0 |
| logical | 1 | 1 | 0 |
| decision | 0 | 0 | 0 |


## 3. Per-dimension score regression

| Dimension | MAE | Pearson r |
|---|---|---|
| semantic | 0.403 | -0.274 |
| logical | 0.375 | -0.442 |
| decision | 0.095 | nan |


## 4. Top-5 largest disagreements

### video 06 (sum |Δ| = 1.60)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.1 | 0.0 | 0.0 | False |
| VLM | semantic | 0.8 | 0.55 | 0.35 | True |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** The generated images exhibit several semantic errors, including blurred vehicles and missing objects, such as the motorcycle in the BACK_LEFT view. There are also cross-view inconsistencies with the bus and motorcycle.

### video 03 (sum |Δ| = 1.25)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.1 | 0.0 | 0.0 | False |
| VLM | semantic | 0.8 | 0.55 | 0.0 | True |

- **human reasoning:** Humans in the last frame seem to be deformed a bit, but can still be deciphered as human

- **VLM reasoning:** The most severe issue is the merging of two pedestrians into one in the BACK_RIGHT view, which poses a significant safety concern.

### video 05 (sum |Δ| = 1.17)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.1 | 0.0 | 0.0 | False |
| VLM | semantic | 0.575 | 0.35 | 0.35 | True |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** The FRONT_RIGHT and BACK_LEFT views show significant semantic errors with vehicles appearing merged and distorted. The FRONT view has a slightly blurred truck, and the BACK_RIGHT view has a slightly blurred stop sign, but these do not significantly impact identifiability.

### video 07 (sum |Δ| = 1.05)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.0 | 0.0 | 0.0 | False |
| VLM | semantic | 0.4 | 0.4 | 0.25 | False |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** The main issue is a distorted traffic sign in the FRONT view, which is unreadable. No cross-view inconsistencies were found.

### video 09 (sum |Δ| = 0.93)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.0 | 0.0 | 0.0 | False |
| VLM | logical | 0.425 | 0.5 | 0.0 | False |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** Multiple semantic errors are present, including missing pedestrians and cyclists, and a distorted car. Cross-view inconsistencies are observed with the cyclist and pedestrian appearing in some views but not others.
