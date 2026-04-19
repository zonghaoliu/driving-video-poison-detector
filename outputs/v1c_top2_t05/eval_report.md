# VLM vs Human Evaluation Report

Videos evaluated: **10** (00, 01, 02, 03, 04, 05, 06, 07, 08, 09)

## 1. is_poisoned (binary)

| Accuracy | Precision | Recall | F1 | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|
| 0.100 | 0.111 | 0.500 | 0.182 | 1 | 0 | 8 | 1 |


## 2. attack_level (3-class)

Accuracy: **0.800**

Confusion matrix (rows = human, cols = VLM):

|  | semantic | logical | decision |
|---|---|---|---|
| semantic | 8 | 0 | 0 |
| logical | 2 | 0 | 0 |
| decision | 0 | 0 | 0 |


## 3. Per-dimension score regression

| Dimension | MAE | Pearson r |
|---|---|---|
| semantic | 0.670 | -0.692 |
| logical | 0.495 | -0.146 |
| decision | 0.125 | nan |


## 4. Top-5 largest disagreements

### video 06 (sum |Δ| = 1.70)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.1 | 0.0 | 0.0 | False |
| VLM | semantic | 0.8 | 0.7 | 0.3 | True |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** The generated images exhibit semantic errors such as blurring and deformation of vehicles and objects, affecting identifiability. Cross-view inconsistencies are observed with the bus and motorcycle appearing differently across views.

### video 05 (sum |Δ| = 1.60)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.1 | 0.0 | 0.0 | False |
| VLM | semantic | 0.8 | 0.5 | 0.4 | True |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** The generated images exhibit several semantic errors, including blurring, distortion, and merging of vehicles. The most severe issue is the missing stop sign in the BACK_RIGHT view.

### video 09 (sum |Δ| = 1.50)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.0 | 0.0 | 0.0 | False |
| VLM | semantic | 0.8 | 0.7 | 0.0 | True |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** The most severe issue is the significant change in the cyclist's position and posture in the FRONT_LEFT and BACK_RIGHT views, indicating a semantic error. There are also minor blurring issues in other views.

### video 03 (sum |Δ| = 1.50)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.1 | 0.0 | 0.0 | False |
| VLM | semantic | 0.8 | 0.8 | 0.0 | True |

- **human reasoning:** Humans in the last frame seem to be deformed a bit, but can still be deciphered as human

- **VLM reasoning:** The most severe issue is the merging of two pedestrians into one in the BACK_RIGHT view, which is a significant semantic error. Other views show minor blurring affecting identifiability.

### video 07 (sum |Δ| = 1.35)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.0 | 0.0 | 0.0 | False |
| VLM | semantic | 0.8 | 0.25 | 0.3 | True |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** The generated images exhibit significant semantic errors, including blurring and distortion of vehicles and road markings. The blue car is particularly affected, appearing distorted and merged with road elements in the FRONT and FRONT_RIGHT views.
