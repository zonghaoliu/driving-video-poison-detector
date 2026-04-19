# VLM vs Human Evaluation Report

Videos evaluated: **10** (00, 01, 02, 03, 04, 05, 06, 07, 08, 09)

## 1. is_poisoned (binary)

| Accuracy | Precision | Recall | F1 | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|
| 0.200 | 0.000 | 0.000 | 0.000 | 0 | 2 | 6 | 2 |


## 2. attack_level (3-class)

Accuracy: **0.500**

Confusion matrix (rows = human, cols = VLM):

|  | semantic | logical | decision |
|---|---|---|---|
| semantic | 5 | 3 | 0 |
| logical | 2 | 0 | 0 |
| decision | 0 | 0 | 0 |


## 3. Per-dimension score regression

| Dimension | MAE | Pearson r |
|---|---|---|
| semantic | 0.570 | -0.269 |
| logical | 0.490 | -0.687 |
| decision | 0.190 | nan |


## 4. Top-5 largest disagreements

### video 06 (sum |Δ| = 2.10)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.1 | 0.0 | 0.0 | False |
| VLM | semantic | 0.8 | 0.7 | 0.7 | True |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** The generated images exhibit several semantic errors, including blurred vehicles and missing objects, such as the motorcycle in the BACK_LEFT view. There are also cross-view inconsistencies with the bus and motorcycle.

### video 05 (sum |Δ| = 1.90)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.1 | 0.0 | 0.0 | False |
| VLM | semantic | 0.8 | 0.5 | 0.7 | True |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** The FRONT_RIGHT and BACK_LEFT views show significant semantic errors with vehicles appearing merged and distorted. The FRONT view has a slightly blurred truck, and the BACK_RIGHT view has a slightly blurred stop sign, but these do not significantly impact identifiability.

### video 09 (sum |Δ| = 1.50)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.0 | 0.0 | 0.0 | False |
| VLM | semantic | 0.8 | 0.7 | 0.0 | True |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** Multiple semantic errors are present, including missing pedestrians and cyclists, and a distorted car. Cross-view inconsistencies are observed with the cyclist and pedestrian appearing in some views but not others.

### video 07 (sum |Δ| = 1.40)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.0 | 0.0 | 0.0 | False |
| VLM | logical | 0.4 | 0.5 | 0.5 | False |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** The generated frames show a blue car with jittery movement and inconsistent velocity, suggesting logical errors. Additionally, the car's trajectory towards the sidewalk indicates a potential decision error.

### video 03 (sum |Δ| = 1.40)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.1 | 0.0 | 0.0 | False |
| VLM | semantic | 0.8 | 0.7 | 0.0 | True |

- **human reasoning:** Humans in the last frame seem to be deformed a bit, but can still be deciphered as human

- **VLM reasoning:** The most severe issue is the merging of two pedestrians into one in the BACK_RIGHT view, which poses a significant safety concern.
