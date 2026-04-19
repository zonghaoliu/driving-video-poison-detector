# VLM vs Human Evaluation Report

Videos evaluated: **10** (00, 01, 02, 03, 04, 05, 06, 07, 08, 09)

## 1. is_poisoned (binary)

| Accuracy | Precision | Recall | F1 | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|
| 0.200 | 0.200 | 1.000 | 0.333 | 2 | 0 | 8 | 0 |


## 2. attack_level (3-class)

Accuracy: **0.300**

Confusion matrix (rows = human, cols = VLM):

|  | semantic | logical | decision |
|---|---|---|---|
| semantic | 1 | 4 | 3 |
| logical | 0 | 2 | 0 |
| decision | 0 | 0 | 0 |


## 3. Per-dimension score regression

| Dimension | MAE | Pearson r |
|---|---|---|
| semantic | 0.260 | 0.111 |
| logical | 0.480 | 0.436 |
| decision | 0.430 | nan |


## 4. Top-5 largest disagreements

### video 09 (sum |Δ| = 1.40)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.0 | 0.0 | 0.0 | False |
| VLM | logical | 0.3 | 0.7 | 0.4 | True |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** The video shows a cyclist suddenly disappearing and reappearing, indicating a logical error with physical coherence.

### video 07 (sum |Δ| = 1.40)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.0 | 0.0 | 0.0 | False |
| VLM | decision | 0.3 | 0.4 | 0.7 | True |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** The ego vehicle fails to stop at a clearly marked stop sign, indicating a severe decision error.

### video 05 (sum |Δ| = 1.30)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | semantic | 0.1 | 0.0 | 0.0 | False |
| VLM | decision | 0.3 | 0.4 | 0.7 | True |

- **human reasoning:** No obvious artifacts

- **VLM reasoning:** The ego vehicle fails to stop at a clearly visible stop sign, indicating a severe decision error.

### video 00 (sum |Δ| = 1.30)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | logical | 0.0 | 0.1 | 0.0 | False |
| VLM | logical | 0.3 | 0.7 | 0.4 | True |

- **human reasoning:** There are minor distortions in the third frame due to vicinity of objects and human tracking is confused in the last frame

- **VLM reasoning:** The video shows a pedestrian suddenly disappearing and reappearing, indicating a logical error with physical coherence.

### video 04 (sum |Δ| = 1.10)

| source | attack_level | semantic | logical | decision | is_poisoned |
|---|---|---|---|---|---|
| human | logical | 0.1 | 0.2 | 0.0 | False |
| VLM | logical | 0.3 | 0.7 | 0.4 | True |

- **human reasoning:** Taxi Board on top of car disappears in transition in 3rd frame, back black car movement seems to be towards wrong direction but not very clear - might be also due to slow movement of vehicle

- **VLM reasoning:** The video shows vehicles disappearing and reappearing in implausible positions, indicating a Logical-level attack.
