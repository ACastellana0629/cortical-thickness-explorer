# DS7400 HW3 — Cortical-Thickness Network Design Experiment

**Purpose:** Explanation notes for my HW3 report and short demonstration video.  
**Project:** Cortical Thickness Explorer  
**Question:** Can explicitly modeling left–right cortical-thickness differences improve classification, especially for mild cognitive impairment (MCI)?

## 1. The problem: What limitation did I identify in HW2?

In HW2, I trained a multilayer perceptron (MLP) to classify observations using **68 cortical-thickness features** from 34 anatomical regions in each hemisphere. The three prediction groups were **cognitively normal (CN), mild cognitive impairment (MCI), and AD_TRAJECTORY** (baseline Alzheimer's dementia or a later converter, as defined in this project).

The baseline architecture was **68 inputs → 32 hidden units → 16 hidden units → 3 outputs**, using ReLU activations and 20% dropout. Although the network learned useful distinctions, it had trouble recognizing MCI. In our reproduced HW3 baseline run (seed 42), it correctly classified only **6 of 39 MCI test observations (15.38% recall)**. The dataset also was relatively small (**905 observations**), so I was concerned about overfitting and whether the original network was learning the most informative relationships among the features.

**The limitation I chose to investigate:** The original MLP received measurements from both hemispheres but had no explicit mechanism for processing differences between matching left- and right-hemisphere regions.

## 2. My proposed design: What did I change, and why?

My hypothesis was that **left–right cortical-thickness asymmetry** could be useful for distinguishing the diagnostic groups, particularly MCI. This was a design hypothesis, **not an established claim** that asymmetry predicts progression to Alzheimer's dementia.

I retained the original MLP and added a **second pathway** that accepts 34 paired differences, computed as **left-hemisphere thickness minus right-hemisphere thickness** for each corresponding region. Those differences were standardized using parameters fitted to the training partition only. The new branch maps 34 asymmetry values into a 16-dimensional representation using a linear layer, ReLU, and 20% dropout.

The original MLP also produces a 16-dimensional representation. In my main modification, I combine them using a **trainable scalar gate**:

`combined_features = original_features + sigmoid(gate_logit) * asymmetry_features`

The gate starts around **0.119** so the original pathway has more influence initially; it can adjust through training. A possible failure mode is that the branch adds redundant information or complexity, or the gate scales down information that would otherwise help. The ungated version adds the two representations directly with a weight of 1.

- Original MLP: **2,787** trainable parameters.
- MLP + gated asymmetry branch: **3,348** parameters.
- MLP + ungated asymmetry branch: **3,347** parameters.

## 3. My experiments: How did I compare the networks?

I trained **three architectures**: (1) the original HW2 MLP, (2) the MLP with a gated asymmetry branch, and (3) the same asymmetry branch **without** the gate. The third model is an **ablation** that tests whether the learned gate itself contributes to performance.

To make comparisons as controlled as possible, all models used the **same stratified row-level split** of the 905 observations: **633 training, 136 validation, and 136 test** (approximately 70%/15%/15%). The split was held fixed at seed 42. Feature scalers were fitted only on the training rows. I used **cross-entropy loss, AdamW (learning rate 0.001, weight decay 0.0001), batch size 64**, up to 300 epochs, and **early stopping after 40 epochs without improvement in validation loss**. The checkpoint with the lowest validation loss was evaluated on the test set.

First, I compared the architectures with training seed **42**, examining the loss curves and confusion matrices. Then I repeated training using **seeds 42, 123, and 2026**, while keeping the data split unchanged. This tests sensitivity to random initialization, dropout, and minibatch ordering—not sensitivity to the choice of test participants or the split itself.

My primary metrics were **test accuracy, macro F1, and MCI recall**. I emphasized MCI recall because this was the main weakness motivating the design. I also inspected the training and validation cross-entropy curves for evidence of overfitting.

## 4. My results: What improved, what did not, and what surprised me?

### Mean test results over three training seeds

Values show **mean ± sample standard deviation** over the three seeds. Every run uses the same test observations.

| Architecture | Test accuracy | Test macro F1 | MCI recall |
|---|---:|---:|---:|
| Original MLP | 54.17% ± 1.53 pp | 0.4387 ± 0.0175 | 10.26% ± 4.44 pp |
| Asymmetry + learned gate | **56.13% ± 0.42 pp** | **0.4496 ± 0.0148** | 13.68% ± 3.92 pp |
| Asymmetry without gate | 53.92% ± 0.42 pp | 0.4482 ± 0.0044 | **17.95% ± 2.56 pp** |

Relative to baseline, the **gated** model improved average accuracy by **1.96 percentage points** and mean MCI recall by **3.42 percentage points**. The **ungated** model improved mean MCI recall by **7.69 percentage points**, but average accuracy was slightly lower (**−0.25 percentage points**). The ungated model's MCI recall was higher than baseline for **all three seeds**; the gated model's was higher for **two of three**.

### Why the repeated runs mattered

The first seed-42 run alone suggested the gated modification performed worse on MCI recall: **12.82%**, compared with **15.38%** for the baseline. But across three seeds, the gated model had a **higher average** MCI recall and accuracy. This was the most surprising result: my assessment of the design changed once I checked training randomness.

The loss curves from seed 42 showed that **training loss kept decreasing while validation loss flattened and then increased**. This indicates overfitting as training continued. The original MLP's best validation-loss checkpoint was at epoch **42**; both modified models reached theirs at epoch **27** in that run. The gated model's learned scalar weight ended at approximately **0.132** in that run, though the gate value alone does not tell us the branch's actual contribution without considering its output magnitude.

## 5. What I learned: What conclusions can I draw?

My main conclusion is that **a biologically motivated feature pathway can change class-specific performance without producing a large or consistent improvement in overall classification**. Across three seeds, the gated model showed small mean gains in accuracy and macro F1. The ungated design was more consistently helpful for **MCI recall**, suggesting that the new asymmetry information may be useful, but **not demonstrating that the gate itself is responsible**. This contrasts with my expectation that learning how much to trust the new pathway would necessarily help.

I also learned why **one training run is not enough** to judge a small architectural modification. The differences between models were modest and sometimes changed direction across seeds. Even the best mean MCI recall was **17.95%**, so none of these designs solved the MCI identification problem.

**Limitations and next steps:** This is a small, preliminary experiment with **only three training seeds and one fixed row-level data split**. The MCI test group contains only **39 observations**, meaning each additional correctly recognized MCI observation changes recall by about **2.56 percentage points**. A row-level split does not, by itself, guarantee participant-level independence. These results do **not** establish statistical significance, clinical utility, or generalization to an external cohort. If extending the study, I would evaluate participant-level separation, repeat experiments across multiple splits or cross-validation folds, examine branch-output magnitudes and gate behavior, and seek external validation.

---

## Files to show in the video

- `training/model.py` — the original HW2 baseline model.
- `hw3/run_experiments.py` — the modified asymmetry models and controlled training setup.
- `hw3/results/loss_curves.png` — seed-42 training/validation loss curves.
- `hw3/results/seed_check/summary.csv` — three-seed aggregate comparison.
- `hw3/results/seed_check/paired_differences.csv` — paired modifications minus baseline.

**Suggested video flow (under 3 minutes):** problem (~30 s), proposed design (~40 s), experiment setup (~35 s), results (~50 s), lessons/limitations (~25 s). Use the tables and plots rather than reading code line by line.

**Reproduction (from project root with the Python environment active):**

```bash
python hw3/run_experiments.py
python hw3/run_seed_check.py
```

**Data safety:** The original `private_data/` files are local research inputs and must **not** be committed to GitHub. Only code, documentation, aggregate metrics, and plots should be shared. The existing project `.gitignore` already excludes `private_data/` and Python `__pycache__/` folders.
