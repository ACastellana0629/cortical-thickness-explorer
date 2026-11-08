# HW3: Hemisphere-aware MLP experiment

## Prediction problem

Use the same 905-observation, 68-feature dataset and three class labels as HW2: CN, MCI, and AD_TRAJECTORY. Do not commit `private_data/` to GitHub.

## Baseline

The **exact HW2 MLP architecture** (`training/model.py`): `68 -> 32 -> 16 -> 3`, ReLU activations, dropout 0.20; cross-entropy loss and AdamW optimizer.

## Hypothesis to discuss, revise, and own

The HW2 model struggled with MCI recall. Corresponding left/right cortical regions may have distinct thickness patterns. An explicit branch processing 34 matching left-minus-right thickness differences might make those patterns easier to learn than from a generic 68-feature MLP alone. This adds **no new information** because the differences are derived from the original measurements; it introduces a **structural inductive bias**. With only 905 observations, the extra branch could overfit or amplify noise. Please formulate the final design rationale in your own words.

## Experiments

1. `hw2_baseline`: original MLP, retrained.
2. `asymmetry_gated`: main MLP + 34->16 asymmetry pathway, fused before the classifier using a learned scalar sigmoid gate (starts around 0.12).
3. `asymmetry_no_gate`: same added pathway, but full-strength additive fusion; checks whether learned gating matters.

Training settings: same random seed 42, 70/15/15 stratified row split, batch size 64, learning rate 0.001, AdamW weight decay 0.0001, up to 300 epochs, early stopping patience 40 using validation loss, and train-only standardization. The cortical asymmetry features are calculated in raw millimeters and separately scaled using training data only.

Note: the split reproduces HW2's **row-level** approach, not a verified participant-level split. Correlated measurements from one participant across rows, if any, could limit performance interpretation. Test results should not be tuned against.

## Running

Activate your Conda environment and run from the repository root:

```bash
conda activate cortical-hw3
python hw3/run_experiments.py
```

All output stays in `hw3/results/` and includes `comparison.csv`, `summary.json`, and `loss_curves.png`. The script does **not** overwrite `models/` or `public/data/`.

## Discussion prompts

- Did the training and validation losses improve or diverge?
- Did MCI recall change, and what happened to macro F1 and overall accuracy?
- Did the learned gate shrink the asymmetry contribution, or emphasize it?
- Could the differences come from extra parameters, representational bias, or random training variability?
- What controlled next experiment would separate these explanations?

Results and conclusions should only be filled in after running experiments with the real data.
