# HW3 three-seed reproducibility check

Unzip this package **from your project root** to add `hw3/run_seed_check.py` alongside the already existing `hw3/run_experiments.py`.

```bash
unzip -o ~/Downloads/HW3_seed_check.zip -d .
python hw3/run_seed_check.py
cat hw3/results/seed_check/summary.csv
cat hw3/results/seed_check/paired_differences.csv
```

This runner reuses the previous architecture and training routine. It fixes the original stratified **row-level** split with seed **42** once, then re-trains all three models at initialization/training seeds **42**, **123**, and **2026**. The within-seed comparisons use the same split and training seed. Batch order and model initialization can vary across seeds; your dataset and train-only scalers do not vary. It does not make a new 70/15/15 split per seed.

Only aggregate results are saved in `hw3/results/seed_check/`: model-by-seed metrics, mean and sample SD across three seeds, paired differences from baseline, and protocol metadata. This runner does **not** overwrite your earlier HW3 results or HW2 artifacts. It does not write patient-level predictions.

**Interpretation:** three training seeds show sensitivity to training randomness only. They are not independent held-out datasets, uncertainty intervals, a proof of statistical significance, or evidence that the same result will hold across participant-level splits. Do not choose the model by repeatedly inspecting its test performance.
