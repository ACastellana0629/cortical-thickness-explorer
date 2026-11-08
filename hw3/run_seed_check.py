"""HW3 robustness check: three training seeds, one fixed data split.

From the project root:
    python hw3/run_seed_check.py

Depends on the existing hw3/run_experiments.py. Makes no changes to it or
HW2 artifacts. Outputs aggregate-only files under hw3/results/seed_check/.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import run_experiments as exp

TRAINING_SEEDS = (42, 123, 2026)
FIXED_SPLIT_SEED = 42
MODEL_NAMES = ("hw2_baseline", "asymmetry_gated", "asymmetry_no_gate")
METRICS = ("test_accuracy", "test_macro_f1", "test_mci_recall")
OUT_DIR = Path(__file__).resolve().parent / "results" / "seed_check"


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    if exp.SEED != FIXED_SPLIT_SEED:
        raise RuntimeError("Unexpected split seed in run_experiments.py: expected 42")

    # Critical: build the split once BEFORE changing the training random seed.
    # Keeping participants/rows in the same partitions isolates training
    # randomness (initialization, batch shuffling and dropout).
    subsets, metadata = exp.split_and_transform()
    split_sizes = {k: len(v["y"]) for k, v in subsets.items()}
    print("Fixed train/validation/test split (seed 42):", split_sizes)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_seed = []
    try:
        for seed in TRAINING_SEEDS:
            exp.SEED = seed  # Existing run_one uses this for all training RNGs.
            print(f"\n========== Training seed {seed} ==========", flush=True)
            for model_name in MODEL_NAMES:
                print(f"\nModel: {model_name}", flush=True)
                result = exp.run_one(model_name, subsets)
                row = {
                    "seed": seed,
                    "name": model_name,
                    "parameters": result["parameter_count"],
                    "best_epoch": result["best_epoch"],
                    "test_accuracy": result["test"]["accuracy"],
                    "test_macro_f1": result["test"]["macro_f1"],
                    "test_mci_recall": result["test"]["mci_recall"],
                    "final_gate_weight": result["final_gate_weight"],
                }
                per_seed.append(row)
                print(
                    "  Test: accuracy={:.4f}; macro F1={:.4f}; MCI recall={:.4f}".format(
                        row["test_accuracy"], row["test_macro_f1"], row["test_mci_recall"]
                    ),
                    flush=True,
                )
    finally:
        exp.SEED = FIXED_SPLIT_SEED

    by_key = {(r["seed"], r["name"]): r for r in per_seed}
    summaries = []
    for name in MODEL_NAMES:
        rows = [by_key[(seed, name)] for seed in TRAINING_SEEDS]
        summary = {"name": name, "n_seeds": len(TRAINING_SEEDS)}
        for metric in METRICS:
            average, sd = mean_sd([float(r[metric]) for r in rows])
            summary[f"{metric}_mean"] = average
            summary[f"{metric}_sd"] = sd
        summaries.append(summary)

    paired = []
    for comparison in MODEL_NAMES[1:]:
        for metric in METRICS:
            differences = [
                float(by_key[(seed, comparison)][metric]) -
                float(by_key[(seed, "hw2_baseline")][metric])
                for seed in TRAINING_SEEDS
            ]
            average, sd = mean_sd(differences)
            paired.append({
                "comparison_minus_baseline": comparison,
                "metric": metric,
                "mean_difference": average,
                "sd_difference": sd,
                "seeds_better": sum(d > 0 for d in differences),
                "seeds_equal": sum(d == 0 for d in differences),
                "seeds_worse": sum(d < 0 for d in differences),
            })

    write_csv(OUT_DIR / "per_seed.csv", per_seed,
              ["seed", "name", "parameters", "best_epoch", *METRICS, "final_gate_weight"])
    write_csv(OUT_DIR / "summary.csv", summaries,
              ["name", "n_seeds", *[f"{m}_{s}" for m in METRICS for s in ("mean", "sd")]])
    write_csv(OUT_DIR / "paired_differences.csv", paired,
              ["comparison_minus_baseline", "metric", "mean_difference",
               "sd_difference", "seeds_better", "seeds_equal", "seeds_worse"])

    protocol = {
        "purpose": "Sensitivity to random initialization, dropout, and minibatch order",
        "split_seed": FIXED_SPLIT_SEED,
        "training_seeds": list(TRAINING_SEEDS),
        "split_sizes": split_sizes,
        "model_names": list(MODEL_NAMES),
        "note": (
            "Fixed stratified row-level split. Test set reused only for post-hoc "
            "robustness description, not model selection. Results across seeds "
            "do not demonstrate generalization across new participants or splits. "
            "With n=3, mean±SD is descriptive, not a significance test."
        ),
    }
    with (OUT_DIR / "protocol.json").open("w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)

    print("\n===== Summary: test results (mean ± sample SD across three seeds) =====")
    for row in summaries:
        print(f"\n{row['name']}")
        for metric in METRICS:
            print(f"  {metric}: {row[metric + '_mean']:.4f} ± {row[metric + '_sd']:.4f}")
    print("\n===== Paired differences from baseline (modified minus baseline) =====")
    for row in paired:
        print(f"  {row['comparison_minus_baseline']} / {row['metric']}: "
              f"{row['mean_difference']:+.4f} (better in {row['seeds_better']}/3 seeds)")
    print("\nSaved aggregate-only results in:", OUT_DIR)
    print("Files: per_seed.csv, summary.csv, paired_differences.csv, protocol.json")
    print("Original HW2 files and single-seed HW3 outputs were not modified.")


if __name__ == "__main__":
    main()
