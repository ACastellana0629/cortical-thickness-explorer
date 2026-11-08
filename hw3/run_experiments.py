"""HW3: compare the original HW2 MLP with a hemispheric-asymmetry branch.

Run from the project root:
    python hw3/run_experiments.py

Only aggregate metrics and plots are written to hw3/results/. The original
HW2 model, website files, and private research data are not modified.
"""

from __future__ import annotations

import copy
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Make this runnable as `python hw3/run_experiments.py` from any directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from training.data import GROUP_ORDER, load_modeling_data  # noqa: E402
from training.model import CorticalMLP  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "hw3" / "results"
SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
MAX_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 40
HIDDEN_DIMS = (32, 16)
DROPOUT = 0.20


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_hemisphere_indices(feature_names: list[str]) -> tuple[list[int], list[int]]:
    """Pair anatomical regions by name, never by assuming positions match."""
    left = {
        name.removeprefix("lh_"): i
        for i, name in enumerate(feature_names)
        if name.startswith("lh_")
    }
    right = {
        name.removeprefix("rh_"): i
        for i, name in enumerate(feature_names)
        if name.startswith("rh_")
    }
    if len(left) != 34 or len(right) != 34 or set(left) != set(right):
        raise ValueError("Expected exactly 34 corresponding LH/RH regions.")
    regions = sorted(left)
    return [left[r] for r in regions], [right[r] for r in regions]


def split_and_transform():
    """Reproduce HW2's stratified, row-level 70/15/15 split.

    Scalers are learned on training rows ONLY. Hemispheric differences are
    computed in the raw thickness units before they are scaled separately.
    """
    features, labels, feature_names, metadata = load_modeling_data()
    x = features.to_numpy(dtype=np.float32)
    y = labels.to_numpy(dtype=np.int64)
    lh, rh = make_hemisphere_indices(list(feature_names))
    asym = x[:, lh] - x[:, rh]  # paired left-minus-right differences in mm

    indices = np.arange(len(y))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=SEED, stratify=y
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=SEED, stratify=y[temp_idx]
    )

    raw_scaler = StandardScaler().fit(x[train_idx])
    asym_scaler = StandardScaler().fit(asym[train_idx])

    subsets = {}
    for label, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        subsets[label] = {
            "x": raw_scaler.transform(x[idx]).astype(np.float32),
            "a": asym_scaler.transform(asym[idx]).astype(np.float32),
            "y": y[idx],
        }
    return subsets, metadata


class AsymmetryMLP(nn.Module):
    """Original HW2 MLP plus a trainable pathway for hemispheric differences.

    Feature extraction is the original architecture up through its 16-unit
    hidden layer. The 34-dimensional asymmetry branch also makes 16 features,
    then the two representations are combined before the original output layer.
    """

    def __init__(self, learned_gate: bool):
        super().__init__()
        self.base = CorticalMLP(
            input_dim=68, hidden_dims=HIDDEN_DIMS,
            num_classes=len(GROUP_ORDER), dropout=DROPOUT
        )
        self.asym_branch = nn.Sequential(
            nn.Linear(34, 16),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        nn.init.xavier_uniform_(self.asym_branch[0].weight)
        nn.init.zeros_(self.asym_branch[0].bias)
        self.learned_gate = learned_gate
        if learned_gate:
            # sigmoid(-2) is about 0.12: initially trust the original pathway.
            self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor, asym: torch.Tensor) -> torch.Tensor:
        base_features = self.base.network[:-1](x)
        additional_features = self.asym_branch(asym)
        if self.learned_gate:
            weight = torch.sigmoid(self.gate_logit)
        else:
            weight = 1.0  # no gating: add the entire asymmetry representation
        return self.base.network[-1](base_features + weight * additional_features)


def model_forward(model: nn.Module, x: torch.Tensor, a: torch.Tensor):
    if isinstance(model, CorticalMLP):
        return model(x)
    return model(x, a)


def evaluate(model: nn.Module, partition: dict, criterion: nn.Module):
    model.eval()
    x = torch.from_numpy(partition["x"])
    a = torch.from_numpy(partition["a"])
    y = torch.from_numpy(partition["y"])
    with torch.no_grad():
        logits = model_forward(model, x, a)
        loss = float(criterion(logits, y).item())
        pred = logits.argmax(dim=1).numpy()
    return loss, float(accuracy_score(partition["y"], pred)), pred


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    per_class = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "mci_recall": float(per_class[1]),
        "recall_by_class": {name: float(v) for name, v in zip(GROUP_ORDER, per_class)},
        "confusion_matrix_rows_actual_cols_predicted": confusion_matrix(
            y, pred, labels=[0, 1, 2]
        ).tolist(),
    }


def run_one(name: str, subsets: dict) -> dict:
    # Match the main model's random initialization across experiments.
    set_seed(SEED)
    if name == "hw2_baseline":
        model = CorticalMLP(68, HIDDEN_DIMS, len(GROUP_ORDER), DROPOUT)
    elif name == "asymmetry_gated":
        model = AsymmetryMLP(learned_gate=True)
    elif name == "asymmetry_no_gate":
        model = AsymmetryMLP(learned_gate=False)
    else:
        raise ValueError(name)

    criterion = nn.CrossEntropyLoss()  # same objective as HW2
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    training = subsets["train"]
    ds = TensorDataset(
        torch.from_numpy(training["x"]),
        torch.from_numpy(training["a"]),
        torch.from_numpy(training["y"]),
    )
    # Same minibatch order across experiments.
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED)
    )

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    no_improvement = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for x, a, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model_forward(model, x, a), y)
            loss.backward()
            optimizer.step()

        train_loss, train_acc, _ = evaluate(model, training, criterion)
        val_loss, val_acc, _ = evaluate(model, subsets["val"], criterion)
        history.append({
            "epoch": epoch,
            "training_loss": train_loss,
            "validation_loss": val_loss,
            "training_accuracy": train_acc,
            "validation_accuracy": val_acc,
        })
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"    Epoch {epoch:3} | training loss {train_loss:.4f} "
                f"| validation loss {val_loss:.4f}"
            )
        if no_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"    Early stopping: epoch {epoch}; best was {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training never produced a best checkpoint")
    model.load_state_dict(best_state)

    per_split = {}
    for subset_name in ("train", "val", "test"):
        partition = subsets[subset_name]
        loss, _, pred = evaluate(model, partition, criterion)
        per_split[subset_name] = {"loss": loss, **metrics(partition["y"], pred)}

    gate = None
    if isinstance(model, AsymmetryMLP) and model.learned_gate:
        gate = float(torch.sigmoid(model.gate_logit).detach().item())

    return {
        "name": name,
        "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "final_gate_weight": gate,
        "train": per_split["train"],
        "validation": per_split["val"],
        "test": per_split["test"],
        "history": history,
    }


def save_plots(results: list[dict]):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for result in results:
        epochs = [point["epoch"] for point in result["history"]]
        axes[0].plot(epochs, [p["training_loss"] for p in result["history"]],
                     label=result["name"])
        axes[1].plot(epochs, [p["validation_loss"] for p in result["history"]],
                     label=result["name"])
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Cross-entropy loss")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "loss_curves.png", dpi=160)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading HW2 data and reproducing the same train/val/test split...")
    subsets, cohort = split_and_transform()
    print("Split sizes:", {name: len(part["y"]) for name, part in subsets.items()})
    results = []
    for name in ("hw2_baseline", "asymmetry_gated", "asymmetry_no_gate"):
        print(f"\nExperiment: {name}")
        result = run_one(name, subsets)
        results.append(result)
        test = result["test"]
        print(
            f"    Test accuracy={test['accuracy']:.4f}, "
            f"macro F1={test['macro_f1']:.4f}, "
            f"MCI recall={test['mci_recall']:.4f}"
        )
        print("    Confusion matrix (rows: actual, columns: predicted):")
        for row in test["confusion_matrix_rows_actual_cols_predicted"]:
            print("   ", row)
        if result["final_gate_weight"] is not None:
            print(f"    Learned gate weight={result['final_gate_weight']:.4f}")

    payload = {
        "design_hypothesis": (
            "An explicit pathway for corresponding LH-RH thickness differences "
            "may improve three-class classification, especially MCI recall."
        ),
        "caveat": (
            "Differences are deterministic functions of the existing inputs; "
            "this tests an inductive bias, not new information. Test-set "
            "results are exploratory and should not be used for tuning."
        ),
        "class_order": GROUP_ORDER,
        "cohort_aggregate": cohort,
        "split": {name: len(part["y"]) for name, part in subsets.items()},
        "random_seed": SEED,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": EARLY_STOPPING_PATIENCE,
        "results": results,
    }
    with (RESULTS_DIR / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    fields = ["name", "parameter_count", "best_epoch", "test_accuracy",
              "test_macro_f1", "test_mci_recall", "final_gate_weight"]
    with (RESULTS_DIR / "comparison.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "name": r["name"],
                "parameter_count": r["parameter_count"],
                "best_epoch": r["best_epoch"],
                "test_accuracy": r["test"]["accuracy"],
                "test_macro_f1": r["test"]["macro_f1"],
                "test_mci_recall": r["test"]["mci_recall"],
                "final_gate_weight": r["final_gate_weight"],
            })
    save_plots(results)
    print("\nSaved aggregate-only experiment files to:", RESULTS_DIR)
    print("  comparison.csv, summary.json, loss_curves.png")
    print("Your original HW2 files were not modified.")


if __name__ == "__main__":
    main()
