"""Train and evaluate the cortical-thickness classification model."""

import copy
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data import GROUP_ORDER, INDEX_TO_LABEL, load_modeling_data
from model import CorticalMLP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = PROJECT_ROOT / "models"
PUBLIC_DATA_DIRECTORY = PROJECT_ROOT / "public" / "data"

MODEL_FILE = MODEL_DIRECTORY / "cortical_mlp.pt"
PREPROCESSING_FILE = MODEL_DIRECTORY / "preprocessing.json"
METRICS_FILE = PUBLIC_DATA_DIRECTORY / "model_metrics.json"
DEMO_PROFILES_FILE = PUBLIC_DATA_DIRECTORY / "demo_profiles.json"
CURVES_FILE = PUBLIC_DATA_DIRECTORY / "training_curves.png"

RANDOM_SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 40
HIDDEN_DIMS = (32, 16)
DROPOUT = 0.20


def set_random_seeds(seed):
    """Make the main sources of randomness reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_tensor_loader(features, labels, shuffle):
    """Create a PyTorch mini-batch data loader."""
    feature_tensor = torch.tensor(features, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(feature_tensor, label_tensor)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )


def evaluate_mlp(model, features, labels, loss_function):
    """Evaluate the MLP without updating its parameters."""
    model.eval()

    feature_tensor = torch.tensor(features, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)

    with torch.no_grad():
        logits = model(feature_tensor)
        loss = loss_function(logits, label_tensor).item()
        predictions = logits.argmax(dim=1).cpu().numpy()

    accuracy = accuracy_score(labels, predictions)

    return loss, float(accuracy), predictions


def summarize_predictions(labels, predictions):
    """Calculate classification metrics in JSON-compatible form."""
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro")
        ),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted")
        ),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=list(range(len(GROUP_ORDER))),
        ).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=list(range(len(GROUP_ORDER))),
            target_names=GROUP_ORDER,
            output_dict=True,
            zero_division=0,
        ),
    }


def save_training_curves(history):
    """Save loss and accuracy curves as a public, anonymous figure."""
    epochs = [record["epoch"] for record in history]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    figure.patch.set_facecolor("#08051a")

    for axis in axes:
        axis.set_facecolor("#0e0a2b")
        axis.tick_params(colors="#ddd8ff")
        axis.xaxis.label.set_color("#ddd8ff")
        axis.yaxis.label.set_color("#ddd8ff")
        axis.title.set_color("#ffffff")

        for spine in axis.spines.values():
            spine.set_color("#595080")

        axis.grid(alpha=0.18, color="#b9adff")

    axes[0].plot(
        epochs,
        [record["training_loss"] for record in history],
        label="Training",
        color="#35d9ff",
        linewidth=2,
    )
    axes[0].plot(
        epochs,
        [record["validation_loss"] for record in history],
        label="Validation",
        color="#ff4fc3",
        linewidth=2,
    )
    axes[0].set_title("Loss vs. Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()

    axes[1].plot(
        epochs,
        [record["training_accuracy"] for record in history],
        label="Training",
        color="#35d9ff",
        linewidth=2,
    )
    axes[1].plot(
        epochs,
        [record["validation_accuracy"] for record in history],
        label="Validation",
        color="#ff4fc3",
        linewidth=2,
    )
    axes[1].set_title("Accuracy vs. Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    figure.suptitle(
        "Cortical MLP Training Monitor",
        color="#ffffff",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(
        CURVES_FILE,
        dpi=180,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def main():
    """Run the complete benchmark, training, and evaluation pipeline."""
    set_random_seeds(RANDOM_SEED)

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    PUBLIC_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    features, labels, feature_names, cohort_metadata = (
        load_modeling_data()
    )

    feature_array = features.to_numpy(dtype=np.float32)
    label_array = labels.to_numpy(dtype=np.int64)

    (
        training_features_raw,
        temporary_features_raw,
        training_labels,
        temporary_labels,
    ) = train_test_split(
        feature_array,
        label_array,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=label_array,
    )

    (
        validation_features_raw,
        test_features_raw,
        validation_labels,
        test_labels,
    ) = train_test_split(
        temporary_features_raw,
        temporary_labels,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temporary_labels,
    )

    scaler = StandardScaler()

    training_features = scaler.fit_transform(
        training_features_raw
    ).astype(np.float32)

    validation_features = scaler.transform(
        validation_features_raw
    ).astype(np.float32)

    test_features = scaler.transform(
        test_features_raw
    ).astype(np.float32)

    print("\nData split:")
    print(f"  Training:   {len(training_labels)}")
    print(f"  Validation: {len(validation_labels)}")
    print(f"  Test:       {len(test_labels)}")

    majority_class = int(
        np.bincount(training_labels).argmax()
    )
    majority_predictions = np.full_like(
        test_labels,
        majority_class,
    )
    majority_results = summarize_predictions(
        test_labels,
        majority_predictions,
    )

    benchmark = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    benchmark.fit(training_features, training_labels)

    benchmark_predictions = benchmark.predict(test_features)
    benchmark_results = summarize_predictions(
        test_labels,
        benchmark_predictions,
    )

    print("\nBenchmarks:")
    print(
        "  Majority-class test accuracy:",
        f"{majority_results['accuracy']:.4f}",
    )
    print(
        "  Logistic-regression test accuracy:",
        f"{benchmark_results['accuracy']:.4f}",
    )

    model = CorticalMLP(
        input_dim=len(feature_names),
        hidden_dims=HIDDEN_DIMS,
        num_classes=len(GROUP_ORDER),
        dropout=DROPOUT,
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001,
    )

    training_loader = make_tensor_loader(
        training_features,
        training_labels,
        shuffle=True,
    )

    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []

    print("\nTraining MLP:")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        for batch_features, batch_labels in training_loader:
            optimizer.zero_grad(set_to_none=True)

            logits = model(batch_features)
            loss = loss_function(logits, batch_labels)

            loss.backward()
            optimizer.step()

        (
            training_loss,
            training_accuracy,
            _,
        ) = evaluate_mlp(
            model,
            training_features,
            training_labels,
            loss_function,
        )

        (
            validation_loss,
            validation_accuracy,
            _,
        ) = evaluate_mlp(
            model,
            validation_features,
            validation_labels,
            loss_function,
        )

        history.append(
            {
                "epoch": epoch,
                "training_loss": float(training_loss),
                "validation_loss": float(validation_loss),
                "training_accuracy": float(training_accuracy),
                "validation_accuracy": float(
                    validation_accuracy
                ),
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"  Epoch {epoch:3d} | "
                f"train loss {training_loss:.4f} | "
                f"val loss {validation_loss:.4f} | "
                f"train acc {training_accuracy:.4f} | "
                f"val acc {validation_accuracy:.4f}"
            )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                f"  Early stopping after epoch {epoch}. "
                f"Best epoch: {best_epoch}."
            )
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a model state.")

    model.load_state_dict(best_state)

    (
        final_training_loss,
        final_training_accuracy,
        training_predictions,
    ) = evaluate_mlp(
        model,
        training_features,
        training_labels,
        loss_function,
    )

    (
        final_validation_loss,
        final_validation_accuracy,
        validation_predictions,
    ) = evaluate_mlp(
        model,
        validation_features,
        validation_labels,
        loss_function,
    )

    (
        final_test_loss,
        final_test_accuracy,
        test_predictions,
    ) = evaluate_mlp(
        model,
        test_features,
        test_labels,
        loss_function,
    )

    training_results = summarize_predictions(
        training_labels,
        training_predictions,
    )
    validation_results = summarize_predictions(
        validation_labels,
        validation_predictions,
    )
    test_results = summarize_predictions(
        test_labels,
        test_predictions,
    )

    print("\nFinal MLP results:")
    print(f"  Best epoch:         {best_epoch}")
    print(f"  Training accuracy:  {final_training_accuracy:.4f}")
    print(f"  Validation accuracy:{final_validation_accuracy:.4f}")
    print(f"  Test accuracy:      {final_test_accuracy:.4f}")
    print(f"  Test macro F1:      {test_results['macro_f1']:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": len(feature_names),
            "hidden_dims": list(HIDDEN_DIMS),
            "num_classes": len(GROUP_ORDER),
            "dropout": DROPOUT,
            "class_names": GROUP_ORDER,
            "best_epoch": best_epoch,
            "random_seed": RANDOM_SEED,
        },
        MODEL_FILE,
    )

    preprocessing_payload = {
        "feature_names": feature_names,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "class_names": GROUP_ORDER,
        "label_to_index": {
            label: index
            for index, label in INDEX_TO_LABEL.items()
        },
    }

    with PREPROCESSING_FILE.open(
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(preprocessing_payload, output, indent=2)

    metrics_payload = {
        "model": {
            "name": "CorticalMLP",
            "task": (
                "Three-class cognitive-trajectory classification "
                "from 68 regional cortical-thickness measurements"
            ),
            "input_features": len(feature_names),
            "hidden_layers": list(HIDDEN_DIMS),
            "output_classes": GROUP_ORDER,
            "activation": "ReLU",
            "dropout": DROPOUT,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "loss_function": "CrossEntropyLoss",
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        },
        "cohort": cohort_metadata,
        "split": {
            "random_seed": RANDOM_SEED,
            "training_observations": int(
                len(training_labels)
            ),
            "validation_observations": int(
                len(validation_labels)
            ),
            "test_observations": int(len(test_labels)),
            "stratified": True,
        },
        "majority_baseline": majority_results,
        "logistic_regression_benchmark": benchmark_results,
        "mlp": {
            "best_epoch": best_epoch,
            "training_loss": float(final_training_loss),
            "validation_loss": float(final_validation_loss),
            "test_loss": float(final_test_loss),
            "training": training_results,
            "validation": validation_results,
            "test": test_results,
        },
        "history": history,
        "privacy": (
            "This file contains aggregate model-performance "
            "statistics only and no participant-level records."
        ),
        "disclaimer": (
            "Research and educational prototype only. "
            "Predictions are not medical diagnoses."
        ),
    }

    with METRICS_FILE.open("w", encoding="utf-8") as output:
        json.dump(metrics_payload, output, indent=2)

    demo_profiles = []

    for class_index, class_name in enumerate(GROUP_ORDER):
        class_profile = training_features_raw[
            training_labels == class_index
        ].mean(axis=0)

        demo_profiles.append(
            {
                "id": f"{class_name.lower()}_prototype",
                "display_name": (
                    f"{class_name.replace('_', ' ')} "
                    "aggregate prototype"
                ),
                "source": (
                    "Synthetic demonstration profile constructed "
                    "from the training-set class mean."
                ),
                "expected_pattern": class_name,
                "features": {
                    feature_name: round(float(value), 4)
                    for feature_name, value in zip(
                        feature_names,
                        class_profile,
                    )
                },
            }
        )

    with DEMO_PROFILES_FILE.open(
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(
            {
                "profiles": demo_profiles,
                "privacy": (
                    "These are aggregate class-mean demonstration "
                    "profiles, not participant records."
                ),
            },
            output,
            indent=2,
        )

    save_training_curves(history)

    print("\nSaved:")
    print(f"  Model weights:       {MODEL_FILE}")
    print(f"  Preprocessing:       {PREPROCESSING_FILE}")
    print(f"  Performance metrics: {METRICS_FILE}")
    print(f"  Demo profiles:       {DEMO_PROFILES_FILE}")
    print(f"  Training curves:     {CURVES_FILE}")


if __name__ == "__main__":
    main()