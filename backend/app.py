"""FastAPI prediction service for the cortical-thickness MLP."""

import json
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from training.model import CorticalMLP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = PROJECT_ROOT / "models" / "cortical_mlp.pt"
PREPROCESSING_FILE = (
    PROJECT_ROOT / "models" / "preprocessing.json"
)

with PREPROCESSING_FILE.open("r", encoding="utf-8") as file:
    preprocessing = json.load(file)

checkpoint = torch.load(
    MODEL_FILE,
    map_location="cpu",
    weights_only=True,
)

feature_names = preprocessing["feature_names"]
feature_mean = np.asarray(
    preprocessing["mean"],
    dtype=np.float32,
)
feature_scale = np.asarray(
    preprocessing["scale"],
    dtype=np.float32,
)
class_names = preprocessing["class_names"]

model = CorticalMLP(
    input_dim=checkpoint["input_dim"],
    hidden_dims=tuple(checkpoint["hidden_dims"]),
    num_classes=checkpoint["num_classes"],
    dropout=checkpoint["dropout"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


class PredictionRequest(BaseModel):
    """A complete set of regional cortical-thickness values."""

    features: dict[str, float] = Field(
        description=(
            "Mapping from each required cortical-region feature "
            "to its thickness in millimeters."
        )
    )


app = FastAPI(
    title="Cortical Explorer Prediction API",
    description=(
        "Research prototype serving predictions from a PyTorch "
        "cortical-thickness classifier."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Describe the API."""
    return {
        "name": "Cortical Explorer Prediction API",
        "status": "running",
        "documentation": "/docs",
    }


@app.get("/health")
def health():
    """Confirm that the API and trained model are available."""
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": "CorticalMLP",
        "input_features": len(feature_names),
        "output_classes": class_names,
    }


@app.get("/model-info")
def model_info():
    """Return non-sensitive information about the trained model."""
    return {
        "model_name": "CorticalMLP",
        "input_features": len(feature_names),
        "hidden_layers": checkpoint["hidden_dims"],
        "output_classes": class_names,
        "activation": "ReLU",
        "dropout": checkpoint["dropout"],
        "best_epoch": checkpoint["best_epoch"],
        "random_seed": checkpoint["random_seed"],
        "disclaimer": (
            "Research and educational prototype only. "
            "Predictions are not medical diagnoses."
        ),
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    """Validate a cortical profile and return class probabilities."""
    supplied_features = request.features

    missing_features = [
        name for name in feature_names
        if name not in supplied_features
    ]
    unexpected_features = [
        name for name in supplied_features
        if name not in feature_names
    ]

    if missing_features or unexpected_features:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "The submitted cortical profile does not match "
                    "the model's required features."
                ),
                "missing_features": missing_features,
                "unexpected_features": unexpected_features,
            },
        )

    values = np.asarray(
        [supplied_features[name] for name in feature_names],
        dtype=np.float32,
    )

    if not np.isfinite(values).all():
        raise HTTPException(
            status_code=422,
            detail="Every cortical-thickness value must be finite.",
        )

    if np.any(values <= 0) or np.any(values > 6):
        raise HTTPException(
            status_code=422,
            detail=(
                "Cortical-thickness values must be greater than "
                "0 and no greater than 6 millimeters."
            ),
        )

    standardized_values = (
        values - feature_mean
    ) / feature_scale

    input_tensor = torch.tensor(
        standardized_values,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(
            logits,
            dim=1,
        ).squeeze(0).cpu().numpy()

    predicted_index = int(probabilities.argmax())
    predicted_class = class_names[predicted_index]

    probability_mapping = {
        class_name: round(float(probability), 6)
        for class_name, probability in zip(
            class_names,
            probabilities,
        )
    }

    return {
        "predicted_class": predicted_class,
        "probabilities": probability_mapping,
        "model": "CorticalMLP",
        "feature_count": len(feature_names),
        "disclaimer": (
            "Research and educational prototype only. "
            "This prediction is not a medical diagnosis."
        ),
    }