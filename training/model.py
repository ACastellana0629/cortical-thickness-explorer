"""PyTorch model used to classify cortical-thickness profiles."""

import torch
from torch import nn


class CorticalMLP(nn.Module):
    """Small multilayer perceptron for three-class classification."""

    def __init__(
        self,
        input_dim=68,
        hidden_dims=(32, 16),
        num_classes=3,
        dropout=0.20,
    ):
        super().__init__()

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, num_classes))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize linear-layer parameters before training."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, features):
        """Perform forward propagation and return class logits."""
        return self.network(features)


if __name__ == "__main__":
    model = CorticalMLP()
    example_batch = torch.randn(4, 68)
    output = model(example_batch)

    print(model)
    print(f"\nExample input shape: {example_batch.shape}")
    print(f"Example output shape: {output.shape}")
    print(
        "Trainable parameters:",
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )