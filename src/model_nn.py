from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model_rf import build_model_features, prepare_demand_dataset
from src.utils import FIGURES_DIR, REPORTS_DIR


class DemandMLP(nn.Module):
    """Simple multilayer perceptron for tabular demand prediction."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_neural_network(df: pd.DataFrame) -> dict:
    """Train a PyTorch neural network for zone-hour demand prediction."""
    torch.manual_seed(42)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    demand_df = prepare_demand_dataset(df)
    features, target = build_model_features(demand_df)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=512, shuffle=True)

    model = DemandMLP(X_train_tensor.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses: list[float] = []
    epochs = 25

    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        for batch_features, batch_target in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_features)

        average_loss = epoch_loss / len(train_loader.dataset)
        losses.append(float(average_loss))

    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).squeeze(1).numpy()

    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(mean_squared_error(y_test, predictions) ** 0.5)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), losses, marker="o", color="#6a4c93")
    plt.title("Neural Network Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "nn_loss_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    result = {
        "model_name": "PyTorchMLP",
        "mae": mae,
        "rmse": rmse,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "feature_count": int(X_train.shape[1]),
        "epochs": epochs,
        "loss_curve_path": str(FIGURES_DIR / "nn_loss_curve.png"),
    }

    with open(REPORTS_DIR / "neural_network_metrics.json", "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    return result
