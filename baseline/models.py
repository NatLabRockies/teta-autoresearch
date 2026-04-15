"""Model training implementations for all supported families.

``run_trial(config, df, budget_seconds)`` is the single entry-point.
It dispatches to the appropriate trainer based on ``config["family"]``.
"""

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).parent.parent))
from fixed_utils import evaluate  # noqa: E402

from feature_pipeline import (
    TARGET,
    build_sequences,
    sequential_split,
    tabular_split,
)


# ---------------------------------------------------------------------------
# Tabular models
# ---------------------------------------------------------------------------


def train_tabular(
    config: dict,
    train_df: Any,
    test_df: Any,
    budget_seconds: float,
) -> float:
    """Train a sklearn / XGBoost / LightGBM model and return RMSE."""
    family = config["family"]
    features: list[str] = config["features"]
    model_params: dict = config["model_params"]

    X_train = train_df[features].to_numpy(dtype=np.float64)
    y_train = train_df[TARGET].to_numpy(dtype=np.float64)
    X_test = test_df[features].to_numpy(dtype=np.float64)
    y_test = test_df[TARGET].to_numpy(dtype=np.float64)

    if family == "mlp":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if family == "rf":
        from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]
        model = RandomForestRegressor(**model_params)
    elif family == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor  # type: ignore[import-untyped]
        model = ExtraTreesRegressor(**model_params)
    elif family == "hgbr":
        from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]
        model = HistGradientBoostingRegressor(**model_params)
    elif family == "xgb":
        from xgboost import XGBRegressor  # type: ignore[import-untyped]
        model = XGBRegressor(**model_params)
    elif family == "lgbm":
        from lightgbm import LGBMRegressor  # type: ignore[import-untyped]
        model = LGBMRegressor(**model_params)
    elif family == "mlp":
        from sklearn.neural_network import MLPRegressor  # type: ignore[import-untyped]
        model = MLPRegressor(**model_params)
    else:
        raise ValueError(f"Unknown tabular family: {family}")

    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    if elapsed > budget_seconds:
        print(f"  [warn] training took {elapsed:.0f}s, exceeded budget {budget_seconds:.0f}s")

    preds = model.predict(X_test)
    return evaluate(y_test, preds)["rmse"]


# ---------------------------------------------------------------------------
# Helpers shared by CNN and GRU
# ---------------------------------------------------------------------------


def _get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _scale_sequences(
    X_seq_tr: np.ndarray,
    X_static_tr: np.ndarray,
    X_seq_te: np.ndarray,
    X_static_te: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit scalers on train, transform both splits."""
    N_tr, T, F = X_seq_tr.shape
    N_te = X_seq_te.shape[0]

    scaler_seq = StandardScaler()
    X_seq_tr = scaler_seq.fit_transform(X_seq_tr.reshape(-1, F)).reshape(N_tr, T, F)
    X_seq_te = scaler_seq.transform(X_seq_te.reshape(-1, F)).reshape(N_te, T, F)

    if X_static_tr.shape[1] > 0:
        scaler_static = StandardScaler()
        X_static_tr = scaler_static.fit_transform(X_static_tr)
        X_static_te = scaler_static.transform(X_static_te)

    return X_seq_tr, X_static_tr, X_seq_te, X_static_te


def _make_loader(X_seq, X_static, y, batch_size: int, shuffle: bool, device):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    pin = device.type == "cuda"
    ds = TensorDataset(
        torch.tensor(X_seq, dtype=torch.float32),
        torch.tensor(X_static, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=pin)


def _run_nn_eval(model, X_seq_te, X_static_te, y_te, device) -> float:
    import torch

    model.eval()
    with torch.no_grad():
        x_s = torch.tensor(X_seq_te, dtype=torch.float32).to(device)
        x_st = torch.tensor(X_static_te, dtype=torch.float32).to(device)
        preds = model(x_s, x_st).cpu().numpy()
    return evaluate(y_te, preds)["rmse"]


# ---------------------------------------------------------------------------
# 1-D CNN
# ---------------------------------------------------------------------------


def train_cnn(
    config: dict,
    train_df: Any,
    test_df: Any,
    budget_seconds: float,
) -> float:
    """Train a 1-D convolutional network and return RMSE."""
    import torch
    import torch.nn as nn

    params = config["model_params"]
    seq_len: int = params["seq_len"]
    seq_feats: list[str] = config["seq_features"]
    static_feats: list[str] = config["static_features"]

    X_seq_tr, X_static_tr, y_tr = build_sequences(train_df, seq_feats, static_feats, seq_len)
    X_seq_te, X_static_te, y_te = build_sequences(test_df, seq_feats, static_feats, seq_len)
    X_seq_tr, X_static_tr, X_seq_te, X_static_te = _scale_sequences(
        X_seq_tr, X_static_tr, X_seq_te, X_static_te
    )

    n_seq_feats = len(seq_feats)
    n_static = len(static_feats)
    channels: int = params["channels"]
    n_layers: int = params["n_layers"]
    kernel_size: int = params["kernel_size"]
    dropout: float = params["dropout"]

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            conv_layers = []
            in_ch = n_seq_feats
            for _ in range(n_layers):
                conv_layers += [
                    nn.Conv1d(in_ch, channels, kernel_size, padding=kernel_size // 2),
                    nn.ReLU(),
                ]
                in_ch = channels
            self.conv = nn.Sequential(*conv_layers)
            head_in = channels * seq_len + n_static
            self.head = nn.Sequential(
                nn.Linear(head_in, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

        def forward(self, x_seq, x_static):
            x = self.conv(x_seq.permute(0, 2, 1)).flatten(1)
            if n_static > 0:
                x = torch.cat([x, x_static], dim=1)
            return self.head(x).squeeze(-1)

    device = _get_device()
    model = CNN().to(device)
    loader = _make_loader(X_seq_tr, X_static_tr, y_tr, params["batch_size"], True, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    # CosineAnnealingLR with a large T_max is safe for any training duration
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10_000, eta_min=params["lr"] * 0.01
    )
    criterion = nn.MSELoss()
    deadline = time.time() + budget_seconds

    while time.time() < deadline:
        model.train()
        for x_s, x_st, y_b in loader:
            if time.time() >= deadline:
                break
            x_s = x_s.to(device, non_blocking=True)
            x_st = x_st.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(x_s, x_st), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), params["grad_clip"])
            optimizer.step()
            scheduler.step()

    return _run_nn_eval(model, X_seq_te, X_static_te, y_te, device)


# ---------------------------------------------------------------------------
# GRU
# ---------------------------------------------------------------------------


def train_gru(
    config: dict,
    train_df: Any,
    test_df: Any,
    budget_seconds: float,
) -> float:
    """Train a GRU sequence model and return RMSE."""
    import torch
    import torch.nn as nn

    params = config["model_params"]
    seq_len: int = params["seq_len"]
    seq_feats: list[str] = config["seq_features"]
    static_feats: list[str] = config["static_features"]

    X_seq_tr, X_static_tr, y_tr = build_sequences(train_df, seq_feats, static_feats, seq_len)
    X_seq_te, X_static_te, y_te = build_sequences(test_df, seq_feats, static_feats, seq_len)
    X_seq_tr, X_static_tr, X_seq_te, X_static_te = _scale_sequences(
        X_seq_tr, X_static_tr, X_seq_te, X_static_te
    )

    n_seq_feats = len(seq_feats)
    n_static = len(static_feats)
    hidden_size: int = params["hidden_size"]
    n_layers: int = params["n_layers"]
    # nn.GRU requires dropout=0 when n_layers == 1
    gru_dropout = params["dropout"] if n_layers > 1 else 0.0

    class GRUModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(
                n_seq_feats, hidden_size, n_layers,
                batch_first=True, dropout=gru_dropout,
            )
            head_in = hidden_size + n_static
            self.head = nn.Sequential(
                nn.Linear(head_in, 128),
                nn.ReLU(),
                nn.Dropout(params["dropout"]),
                nn.Linear(128, 1),
            )

        def forward(self, x_seq, x_static):
            _, h = self.gru(x_seq)
            x = h[-1]  # last layer's final hidden state
            if n_static > 0:
                x = torch.cat([x, x_static], dim=1)
            return self.head(x).squeeze(-1)

    device = _get_device()
    model = GRUModel().to(device)
    loader = _make_loader(X_seq_tr, X_static_tr, y_tr, params["batch_size"], True, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10_000, eta_min=params["lr"] * 0.01
    )
    criterion = nn.MSELoss()
    deadline = time.time() + budget_seconds

    while time.time() < deadline:
        model.train()
        for x_s, x_st, y_b in loader:
            if time.time() >= deadline:
                break
            x_s = x_s.to(device, non_blocking=True)
            x_st = x_st.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(x_s, x_st), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), params["grad_clip"])
            optimizer.step()
            scheduler.step()

    return _run_nn_eval(model, X_seq_te, X_static_te, y_te, device)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def run_trial(config: dict, df: Any, budget_seconds: float) -> float:
    """Train and evaluate one config; return RMSE."""
    family = config["family"]
    if family in ("cnn", "gru"):
        train_df, test_df = sequential_split(df)
        if family == "cnn":
            return train_cnn(config, train_df, test_df, budget_seconds)
        return train_gru(config, train_df, test_df, budget_seconds)
    else:
        train_df, test_df = tabular_split(df)
        return train_tabular(config, train_df, test_df, budget_seconds)
