import time
import math
import pandas as pd
import numpy as np
from shapely import wkb  # type: ignore[import-untyped]
from concurrent.futures import ProcessPoolExecutor, TimeoutError

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from fixed_utils import (
    evaluate,
    train_test_split,
)

# --- shared defaults ---
TIME_BUDGET_SECONDS = 10 * 60

# Per-link features that each timestep in the sequence has
LINK_FEATURES = [
    "speed_mph",
    "grade_percent",
    "miles",
    "time_seconds",
    "sinuosity",
    "abs_bearing_delta",
]

# Static features (not part of the sequence, concatenated after conv)
STATIC_FEATURES: list[str] = []

SEQ_LEN = 5  # current link + 4 previous
TARGET = "energy_rate_gge"

# --- data config ---
# Registry of powertrain types. Each session targets exactly one.
POWERTRAINS = {
    "bev": {
        "name": "2017_Chevy_Bolt",
        "data_path": "data/processed/2017_Chevy_Bolt.parquet",
        "energy_type": "bev",
    },
    "ice": {
        "name": "2016_Toyota_Camry",
        "data_path": "data/processed/2016_Toyota_Camry.parquet",
        "energy_type": "ice",
    },
    "phev": {
        # Filled in when PHEV raw data lands in data/processed/.
        "name": "TBD_PHEV",
        "data_path": "data/processed/TBD_PHEV.parquet",
        "energy_type": "phev",
    },
}

POWERTRAIN = "bev"  # session selector — the only vehicle-related line to change
CONFIG = POWERTRAINS[POWERTRAIN]


class Conv1DModel(nn.Module):
    def __init__(self, n_link_features, n_static_features, seq_len):
        super().__init__()
        # Conv1d: (batch, channels=n_link_features, length=seq_len)
        self.conv = nn.Sequential(
            nn.Conv1d(n_link_features, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # After conv: (batch, 128, seq_len) -> flatten -> 128*seq_len
        conv_out_dim = 128 * seq_len
        self.head = nn.Sequential(
            nn.Linear(conv_out_dim + n_static_features, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x_seq, x_static):
        # x_seq: (batch, n_link_features, seq_len)
        # x_static: (batch, n_static_features)
        h = self.conv(x_seq)
        h = h.flatten(1)  # (batch, 64*seq_len)
        h = torch.cat([h, x_static], dim=1)
        return self.head(h).squeeze(-1)


def train_model() -> dict:
    """Train and evaluate. Returns results dict."""
    t0 = time.time()

    # data
    df = pd.read_parquet(CONFIG["data_path"])

    # sort by journey and time
    df = df.sort_values(["journey_id", "link_start_time"])

    # geometry: extract sinuosity and bearing in single pass
    def calc_geom_features(geom_hex):
        g = wkb.loads(geom_hex, hex=True)
        coords = list(g.coords)
        start, end = coords[0], coords[-1]
        dx, dy = end[0] - start[0], end[1] - start[1]
        straight = math.sqrt(dx**2 + dy**2)
        if straight < 1e-10:
            sinuosity = 1.0
        else:
            road_len = sum(
                math.sqrt(
                    (coords[i + 1][0] - coords[i][0]) ** 2
                    + (coords[i + 1][1] - coords[i][1]) ** 2
                )
                for i in range(len(coords) - 1)
            )
            sinuosity = road_len / straight
        bearing = math.atan2(dx, dy) * 180 / math.pi
        return sinuosity, bearing

    geom_feats = df["geometry"].apply(calc_geom_features)
    df["sinuosity"] = geom_feats.apply(lambda x: x[0])
    df["bearing"] = geom_feats.apply(lambda x: x[1])
    prev_bearing = df.groupby("journey_id")["bearing"].shift(1)
    raw_delta = df["bearing"] - prev_bearing
    df["bearing_delta"] = (raw_delta + 180) % 360 - 180
    df["abs_bearing_delta"] = df["bearing_delta"].abs()

    # Build sequence windows: for each link, get current + 4 previous links' features
    # Shift link features within each journey
    for feat in LINK_FEATURES:
        for lag in range(1, SEQ_LEN):
            col_name = f"{feat}_lag{lag}"
            df[col_name] = df.groupby("journey_id")[feat].shift(lag)

    # Drop rows without full sequence history
    lag_cols = [f"{feat}_lag{SEQ_LEN - 1}" for feat in LINK_FEATURES]
    df = df.dropna(subset=lag_cols + ["bearing_delta"])

    train_df, test_df = train_test_split(df, test_size=0.2, random_seed=42)

    # Build sequence arrays: (N, n_link_features, seq_len)
    # seq_len ordering: [lag4, lag3, lag2, lag1, current] (oldest to newest)
    def build_sequences(data):
        n = len(data)
        n_feat = len(LINK_FEATURES)
        seq = np.zeros((n, n_feat, SEQ_LEN), dtype=np.float32)
        for i, feat in enumerate(LINK_FEATURES):
            # oldest to newest
            for lag in range(SEQ_LEN - 1, 0, -1):
                col = f"{feat}_lag{lag}"
                seq[:, i, SEQ_LEN - 1 - lag] = data[col].values
            seq[:, i, SEQ_LEN - 1] = data[feat].values
        return seq

    train_seq = build_sequences(train_df)
    test_seq = build_sequences(test_df)

    train_static = train_df[STATIC_FEATURES].values.astype(np.float32)
    test_static = test_df[STATIC_FEATURES].values.astype(np.float32)

    y_train = train_df[TARGET].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET].to_numpy(dtype=np.float32)

    # Normalize: fit on train, transform both
    # Normalize sequence features per-feature across all timesteps
    n_feat = len(LINK_FEATURES)
    seq_scaler = StandardScaler()
    # Reshape to (N*seq_len, n_feat) for fitting
    train_seq_flat = train_seq.transpose(0, 2, 1).reshape(-1, n_feat)
    seq_scaler.fit(train_seq_flat)
    # Transform
    train_seq_flat = seq_scaler.transform(train_seq_flat)
    train_seq = (
        train_seq_flat.reshape(-1, SEQ_LEN, n_feat)
        .transpose(0, 2, 1)
        .astype(np.float32)
    )
    test_seq_flat = test_seq.transpose(0, 2, 1).reshape(-1, n_feat)
    test_seq_flat = seq_scaler.transform(test_seq_flat)
    test_seq = (
        test_seq_flat.reshape(-1, SEQ_LEN, n_feat).transpose(0, 2, 1).astype(np.float32)
    )

    if STATIC_FEATURES:
        static_scaler = StandardScaler()
        train_static = static_scaler.fit_transform(train_static).astype(np.float32)
        test_static = static_scaler.transform(test_static).astype(np.float32)

    # PyTorch setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TensorDataset(
        torch.from_numpy(train_seq),
        torch.from_numpy(train_static),
        torch.from_numpy(y_train),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=2048, shuffle=True, num_workers=0
    )

    model = Conv1DModel(n_feat, len(STATIC_FEATURES), SEQ_LEN).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-3,
        steps_per_epoch=steps_per_epoch,
        epochs=15,
    )
    loss_fn = nn.MSELoss()

    # Train until time budget
    train_end = t0 + TIME_BUDGET_SECONDS
    epoch = 0
    while time.time() < train_end:
        model.train()
        for batch_seq, batch_static, batch_y in train_loader:
            if time.time() >= train_end:
                break
            batch_seq = batch_seq.to(device)
            batch_static = batch_static.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_seq, batch_static)
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        epoch += 1

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_seq_t = torch.from_numpy(test_seq).to(device)
        test_static_t = torch.from_numpy(test_static).to(device)
        predicted = model(test_seq_t, test_static_t).cpu().numpy()

    results = evaluate(y_test, predicted)

    features_str = ",".join(LINK_FEATURES + STATIC_FEATURES)
    for k, v in results.items():
        print(f"{k}: {v:.6f}")
    total_seconds = time.time() - t0
    print(f"total_seconds: {total_seconds:.1f}")
    print(f"features: {features_str}")
    print(f"epochs: {epoch}")

    return results


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(train_model)
        try:
            future.result(timeout=TIME_BUDGET_SECONDS)
        except TimeoutError:
            print(
                f"\n{CONFIG['name']}: timed out after {TIME_BUDGET_SECONDS}s, skipping"
            )
            executor.shutdown(wait=False, cancel_futures=True)
