import torch
import pandas as pd
import numpy as np

from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# Load the dataset
# ============================================================

df = pd.read_csv("data/train_FD001.txt", sep=r"\s+", header=None)

columns = (
    ["Engine_ID", "Cycle"]
    + ["Setting_1", "Setting_2", "Setting_3"]
    + [f"Sensor_{i}" for i in range(1, 22)]
)

df.columns = columns

# ============================================================
# Calculate actual RUL and sort each engine's cycles
# ============================================================

max_cycles = df.groupby("Engine_ID")["Cycle"].transform("max")
df["RUL"] = max_cycles - df["Cycle"]

df = df.sort_values(["Engine_ID", "Cycle"]).copy()

print("Dataset shape:", df.shape)
print(df[["Engine_ID", "Cycle", "Sensor_2", "RUL"]].head())

# ============================================================
# Original 80/20 engine split
# The original 20 validation engines remain outside CV
# ============================================================

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42
)

train_idx, val_idx = next(splitter.split(df, groups=df["Engine_ID"]))

train_df = df.iloc[train_idx].copy()
val_df = df.iloc[val_idx].copy()

print("Training engines:", train_df["Engine_ID"].nunique())
print("Validation engines:", val_df["Engine_ID"].nunique())

# ============================================================
# Create causal windows separately within each engine
# ============================================================

def create_windows(data, feature_columns, window_size):
    windows = []
    targets = []

    for engine_id, engine_data in data.groupby("Engine_ID"):
        engine_features = engine_data[feature_columns].to_numpy(
            dtype=np.float32
        )

        engine_targets = engine_data["RUL"].to_numpy(dtype=np.float32)

        for end in range(len(engine_data)):
            start = max(0, end - window_size + 1)

            window = engine_features[start:end + 1]

            # Early cycles have fewer than 20 available rows.
            # Repeat the engine's first row to fill the window.

            if len(window) < window_size:
                missing_rows = window_size - len(window)

                padding = np.repeat(engine_features[:1], missing_rows, axis=0)

                window = np.concatenate([padding, window], axis=0)

            windows.append(window)

            # Target: actual RUL at the window's last cycle
            targets.append(engine_targets[end])

    return (
        torch.tensor(
            np.stack(windows),
            dtype=torch.float32
        ),
        torch.tensor(
            np.array(targets),
            dtype=torch.float32
        )
    )

# ============================================================
# Define inputs and training settings
# ============================================================

sequence_feature_columns = (
    ["Setting_1", "Setting_2", "Setting_3"]
    + [f"Sensor_{i}" for i in range(1, 22)]
    + ["Cycle"]
)

window_size = 20
epochs = 70

group_cv = GroupKFold(n_splits=5)
cv_groups = train_df["Engine_ID"]

gru_cv_mae_scores = []
gru_cv_mse_scores = []
gru_cv_rmse_scores = []

# ============================================================
# Train and evaluate a fresh GRU for each fold
# ============================================================

for fold, (cv_train_idx, cv_val_idx) in enumerate(group_cv.split(train_df, groups=cv_groups), start=1):
    print("\n--- Starting GRU Fold:", fold, "---")

    # Select this fold's training and evaluation engines
    fold_train_df = train_df.iloc[cv_train_idx].copy()
    fold_val_df = train_df.iloc[cv_val_idx].copy()

    print("Training engines:", fold_train_df["Engine_ID"].nunique())

    print("Evaluation engines:", fold_val_df["Engine_ID"].nunique())

    print(
        "Engines in both groups:",
        set(fold_train_df["Engine_ID"])
        & set(fold_val_df["Engine_ID"])
    )

    # --------------------------------------------------------
    # Fit scaling using only this fold's training rows
    # --------------------------------------------------------

    sequence_scaler = StandardScaler()

    fold_train_scaled = fold_train_df.copy()
    fold_val_scaled = fold_val_df.copy()

    fold_train_scaled[sequence_feature_columns] = (
        sequence_scaler.fit_transform(
            fold_train_df[sequence_feature_columns]
        )
    )

    fold_val_scaled[sequence_feature_columns] = (
        sequence_scaler.transform(
            fold_val_df[sequence_feature_columns]
        )
    )

    # --------------------------------------------------------
    # Create training and evaluation windows
    # --------------------------------------------------------

    X_train_windows, Y_train_windows = create_windows(
        fold_train_scaled,
        sequence_feature_columns,
        window_size
    )

    X_val_windows, Y_val_windows = create_windows(
        fold_val_scaled,
        sequence_feature_columns,
        window_size
    )

    print("Training windows:", X_train_windows.shape)
    print("Training targets:", Y_train_windows.shape)

    print("Evaluation windows:", X_val_windows.shape)
    print("Evaluation targets:", Y_val_windows.shape)

    # --------------------------------------------------------
    # Pair windows with their targets and create batch loaders
    # --------------------------------------------------------

    train_dataset = TensorDataset(X_train_windows, Y_train_windows)

    val_dataset = TensorDataset(X_val_windows, Y_val_windows)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    print("Training batches:", len(train_loader))
    print("Evaluation batches:", len(val_loader))

    # --------------------------------------------------------
    # Create fresh layers for this fold
    # --------------------------------------------------------

    torch.manual_seed(42)

    gru_layer = nn.GRU(
        input_size=len(sequence_feature_columns),
        hidden_size=32,
        batch_first=True
    )

    output_layer = nn.Linear(in_features=32, out_features=1)

    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(
        list(gru_layer.parameters())
        + list(output_layer.parameters()),
        lr=0.001
    )

    # --------------------------------------------------------
    # Train for 70 epochs
    # --------------------------------------------------------

    gru_layer.train()
    output_layer.train()

    for epoch in range(epochs):
        total_squared_error = 0.0
        total_windows = 0

        for batch_inputs, batch_targets in train_loader:
            # Clear previous gradients, not learned parameters
            optimizer.zero_grad()

            # Process each window using a fresh hidden summary
            sequence_outputs, final_hidden = gru_layer(batch_inputs)

            last_hidden = final_hidden[-1]

            # Predict RUL for each window's last cycle
            batch_predictions = output_layer(last_hidden).squeeze(-1)

            # Calculate loss and gradients
            batch_loss = loss_function(batch_predictions, batch_targets)

            batch_loss.backward()

            # Update both layers' parameters once per batch
            optimizer.step()

            # Accumulate errors for this epoch
            batch_size = batch_targets.shape[0]

            total_squared_error += (batch_loss.item() * batch_size)

            total_windows += batch_size

        epoch_mse = total_squared_error / total_windows

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print("Fold:", fold, "Epoch:", epoch + 1, "/", epochs, "Training MSE:", epoch_mse)

    # --------------------------------------------------------
    # Save this fold's trained parameters
    # --------------------------------------------------------

    torch.save(
        {
            "gru": gru_layer.state_dict(),
            "output": output_layer.state_dict()
        },
        f"gru_cv_fold{fold}_parameters.pth"
    )

    print("Saved parameters for fold:", fold)

    # --------------------------------------------------------
    # Evaluate without updating weights or biases
    # --------------------------------------------------------

    gru_layer.eval()
    output_layer.eval()

    val_prediction_parts = []
    val_target_parts = []

    with torch.no_grad():
        for batch_inputs, batch_targets in val_loader:
            sequence_outputs, final_hidden = gru_layer(batch_inputs)

            last_hidden = final_hidden[-1]

            batch_predictions = output_layer(last_hidden).squeeze(-1)

            val_prediction_parts.append(batch_predictions)
            val_target_parts.append(batch_targets)

    # Combine all evaluation batches into two NumPy arrays
    gru_val_pred = torch.cat(val_prediction_parts).numpy()

    gru_val_actual = torch.cat(val_target_parts).numpy()

    # --------------------------------------------------------
    # Calculate this fold's evaluation scores
    # --------------------------------------------------------

    gru_val_mae = mean_absolute_error(gru_val_actual, gru_val_pred)

    gru_val_mse = mean_squared_error(gru_val_actual, gru_val_pred)

    gru_val_rmse = np.sqrt(gru_val_mse)

    gru_cv_mae_scores.append(gru_val_mae)
    gru_cv_mse_scores.append(gru_val_mse)
    gru_cv_rmse_scores.append(gru_val_rmse)

    print("\n--- GRU Fold:", fold, "Evaluation ---")
    print("Evaluation windows:", len(gru_val_pred))
    print("Evaluation MAE:", gru_val_mae)
    print("Evaluation MSE:", gru_val_mse)
    print("Evaluation RMSE:", gru_val_rmse)


# ============================================================
# Average the five fold scores
# ============================================================

print("\n--- GRU: Five-Fold CV ---")
print("Fold MAE scores:", gru_cv_mae_scores)
print("Average CV MAE:", np.mean(gru_cv_mae_scores))
print("Average CV MSE:", np.mean(gru_cv_mse_scores))
print("Average CV RMSE:", np.mean(gru_cv_rmse_scores))
