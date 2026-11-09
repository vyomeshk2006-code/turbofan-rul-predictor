import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error
from features import create_features


# Load the saved Random Forest and its expected feature order
saved = joblib.load("turbofan_rul_predictor.joblib")

model = saved["model"]
feature_columns = saved["feature_columns"]


# Load the official test readings
test_df = pd.read_csv("data/test_FD001.txt", sep=r"\s+", header=None)

columns = (["Engine_ID", "Cycle"] + ["Setting_1", "Setting_2", "Setting_3"] + [f"Sensor_{i}" for i in range(1, 22)])

test_df.columns = columns


# Calculate changes, slopes, and standard deviations
# This function also sorts the rows by engine and cycle
test_df = create_features(test_df)


# Load actual RUL at each test engine's last recorded cycle
test_rul = pd.read_csv("data/RUL_FD001.txt", sep=r"\s+",header=None)

actual_test_rul = test_rul.iloc[:, 0].to_numpy()


# Select each engine's last recorded row
test_last_rows = (test_df.groupby("Engine_ID").tail(1).sort_values("Engine_ID"))


# Verify that engine IDs match the answer-file order
expected_engine_ids = np.arange(1, len(actual_test_rul) + 1)

if not np.array_equal(
    test_last_rows["Engine_ID"].to_numpy(),
    expected_engine_ids
):
    raise ValueError("Test engine IDs do not match the RUL answer-file order.")


# Select the model's 88 features in their saved order
X_test = test_last_rows[feature_columns]


# Predict using the saved forest
test_predictions = model.predict(X_test)


# Calculate errors across all test engines
test_mae = mean_absolute_error(actual_test_rul, test_predictions)

test_mse = mean_squared_error(actual_test_rul, test_predictions)

test_rmse = np.sqrt(test_mse)


# Display results
print("Test input shape:", X_test.shape)

print("\nFirst five predictions:")
print(test_predictions[:5])

print("First five actual RUL values:")
print(actual_test_rul[:5])

print("\n--- Official Test: Engine Endpoints ---")
print("Test engines:", len(test_predictions))
print("Test MAE:", test_mae)
print("Test MSE:", test_mse)
print("Test RMSE:", test_rmse)