import numpy as np


def calculate_slope(readings):
    if len(readings) < 2:
        return 0.0

    positions = np.arange(len(readings))

    centered_positions = positions - positions.mean()
    centered_readings = readings - readings.mean()

    return (np.sum(centered_positions * centered_readings) / np.sum(centered_positions ** 2))


def create_features(data):
    feature_df = data.sort_values(["Engine_ID", "Cycle"]).copy()

    sensor_columns = [f"Sensor_{i}" for i in range(1, 22)]

    for sensor in sensor_columns:
        change_column = sensor + "_Change"
        slope_column = sensor + "_Slope_5"
        std_column = sensor + "_Std_5"

        grouped_sensor = feature_df.groupby("Engine_ID")[sensor]

        feature_df[change_column] = (grouped_sensor.diff().fillna(0))

        feature_df[slope_column] = grouped_sensor.transform(
            lambda readings: readings.rolling(window=5, min_periods=1).apply(calculate_slope, raw=True)
            )

        feature_df[std_column] = grouped_sensor.transform(
            lambda readings: readings.rolling(window=5, min_periods=1).std(ddof=0)
        )

    return feature_df