import pandas as pd
df=pd.read_csv("data/train_FD001.txt",sep=r"\s+",header=None)
columns=(
    ["Engine_ID","Cycle"]
    + ["Setting_1","Setting_2","Setting_3"]
    +[f"Sensor_{i}" for i in range(1,22)]
)
df.columns=columns

# print("Dataset shape",df.shape)
# print("Total Engines",df["Engine_ID"].nunique())
# print("Cycles for each engine")
# print(df.groupby("Engine_ID")["Cycle"].max())

max_cycles=df.groupby("Engine_ID")["Cycle"].transform("max")
df["RUL"]=max_cycles-df["Cycle"]
print(df[["Engine_ID","Cycle","RUL"]].head())

import matplotlib.pyplot as plt
# engine_1=df[df["Engine_ID"]==1]
# for i in range(1,22):
#     plt.plot(engine_1["Cycle"],engine_1[f"Sensor_{i}"])
#     plt.xlabel("Operating Cycle")
#     plt.ylabel(f"Sensor {i} Reading")
#     plt.title(f"Engine 1 - Sensor {i} Over Time")
#     plt.show()

sensor_columns=[f"Sensor_{i}" for i in range(1,22)]
unique_counts=df[sensor_columns].nunique()
print(unique_counts)

missing_values=df.isnull().sum()
print(missing_values)

feature_columns=(
    ["Setting_1","Setting_2","Setting_3"]
    + [f"Sensor_{i}" for i in range(1,22)]
)

X=df[feature_columns]
y=df["RUL"]
print("X shape",X.shape)
print("Y shape",y.shape)

from sklearn.model_selection import GroupShuffleSplit
splitter=GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)
train_idx,val_idx=next(splitter.split(X,y,groups=df["Engine_ID"]))
X_train=X.iloc[train_idx]
X_val=X.iloc[val_idx]
Y_train = y.iloc[train_idx]
Y_val = y.iloc[val_idx]

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)

from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
X_train_scaled=scalar.fit_transform(X_train)
X_val_scaled=scalar.transform(X_val)

print("Original training values:")
print(X_train.head())

print("Scaled training values:")
print(X_train_scaled[:5])

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train_scaled,Y_train)
linear_val_pred=model.predict(X_val_scaled)
linear_train_pred=model.predict(X_train_scaled)


print("First 10 predicted RUL values:")
print(linear_val_pred[:10])

print("First 10 actual RUL values:")
print(Y_val[:10].values)

from sklearn.metrics import mean_absolute_error
linear_val_mae=mean_absolute_error(Y_val,linear_val_pred)
linear_train_mae = mean_absolute_error(Y_train, linear_train_pred)
print("Linear Regression Training MAE:", linear_train_mae)
print("Linear Regression Validation MAE:", linear_val_mae)

from sklearn.metrics import mean_squared_error
import numpy as np
mse=mean_squared_error(Y_val,linear_val_pred)
linear_val_rmse=np.sqrt(mse)
print("Validation RMSE",linear_val_rmse)
linear_train_rmse = np.sqrt(mean_squared_error(Y_train, linear_train_pred))
print("Training RMSE",linear_train_rmse)

from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2,include_bias=False)
X_train_poly=poly.fit_transform(X_train_scaled)
X_val_poly=poly.transform(X_val_scaled)
print("Original shape:", X_train_scaled.shape)
print("Polynomial shape:", X_train_poly.shape)

from sklearn.linear_model import Ridge
poly_model=Ridge(alpha=10)
poly_model.fit(X_train_poly, Y_train)
val_poly_pred = poly_model.predict(X_val_poly)
train_poly_pred= poly_model.predict(X_train_poly)

val_poly_mae = mean_absolute_error(Y_val, val_poly_pred)
train_poly_mae= mean_absolute_error(Y_train,train_poly_pred)
val_poly_rmse = np.sqrt(mean_squared_error(Y_val, val_poly_pred))
train_poly_rmse= np.sqrt(mean_squared_error(Y_train,train_poly_pred))

print("\n--- Linear Regression ---")
print("Training MAE:", linear_train_mae)
print("Validation MAE:", linear_val_mae)
print("Training RMSE:", linear_train_rmse)
print("Validation RMSE:", linear_val_rmse)

print("\n--- Polynomial Ridge Regression ---")
print("Training MAE:", train_poly_mae)
print("Validation MAE:", val_poly_mae)
print("Training RMSE:", train_poly_rmse)
print("Validation RMSE:", val_poly_rmse)

print("\nTraining RUL distribution:")
print(Y_train.describe())
print("\nValidation RUL distribution:")
print(Y_val.describe())

ridge_100=Ridge(alpha=100)
ridge_100.fit(X_train_poly,Y_train)
ridge_100_train_pred=ridge_100.predict(X_train_poly)
ridge_100_val_pred=ridge_100.predict(X_val_poly)
print("Alpha 100 Training MAE:",
      mean_absolute_error(Y_train, ridge_100_train_pred))
print("Alpha 100 Validation MAE:",
      mean_absolute_error(Y_val, ridge_100_val_pred))

ridge_1000 = Ridge(alpha=1000)
ridge_1000.fit(X_train_poly, Y_train)
ridge_1000_train_pred = ridge_1000.predict(X_train_poly)
ridge_1000_val_pred = ridge_1000.predict(X_val_poly)
print("Alpha 1000 Training MAE:",
      mean_absolute_error(Y_train, ridge_1000_train_pred))
print("Alpha 1000 Validation MAE:",
      mean_absolute_error(Y_val, ridge_1000_val_pred))

ridge_10000 = Ridge(alpha=10000)
ridge_10000.fit(X_train_poly, Y_train)

ridge_10000_train_pred = ridge_10000.predict(X_train_poly)
ridge_10000_val_pred = ridge_10000.predict(X_val_poly)

print("Alpha 10000 Training MAE:",
      mean_absolute_error(Y_train, ridge_10000_train_pred))
print("Alpha 10000 Validation MAE:",
      mean_absolute_error(Y_val, ridge_10000_val_pred))

# implementing gradient descent
# m,n=X_train_scaled.shape
# w=np.zeros(n)
# b=0.0
# learning_rate=0.01
# iterations=1000
# for iteration in range(iterations):
#     train_predictions=X_train_scaled @ w +b
#     errors=train_predictions-Y_train.to_numpy()

#     dw=np.zeros(n)
#     db=0
#     for i in range(m):
#         for j in range(n):
#             dw[j]+= errors[i]* X_train_scaled[i,j]
#         db+=errors[i]
#     dw = dw / m
#     db = db / m
#     w=w-learning_rate * dw
#     b=b-learning_rate*db

#     train_predictions= X_train_scaled @ w +b
#     errors=train_predictions-Y_train.to_numpy()
#     cost=np.sum(errors**2)/(2*m)
#     if (iteration+1)%100==0:
#         print("Iteration",iteration+1, "Cost:",cost)

# np.savez("gd_parameters.npz", w=w, b=b)
with np.load("gd_parameters.npz") as saved:
    w = saved["w"]
    b = saved["b"].item()

gd_train_pred=X_train_scaled @ w +b
gd_val_pred=X_val_scaled @ w+b
gd_train_mae=mean_absolute_error(Y_train,gd_train_pred)
gd_val_mae=mean_absolute_error(Y_val,gd_val_pred)
gd_train_rmse=np.sqrt(mean_squared_error(Y_train,gd_train_pred))
gd_val_rmse=np.sqrt(mean_squared_error(Y_val,gd_val_pred))

print("\n--- Gradient Descent Linear Regression ---")
print("Training MAE:", gd_train_mae)
print("Validation MAE:", gd_val_mae)
print("Training RMSE:", gd_train_rmse)
print("Validation RMSE:", gd_val_rmse)

trend_df= df.sort_values(["Engine_ID","Cycle"]).copy()
trend_df["Sensor_2_Change"]=(trend_df.groupby("Engine_ID")["Sensor_2"].diff())
print(trend_df[["Engine_ID","Cycle","Sensor_2","Sensor_2_Change"]].head(10))
trend_df["Sensor_2_Change"] = (trend_df["Sensor_2_Change"].fillna(0))
print(trend_df[["Engine_ID", "Cycle", "Sensor_2_Change"]].head())

trend_features_column=feature_columns + ["Sensor_2_Change"]
X_trend=trend_df[trend_features_column]
X_train_trend=X_trend.loc[X_train.index]
X_val_trend=X_trend.loc[X_val.index]
print("Training shape:", X_train_trend.shape)
print("Validation shape:", X_val_trend.shape)

trend_scaler=StandardScaler()
X_train_trend_scaled=trend_scaler.fit_transform(X_train_trend)
X_val_trend_scaled=trend_scaler.transform(X_val_trend)
print("Scaled Training shape:", X_train_trend_scaled.shape)
print("Scaled Validation shape:", X_val_trend_scaled.shape)

trend_model=LinearRegression()
trend_model.fit(X_train_trend_scaled,Y_train)

trend_train_pred=trend_model.predict(X_train_trend_scaled)
trend_val_pred=trend_model.predict(X_val_trend_scaled)

trend_train_mae=mean_absolute_error(Y_train,trend_train_pred)
trend_val_mae=mean_absolute_error(Y_val,trend_val_pred)

trend_train_rmse = np.sqrt(mean_squared_error(Y_train, trend_train_pred))
trend_val_rmse = np.sqrt(mean_squared_error(Y_val, trend_val_pred))

print("\n--- Linear Regression with Sensor 2 Change ---")
print("Training MAE:", trend_train_mae)
print("Validation MAE:", trend_val_mae)
print("Training RMSE:", trend_train_rmse)
print("Validation RMSE:", trend_val_rmse)

sensor_change_columns=[]
for sensor in sensor_columns:
    change_column= sensor + "_Change"
    trend_df[change_column]=trend_df.groupby("Engine_ID")[sensor].diff().fillna(0)
    sensor_change_columns.append(change_column)

print(trend_df[sensor_change_columns].head())

all_features_column=feature_columns+sensor_change_columns
X_all_changes=trend_df[all_features_column]
X_train_all_changes=X_all_changes.loc[X_train.index]
X_val_all_changes=X_all_changes.loc[X_val.index]
print("Training shape:", X_train_all_changes.shape)
print("Validation shape:", X_val_all_changes.shape)

all_changes_scaler=StandardScaler()
X_train_all_changes_scaled=all_changes_scaler.fit_transform(X_train_all_changes)
X_val_all_changes_scaled=all_changes_scaler.transform(X_val_all_changes)
print("Scaled Training shape:", X_train_all_changes_scaled.shape)
print("Scaled Validation shape:", X_val_all_changes_scaled.shape)

all_changes_model=LinearRegression()
all_changes_model.fit(X_train_all_changes_scaled,Y_train)
all_changes_train_pred=all_changes_model.predict(X_train_all_changes_scaled)
all_changes_val_pred=all_changes_model.predict(X_val_all_changes_scaled)

all_changes_train_mae=mean_absolute_error(Y_train,all_changes_train_pred)
all_changes_val_mae=mean_absolute_error(Y_val,all_changes_val_pred)

all_changes_train_rmse=np.sqrt(mean_squared_error(Y_train,all_changes_train_pred))
all_changes_val_rmse=np.sqrt(mean_squared_error(Y_val,all_changes_val_pred))

print("\n--- Linear Regression with all sensor changes ---")
print("Training MAE:", all_changes_train_mae)
print("Validation MAE:", all_changes_val_mae)
print("Training RMSE:", all_changes_train_rmse)
print("Validation RMSE:", all_changes_val_rmse)

from sklearn.ensemble import RandomForestRegressor
rf_model=RandomForestRegressor(n_estimators=100, max_features=0.7,random_state=42,n_jobs=-1)
rf_model.fit(X_train,Y_train)
rf_train_pred=rf_model.predict(X_train)
rf_val_pred=rf_model.predict(X_val)

rf_train_mae=mean_absolute_error(Y_train,rf_train_pred)
rf_val_mae=mean_absolute_error(Y_val,rf_val_pred)
rf_train_rmse=np.sqrt(mean_squared_error(Y_train,rf_train_pred))
rf_val_rmse=np.sqrt(mean_squared_error(Y_val,rf_val_pred))

print("\n--- Random Forest: Original 24 Features ---")
print("Training MAE:", rf_train_mae)
print("Validation MAE:", rf_val_mae)
print("Training RMSE:", rf_train_rmse)
print("Validation RMSE:", rf_val_rmse)

rf_leaf10=RandomForestRegressor(n_estimators=100, max_features=0.7,min_samples_leaf=10,random_state=42,n_jobs=-1)
rf_leaf10.fit(X_train,Y_train)
rf_leaf10_train_pred=rf_leaf10.predict(X_train)
rf_leaf10_val_pred=rf_leaf10.predict(X_val)

print("\n--- Random Forest: Original 24 Features Minimum 10 Leaf ---")
print("Leaf 10 Training MAE:",mean_absolute_error(Y_train, rf_leaf10_train_pred))
print("Leaf 10 Validation MAE:",mean_absolute_error(Y_val, rf_leaf10_val_pred))

rf_leaf5=RandomForestRegressor(n_estimators=100, max_features=0.7,min_samples_leaf=5,random_state=42,n_jobs=-1)
rf_leaf5.fit(X_train,Y_train)
rf_leaf5_train_pred=rf_leaf5.predict(X_train)
rf_leaf5_val_pred=rf_leaf5.predict(X_val)

print("\n--- Random Forest: Original 24 Features Minimum 5 Leaf ---")
print("Leaf 5 Training MAE:",mean_absolute_error(Y_train, rf_leaf5_train_pred))
print("Leaf 5 Validation MAE:",mean_absolute_error(Y_val, rf_leaf5_val_pred))

rf_changes=RandomForestRegressor(n_estimators=100,max_features=0.7,min_samples_leaf=5,random_state=42,n_jobs=-1)
rf_changes.fit(X_train_all_changes,Y_train)
rf_changes_train_pred=rf_changes.predict(X_train_all_changes)
rf_changes_val_pred=rf_changes.predict(X_val_all_changes)

rf_changes_train_mae = mean_absolute_error(Y_train, rf_changes_train_pred)
rf_changes_val_mae = mean_absolute_error(Y_val, rf_changes_val_pred)
rf_changes_train_rmse = np.sqrt(mean_squared_error(Y_train, rf_changes_train_pred))
rf_changes_val_rmse = np.sqrt(mean_squared_error(Y_val, rf_changes_val_pred))

print("\n--- Random Forest: All Sensor Changes, Leaf Size 5 ---")
print("Training MAE:", rf_changes_train_mae)
print("Validation MAE:", rf_changes_val_mae)
print("Training RMSE:", rf_changes_train_rmse)
print("Validation RMSE:", rf_changes_val_rmse)

from sklearn.model_selection import GroupKFold
cv_groups=df.loc[X_train_all_changes.index,"Engine_ID"]
group_cv= GroupKFold(n_splits=5)
print("Training engines available for CV:", cv_groups.nunique())

cv_mae_scores = []
cv_error_parts = []

for fold, (cv_train_idx, cv_val_idx) in enumerate(
    group_cv.split(X_train_all_changes, Y_train, groups=cv_groups),
    start=1
):
    X_fold_train = X_train_all_changes.iloc[cv_train_idx]
    X_fold_val = X_train_all_changes.iloc[cv_val_idx]

    Y_fold_train = Y_train.iloc[cv_train_idx]
    Y_fold_val = Y_train.iloc[cv_val_idx]

    fold_model = RandomForestRegressor(
        n_estimators=100,
        max_features=0.7,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    fold_model.fit(X_fold_train, Y_fold_train)
    fold_pred = fold_model.predict(X_fold_val)

    fold_mae = mean_absolute_error(Y_fold_val, fold_pred)
    cv_mae_scores.append(fold_mae)

    fold_errors = df.loc[X_fold_val.index,["Engine_ID", "Cycle", "RUL"]].copy()

    fold_errors["Fold"] = fold
    fold_errors["Predicted_RUL"] = fold_pred
    fold_errors["Absolute_Error"] = (
        fold_errors["Predicted_RUL"] - fold_errors["RUL"]
    ).abs()

    cv_error_parts.append(fold_errors)

    print("Fold:", fold, "Evaluation MAE:", fold_mae)

cv_error_analysis = pd.concat(cv_error_parts)

print("Average CV MAE:", np.mean(cv_mae_scores))
print("Collected evaluation rows:", len(cv_error_analysis))

cv_near_failure = cv_error_analysis[cv_error_analysis["RUL"] <= 30]

cv_middle_life = cv_error_analysis[
    (cv_error_analysis["RUL"] > 30)
    & (cv_error_analysis["RUL"] <= 100)
]

cv_earlier_life = cv_error_analysis[cv_error_analysis["RUL"] > 100]

print("\n--- CV Errors by Actual RUL ---")

print("RUL 0-30 observations:", len(cv_near_failure))
print("RUL 0-30 MAE:", cv_near_failure["Absolute_Error"].mean())

print("RUL 31-100 observations:", len(cv_middle_life))
print("RUL 31-100 MAE:", cv_middle_life["Absolute_Error"].mean())

print("RUL above 100 observations:", len(cv_earlier_life))
print("RUL above 100 MAE:", cv_earlier_life["Absolute_Error"].mean())

cv_error_analysis["Error"]=cv_error_analysis["Predicted_RUL"]- cv_error_analysis["RUL"]
print("\n--- Average Signed Error by Actual RUL ---")
print("RUL 0-30",cv_error_analysis.loc[
    cv_error_analysis["RUL"]<=30,"Error"
].mean()
)
print("RUL 30-100",cv_error_analysis.loc[
    (cv_error_analysis["RUL"]>30) & (cv_error_analysis["RUL"]<=100),"Error"
].mean()
)
print("RUL above 100:",cv_error_analysis.loc[
        cv_error_analysis["RUL"] > 100, "Error"
    ].mean()
)

X_train_with_cycle= X_train_all_changes.copy()
X_train_with_cycle["Cycle"]=df["Cycle"]
print("Training shape with Cycle:", X_train_with_cycle.shape)

cycle_cv_mae_scores=[]
cycle_cv_error_parts=[]

for fold,(cv_train_idx,cv_val_idx) in enumerate(
    group_cv.split(X_train_with_cycle,Y_train,groups=cv_groups),
    start=1
):
    X_fold_train=X_train_with_cycle.iloc[cv_train_idx]
    X_fold_val=X_train_with_cycle.iloc[cv_val_idx]
    Y_fold_train = Y_train.iloc[cv_train_idx]
    Y_fold_val = Y_train.iloc[cv_val_idx]

    cycle_fold_model=RandomForestRegressor(
        n_estimators=100,
        max_features=0.7,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    cycle_fold_model.fit(X_fold_train, Y_fold_train)
    cycle_fold_pred = cycle_fold_model.predict(X_fold_val)

    cycle_fold_errors=df.loc[X_fold_val.index,["Engine_ID","Cycle","RUL"]].copy()
    cycle_fold_errors["Predicted_RUL"]=cycle_fold_pred
    cycle_fold_errors["Absolute_Error"]=(cycle_fold_errors["Predicted_RUL"]- cycle_fold_errors["RUL"]).abs()
    cycle_cv_error_parts.append(cycle_fold_errors)

    cycle_fold_mae = mean_absolute_error(
        Y_fold_val,
        cycle_fold_pred
    )

    cycle_cv_mae_scores.append(cycle_fold_mae)
    print("Fold:", fold, "With Cycle MAE:", cycle_fold_mae)

print("Average CV MAE with Cycle:", np.mean(cycle_cv_mae_scores))

cycle_cv_error_analysis = pd.concat(cycle_cv_error_parts)
cycle_near_failure= cycle_cv_error_analysis[cycle_cv_error_analysis["RUL"]<=30]
cycle_middle_life = cycle_cv_error_analysis[(cycle_cv_error_analysis["RUL"] > 30) & (cycle_cv_error_analysis["RUL"] <= 100)]
cycle_earlier_life = cycle_cv_error_analysis[cycle_cv_error_analysis["RUL"] > 100]

print("\n--- With Cycle: CV MAE by Actual RUL ---")
print("RUL 0-30 MAE:", cycle_near_failure["Absolute_Error"].mean())
print("RUL 31-100 MAE:", cycle_middle_life["Absolute_Error"].mean())
print("RUL above 100 MAE:", cycle_earlier_life["Absolute_Error"].mean())

def calculate_slope(readings):
    if len(readings) < 2:
        return 0.0
    
    positions = np.arange(len(readings))
    centered_positions = positions - positions.mean()
    centered_readings = readings - readings.mean()

    slope = (
        np.sum(centered_positions * centered_readings)
        / np.sum(centered_positions ** 2)
    )

    return slope

# Create slope and standard-deviation features
window = 5
history_feature_columns = []

for sensor in sensor_columns:
    slope_column = sensor + "_Slope_5"
    std_column = sensor + "_Std_5"

    grouped_sensor = trend_df.groupby("Engine_ID")[sensor]

    # Pandas passes each engine's sensor sequence to this function.
    # Rolling apply passes each small window to calculate_slope.
    trend_df[slope_column] = grouped_sensor.transform(
        lambda engine_readings: engine_readings.rolling(
            window=window,
            min_periods=1
        ).apply(calculate_slope, raw=True)
    )

    trend_df[std_column] = grouped_sensor.transform(
        lambda engine_readings: engine_readings.rolling(
            window=window,
            min_periods=1
        ).std(ddof=0)
    )

    history_feature_columns.extend([slope_column, std_column])


# Add the 42 new features to our existing 46 training features
X_train_history = X_train_with_cycle.copy()

for column in history_feature_columns:
    X_train_history[column] = trend_df.loc[
        X_train_history.index, column
    ]

print("Training shape with slope and std:", X_train_history.shape)

# Evaluate using the same five engine folds
history_cv_mae_scores = []
history_cv_error_parts = []

for fold, (cv_train_idx, cv_val_idx) in enumerate(
    group_cv.split(X_train_history, Y_train, groups=cv_groups),
    start=1
):
    X_fold_train = X_train_history.iloc[cv_train_idx]
    X_fold_val = X_train_history.iloc[cv_val_idx]

    Y_fold_train = Y_train.iloc[cv_train_idx]
    Y_fold_val = Y_train.iloc[cv_val_idx]

    history_fold_model = RandomForestRegressor(
        n_estimators=100,
        max_features=0.7,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    history_fold_model.fit(X_fold_train, Y_fold_train)
    history_fold_pred = history_fold_model.predict(X_fold_val)

    history_fold_mae = mean_absolute_error(Y_fold_val, history_fold_pred)
    history_cv_mae_scores.append(history_fold_mae)

    # Save this fold's held-out predictions and errors
    history_fold_errors = df.loc[X_fold_val.index,["Engine_ID", "Cycle", "RUL"]].copy()

    history_fold_errors["Fold"] = fold
    history_fold_errors["Predicted_RUL"] = history_fold_pred

    history_fold_errors["Absolute_Error"] = (
        history_fold_errors["Predicted_RUL"]
        - history_fold_errors["RUL"]
    ).abs()

    history_cv_error_parts.append(history_fold_errors)

    print("Fold:", fold, "Slope and std MAE:", history_fold_mae)


# Combine the five evaluation tables
history_cv_error_analysis = pd.concat(history_cv_error_parts)

print("\nAverage slope and std CV MAE:",np.mean(history_cv_mae_scores))

# Examine errors in three actual-RUL ranges
history_near_failure = history_cv_error_analysis[history_cv_error_analysis["RUL"] <= 30]

history_middle_life = history_cv_error_analysis[(history_cv_error_analysis["RUL"] > 30) & (history_cv_error_analysis["RUL"] <= 100)]

history_earlier_life = history_cv_error_analysis[history_cv_error_analysis["RUL"] > 100]

print("\n--- Slope and Std: CV MAE by Actual RUL ---")
print("RUL 0-30 MAE:", history_near_failure["Absolute_Error"].mean())
print("RUL 31-100 MAE:", history_middle_life["Absolute_Error"].mean())
print("RUL above 100 MAE:", history_earlier_life["Absolute_Error"].mean())

ridge_cv_mae_scores=[]
ridge_cv_rmse_scores=[]
for fold,(cv_train_idx,cv_val_idx) in enumerate(
    group_cv.split(X_train,Y_train,groups=cv_groups),
    start=1
):
    X_fold_train=X_train.iloc[cv_train_idx]
    X_fold_val=X_train.iloc[cv_val_idx]
    Y_fold_train = Y_train.iloc[cv_train_idx]
    Y_fold_val = Y_train.iloc[cv_val_idx]

    fold_scaler=StandardScaler()
    X_fold_train_scaled=fold_scaler.fit_transform(X_fold_train)
    X_fold_val_scaled=fold_scaler.transform(X_fold_val)

    fold_poly=PolynomialFeatures(degree=2,include_bias=False)
    X_fold_train_poly=fold_poly.fit_transform(X_fold_train_scaled)
    X_fold_val_poly=fold_poly.transform(X_fold_val_scaled)

    fold_ridge=Ridge(alpha=1000)
    fold_ridge.fit(X_fold_train_poly,Y_fold_train)
    fold_ridge_pred=fold_ridge.predict(X_fold_val_poly)

    fold_ridge_mae = mean_absolute_error(Y_fold_val, fold_ridge_pred)
    fold_ridge_rmse = np.sqrt(
        mean_squared_error(Y_fold_val, fold_ridge_pred)
    )
    ridge_cv_mae_scores.append(fold_ridge_mae)
    ridge_cv_rmse_scores.append(fold_ridge_rmse)

    print("\nFold:", fold)
    print("Evaluation MAE:", fold_ridge_mae)
    print("Evaluation RMSE:", fold_ridge_rmse)

print("\n--- Polynomial Ridge: Five-Fold CV ---")
print("Average CV MAE:", np.mean(ridge_cv_mae_scores))
print("Average CV RMSE:", np.mean(ridge_cv_rmse_scores))

from xgboost import XGBRegressor
xgb_model=XGBRegressor(
    objective="reg:squarederror",
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_all_changes,Y_train)
xgb_train_pred=xgb_model.predict(X_train_all_changes)
xgb_val_pred=xgb_model.predict(X_val_all_changes)

xgb_train_mae = mean_absolute_error(Y_train, xgb_train_pred)
xgb_val_mae = mean_absolute_error(Y_val, xgb_val_pred)
xgb_train_rmse = np.sqrt(mean_squared_error(Y_train, xgb_train_pred))
xgb_val_rmse = np.sqrt(mean_squared_error(Y_val, xgb_val_pred))

print("\n--- XGBoost: All Sensor Changes ---")
print("Training MAE:", xgb_train_mae)
print("Validation MAE:", xgb_val_mae)
print("Training RMSE:", xgb_train_rmse)
print("Validation RMSE:", xgb_val_rmse)

xgb_300 = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
    n_jobs=-1
)

xgb_300.fit(X_train_all_changes, Y_train)
xgb_300_train_pred = xgb_300.predict(X_train_all_changes)
xgb_300_val_pred = xgb_300.predict(X_val_all_changes)

xgb_300_train_mae = mean_absolute_error(Y_train, xgb_300_train_pred)
xgb_300_val_mae = mean_absolute_error(Y_val, xgb_300_val_pred)

xgb_300_train_rmse = np.sqrt(
    mean_squared_error(Y_train, xgb_300_train_pred)
)
xgb_300_val_rmse = np.sqrt(
    mean_squared_error(Y_val, xgb_300_val_pred)
)

print("\n--- XGBoost: 300 Trees ---")
print("Training MAE:", xgb_300_train_mae)
print("Validation MAE:", xgb_300_val_mae)
print("Training RMSE:", xgb_300_train_rmse)
print("Validation RMSE:", xgb_300_val_rmse)

xgb_cv_mae_scores = []
for fold, (cv_train_idx, cv_val_idx) in enumerate(
    group_cv.split(X_train_all_changes, Y_train, groups=cv_groups),
    start=1
):
    X_fold_train = X_train_all_changes.iloc[cv_train_idx]
    X_fold_val = X_train_all_changes.iloc[cv_val_idx]
    Y_fold_train = Y_train.iloc[cv_train_idx]
    Y_fold_val = Y_train.iloc[cv_val_idx]

    fold_xgb = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        n_jobs=-1
    )

    fold_xgb.fit(X_fold_train, Y_fold_train)
    fold_xgb_pred = fold_xgb.predict(X_fold_val)
    fold_xgb_mae = mean_absolute_error(Y_fold_val, fold_xgb_pred)
    xgb_cv_mae_scores.append(fold_xgb_mae)
    print("Fold:", fold, "Evaluation MAE:", fold_xgb_mae)

print("Average XGBoost CV MAE:", np.mean(xgb_cv_mae_scores))

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf 
tf.keras.utils.set_random_seed(42)
nn_model=tf.keras.Sequential([
    tf.keras.Input(shape=(45,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="linear")
])
nn_model.summary()

nn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mean_squared_error",
    metrics=["mae"]
)
history= nn_model.fit(
    X_train_all_changes_scaled,
    Y_train,
    validation_data=(X_val_all_changes_scaled,Y_val),
    epochs=50,
    batch_size=64,
    verbose=2
)
nn_train_pred=nn_model.predict(X_train_all_changes_scaled,verbose=0).ravel()
nn_val_pred = nn_model.predict(X_val_all_changes_scaled, verbose=0).ravel()

nn_train_mae = mean_absolute_error(Y_train, nn_train_pred)
nn_val_mae = mean_absolute_error(Y_val, nn_val_pred)

nn_train_rmse = np.sqrt(mean_squared_error(Y_train, nn_train_pred))
nn_val_rmse = np.sqrt(mean_squared_error(Y_val, nn_val_pred))

print("\n--- Neural Network: 45 Features ---")
print("Training MAE:", nn_train_mae)
print("Validation MAE:", nn_val_mae)
print("Training RMSE:", nn_train_rmse)
print("Validation RMSE:", nn_val_rmse)

nn_cv_mae_scores=[]
nn_cv_rmse_scores=[]

for fold, (cv_train_idx,cv_val_idx) in enumerate(
    group_cv.split(X_train_all_changes,Y_train,groups=cv_groups),
    start=1
):
    X_fold_train=X_train_all_changes.iloc[cv_train_idx]
    X_fold_val = X_train_all_changes.iloc[cv_val_idx]
    Y_fold_train = Y_train.iloc[cv_train_idx]
    Y_fold_val = Y_train.iloc[cv_val_idx]

    fold_scaler=StandardScaler()
    X_fold_train_scaled = fold_scaler.fit_transform(X_fold_train)
    X_fold_val_scaled = fold_scaler.transform(X_fold_val)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    fold_nn=tf.keras.Sequential([
        tf.keras.Input(shape=(X_fold_train_scaled.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear")
    ])

    fold_nn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mean_squared_error",
        metrics=["mae"]
    )
    print("\nTraining fold:", fold)
    fold_nn.fit(X_fold_train_scaled,Y_fold_train,epochs=50,batch_size=64,verbose=0)
    fold_nn_pred=fold_nn.predict(X_fold_val_scaled,verbose=0).ravel()
    fold_nn_mae = mean_absolute_error(Y_fold_val, fold_nn_pred)
    fold_nn_rmse = np.sqrt(
        mean_squared_error(Y_fold_val, fold_nn_pred)
    )

    nn_cv_mae_scores.append(fold_nn_mae)
    nn_cv_rmse_scores.append(fold_nn_rmse)
    print("Evaluation MAE:", fold_nn_mae)
    print("Evaluation RMSE:", fold_nn_rmse)

print("\n--- Neural Network: Five-Fold CV ---")
print("Average CV MAE:", np.mean(nn_cv_mae_scores))
print("Average CV RMSE:", np.mean(nn_cv_rmse_scores))

error_analysis=df.loc[X_val_all_changes.index,["Engine_ID","Cycle","RUL"]].copy()
error_analysis["Predicted_RUL"]=rf_changes_val_pred
error_analysis["Error"]=(error_analysis["Predicted_RUL"] - error_analysis["RUL"])
error_analysis["Absolute_Error"] = error_analysis["Error"].abs()
print(error_analysis.head(10))

near_failure=error_analysis[error_analysis["RUL"]<=30]
earlier_life=error_analysis[error_analysis["RUL"]>30]

print("Near-failure observations",len(near_failure))
print("Near-failure Mae",near_failure["Absolute_Error"].mean())

print("Earlier-life observations:", len(earlier_life))
print("Earlier-life MAE:", earlier_life["Absolute_Error"].mean())

print("\nLargest near-failure overestimates:")
print(near_failure.sort_values("Error",ascending=False).head(10))

print("\nLargest earlier-life overestimates:")
print(earlier_life.sort_values("Error", ascending=False).head(10))

missed_warnings=near_failure[near_failure["Predicted_RUL"]>30]
missed_warning_percentage=(len(missed_warnings)/len(near_failure))*100

print("Near-failure observations:", len(near_failure))
print("Missed-warning observations:", len(missed_warnings))
print("Missed-warning percentage:", missed_warning_percentage)

false_alarms=earlier_life[earlier_life["Predicted_RUL"]<=30]
false_alarm_percentage=(len(false_alarms)/len(earlier_life))*100

print("Earlier-life observations:", len(earlier_life))
print("False-alarm observations:", len(false_alarms))
print("False-alarm percentage:", false_alarm_percentage)


missed_warnings_40=near_failure[near_failure["Predicted_RUL"]>40]
missed_warnings_40_percentage=(len(missed_warnings_40)/len(near_failure))*100
false_alarms_40=earlier_life[earlier_life["Predicted_RUL"]<=40]
false_alarm_40_percentage=(len(false_alarms_40)/len(earlier_life))*100

print("\n--- Warning Threshold: 40 Cycles ---")
print("Near-failure observations:", len(near_failure))
print("Missed-warning observations:", len(missed_warnings_40))
print("Missed-warning percentage:", missed_warnings_40_percentage)

print("Earlier-life observations:", len(earlier_life))
print("False-alarm observations:", len(false_alarms_40))
print("False-alarm percentage:", false_alarm_40_percentage)

engine_id=74
engine_errors=error_analysis[error_analysis["Engine_ID"]==engine_id].sort_values("Cycle")
plt.figure(figsize=(10,5))
plt.plot(engine_errors["Cycle"],engine_errors["RUL"],label="Actual RUL",color="blue")
plt.plot(engine_errors["Cycle"],engine_errors["Predicted_RUL"],label="Predicted RUL",color="orange")

plt.axhline(
    y=40,
    color="red",
    linestyle="--",
    label="Prediction warning threshold: 40"
)

plt.axhline(
    y=30,
    color="green",
    linestyle=":",
    label="Actual near-failure cutoff: 30"
)

plt.xlabel("Operating Cycle")
plt.ylabel("Remaining useful life (cycles)")
plt.title(f"Engine {engine_id}: Actual vs Predicted RUL")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

baseline_cv_mae_scores = []
for fold, (cv_train_idx, cv_val_idx) in enumerate(
    group_cv.split(X_train, Y_train, groups=cv_groups),
    start=1
):
    Y_fold_train = Y_train.iloc[cv_train_idx]
    Y_fold_val = Y_train.iloc[cv_val_idx]

    baseline_rul = Y_fold_train.median()
    baseline_pred = np.full(len(Y_fold_val), baseline_rul)

    baseline_mae = mean_absolute_error(Y_fold_val, baseline_pred)
    baseline_cv_mae_scores.append(baseline_mae)

    print("Fold:", fold, "Baseline MAE:", baseline_mae)
print("Average baseline CV MAE:", np.mean(baseline_cv_mae_scores))

rolling_sensor_columns=[]
for sensor in sensor_columns:
    rolling_column= sensor + "_Mean_5"
    trend_df[rolling_column]=trend_df.groupby("Engine_ID")[sensor].transform(
    lambda readings:readings.rolling(
        window=5,
        min_periods=1
    ).mean())
    rolling_sensor_columns.append(rolling_column)

rolling_feature_columns = (feature_columns + sensor_change_columns + rolling_sensor_columns)
X_rolling=trend_df[rolling_feature_columns]
X_train_rolling = X_rolling.loc[X_train.index]
X_val_rolling = X_rolling.loc[X_val.index]
print("Training shape:", X_train_rolling.shape)
print("Validation shape:", X_val_rolling.shape)

rolling_cv_mae_scores=[]
for fold,(cv_train_idx,cv_val_idx) in enumerate(
    group_cv.split(X_train_rolling,Y_train,groups=cv_groups),
    start=1
):
    X_fold_train = X_train_rolling.iloc[cv_train_idx]
    X_fold_val = X_train_rolling.iloc[cv_val_idx]
    Y_fold_train = Y_train.iloc[cv_train_idx]
    Y_fold_val = Y_train.iloc[cv_val_idx]

    rolling_fold_model = RandomForestRegressor(
        n_estimators=100,
        max_features=0.7,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rolling_fold_model.fit(X_fold_train,Y_fold_train)
    rolling_fold_pred=rolling_fold_model.predict(X_fold_val)
    rolling_fold_mae=mean_absolute_error(Y_fold_val,rolling_fold_pred)
    rolling_cv_mae_scores.append(rolling_fold_mae)
    print("Fold:", fold, "Rolling-feature MAE:", rolling_fold_mae)

print("Average rolling-feature CV MAE:", np.mean(rolling_cv_mae_scores))

# Prepare the same 88 features for the validation engines
X_val_history = X_val_all_changes.copy()

X_val_history["Cycle"] = df.loc[X_val_history.index, "Cycle"]

for column in history_feature_columns:
    X_val_history[column] = trend_df.loc[X_val_history.index, column]

# Match the training feature order exactly
X_val_history = X_val_history[X_train_history.columns]

print("Final training shape:", X_train_history.shape)
print("Final validation shape:", X_val_history.shape)


# Train on all 80 training engines
final_rf_model = RandomForestRegressor(
    n_estimators=100,
    max_features=0.7,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

final_rf_model.fit(X_train_history, Y_train)

import joblib

joblib.dump(
    {
        "model": final_rf_model,
        "feature_columns": X_train_history.columns.tolist()
    },
    "turbofan_rul_predictor.joblib"
)

print("Random Forest saved successfully.")

# Predict using the fitted model
final_rf_train_pred = final_rf_model.predict(X_train_history)
final_rf_val_pred = final_rf_model.predict(X_val_history)


# Calculate training and validation errors
final_rf_train_mae = mean_absolute_error(Y_train, final_rf_train_pred)

final_rf_val_mae = mean_absolute_error(Y_val, final_rf_val_pred)

final_rf_train_rmse = np.sqrt(mean_squared_error(Y_train, final_rf_train_pred))

final_rf_val_rmse = np.sqrt(mean_squared_error(Y_val, final_rf_val_pred))

print("\n--- Selected Random Forest: 88 Features ---")
print("Training MAE:", final_rf_train_mae)
print("Validation MAE:", final_rf_val_mae)
print("Training RMSE:", final_rf_train_rmse)
print("Validation RMSE:", final_rf_val_rmse)