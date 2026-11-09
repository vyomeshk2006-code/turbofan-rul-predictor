from fastapi import FastAPI, UploadFile, HTTPException
import pandas as pd
from features import create_features
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path

app = FastAPI(title="Turbofan RUL Predictor")

allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

frontend_url = os.getenv("FRONTEND_URL")

if frontend_url:
    allowed_origins.append(frontend_url.rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["POST"],
    allow_headers=["*"],
)

model_path = Path(__file__).resolve().parent / "turbofan_rul_predictor.joblib"
saved_model = joblib.load(model_path)

model = saved_model["model"]
feature_columns = saved_model["feature_columns"]

@app.get("/")
def home():
    return {"message": "Turbofan RUL Predictor backend is running"}

@app.post("/predict")
def predict(file: UploadFile):
    
    max_file_size = 10 * 1024 * 1024
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > max_file_size:
        raise HTTPException(
            status_code=413,
            detail="The file is too large. Maximum upload size is 10 MB."
        )
    
    # Read the uploaded file
    try:
        uploaded_df = pd.read_csv(file.file, sep=r"\s+", header=None)
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty. Upload engine readings."
        )
    except pd.errors.ParserError:
        raise HTTPException(
            status_code=400,
            detail="Could not read the table. Check that rows have consistent columns."
        )
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Could not read the file as UTF-8 text. Upload a plain-text FD001 file."
        )
    
    if uploaded_df.shape[1] != 26:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected 26 columns, but received "
                f"{uploaded_df.shape[1]}. "
                "Upload engine readings in the FD001 format."
            )
        )

    # Create the 26 column names
    columns = (
        ["Engine_ID", "Cycle"]
        + ["Setting_1", "Setting_2", "Setting_3"]
        + [f"Sensor_{i}" for i in range(1, 22)]
    )

    # Assign those names to the DataFrame
    uploaded_df.columns = columns

    uploaded_df = uploaded_df.apply(pd.to_numeric, errors="coerce")

    if uploaded_df.isna().any().any():
        raise HTTPException(
            status_code=400,
            detail="All 26 columns must contain numbers, with no missing values."
        )

    if not np.isfinite(uploaded_df.to_numpy()).all():
        raise HTTPException(
            status_code=400,
            detail="All values must be finite numbers. Infinity is not allowed."
        )

    id_cycle_columns = uploaded_df[["Engine_ID", "Cycle"]]

    if (id_cycle_columns <= 0).any().any():
        raise HTTPException(
            status_code=400,
            detail="Engine_ID and Cycle must be greater than zero."
        )

    if (id_cycle_columns % 1 != 0).any().any():
        raise HTTPException(
            status_code=400,
            detail="Engine_ID and Cycle must be whole numbers."
        )

    duplicate_rows = uploaded_df.duplicated(subset=["Engine_ID", "Cycle"])

    if duplicate_rows.any():
        raise HTTPException(
            status_code=400,
            detail="Each engine must have only one row per cycle. Duplicate Engine_ID and Cycle pairs were found."
        )

    sorted_df = uploaded_df.sort_values(["Engine_ID", "Cycle"])
    cycle_steps = sorted_df.groupby("Engine_ID")["Cycle"].diff()

    if cycle_steps.dropna().ne(1).any():
        raise HTTPException(
            status_code=400,
            detail="Cycles must be consecutive within each engine. Missing cycles were found."
        )
    
    feature_df=create_features(uploaded_df)

    last_rows = feature_df.groupby("Engine_ID").tail(1).copy()
    model_inputs = last_rows[feature_columns]
    predictions = model.predict(model_inputs)
    results = last_rows[["Engine_ID", "Cycle"]].copy()
    results["Predicted_RUL"] = predictions

    # Send the summary back to the browser
    return {
        "filename": file.filename,
        "rows": uploaded_df.shape[0],
        "columns": uploaded_df.shape[1],
        "engines": uploaded_df["Engine_ID"].nunique(),
        "columns_after_features": feature_df.shape[1],
        "model_feature_count": len(feature_columns),
        "predictions": results.to_dict(orient="records")
    }
