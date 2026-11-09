# Turbofan RUL Predictor

An end-to-end machine learning application that estimates turbofan engine Remaining Useful Life (RUL) from sensor histories using NASA C-MAPSS FD001 data.

**[Live Demo](https://turbofan-rul-predictor-1.onrender.com)** · **[API Docs](https://turbofan-rul-predictor-50ze.onrender.com/docs)**

## Overview

Upload an FD001-format text file to generate predictions for multiple engines. The application validates the input, computes temporal features, and displays estimated remaining cycles after each engine’s last recorded cycle.

- Random Forest inference using 88 engineered features.
- Sensor differences, rolling slopes, and rolling standard deviations.
- Engine-grouped training/validation split and grouped cross-validation.
- FastAPI backend with file validation and JSON responses.
- Responsive React interface with loading states and error handling.
- Frontend and backend deployed on Render.

## Results

Evaluated on the final observed cycle of each of the 100 engines in the official FD001 test set:

| Metric | Result |
|---|---:|
| MAE | **18.78 cycles** |
| RMSE | **25.57 cycles** |

Model development explored linear regression, polynomial Ridge regression, Random Forest, and a GRU sequence model. The deployed Random Forest uses 100 trees and is loaded from a saved artifact without retraining during requests.

## Tech Stack

**Machine Learning:** Python, scikit-learn, pandas, NumPy, joblib  
**Backend:** FastAPI, Uvicorn  
**Frontend:** React, JavaScript, CSS, Vite  
**Deployment:** Render, GitHub

## Run Locally

Clone the repository:

```bash
git clone https://github.com/vyomeshk2006-code/turbofan-rul-predictor.git
cd turbofan-rul-predictor
```

Set up the backend using Python 3.13.7:

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies and start the API:

```bash
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

In a separate terminal, start the frontend using Node.js 24:

```bash
cd frontend
npm ci
npm run dev
```

Open the address printed by Vite, typically `http://localhost:5173`.

## Usage

1. Upload `data/test_FD001.txt`.
2. Click **Predict RUL**.
3. View each engine’s last recorded cycle and predicted remaining cycles.

Input files must contain 26 whitespace-separated numeric columns with no header: engine ID, cycle, three operating settings, and 21 sensor readings. Maximum upload size is 10 MB.

`data/RUL_FD001.txt` contains evaluation targets and is not an input file.

## Configuration

| Variable | Set on | Purpose |
|---|---|---|
| `VITE_API_URL` | Frontend | Backend base URL; defaults to `http://127.0.0.1:8000` |
| `FRONTEND_URL` | Backend | Public frontend origin allowed through CORS |

Rebuild the frontend after changing `VITE_API_URL`.

## Notes

The free backend may take time to wake after inactivity. Predictions are specific to the FD001 dataset and are estimates, not aircraft maintenance instructions.

**Dataset:** NASA C-MAPSS FD001. Credit to NASA and the dataset authors.

## Author

**Vyomesh Kumar** · [GitHub](https://github.com/vyomeshk2006-code)