"""Production-only FastAPI entrypoint for SMARTMAINTAIN RUL inference."""

import logging
import os
import pickle
from typing import List

import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import RedirectResponse
from uvicorn import run as app_run


MODEL_PATH = os.path.join("final_model", "model.keras")
PREPROCESSOR_PATH = os.path.join("final_model", "preprocessing.pkl")
WINDOW_SIZE = 30

FEATURE_COLUMNS = [
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]


class CMAPSSCycle(BaseModel):
    """One C-MAPSS engine-cycle observation in the production feature order."""

    cycle: float
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float

    class Config:
        extra = "forbid"


class RULPredictionRequest(BaseModel):
    cycles: List[CMAPSSCycle]

    class Config:
        extra = "forbid"


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["authentication"])
async def index():
    """Redirect API users to the interactive documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    """Return the service and deployed-model identity."""
    return {"status": "ok", "model": "LSTM", "dataset": "C-MAPSS FD001"}


@app.post("/predict_rul")
async def predict_rul(request: RULPredictionRequest):
    """Predict RUL from exactly 30 consecutive C-MAPSS cycle records."""
    if len(request.cycles) != WINDOW_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Exactly {WINDOW_SIZE} cycle records are required; received {len(request.cycles)}.",
        )
    if not os.path.isfile(PREPROCESSOR_PATH):
        raise HTTPException(
            status_code=503,
            detail=f"Preprocessing artifact not found: {PREPROCESSOR_PATH}.",
        )
    if not os.path.isfile(MODEL_PATH):
        raise HTTPException(
            status_code=503,
            detail=f"Model artifact not found: {MODEL_PATH}.",
        )

    try:
        feature_dataframe = pd.DataFrame(
            [cycle.dict() for cycle in request.cycles], columns=FEATURE_COLUMNS
        )
        with open(PREPROCESSOR_PATH, "rb") as preprocessing_file:
            scaler = pickle.load(preprocessing_file)
        scaled_features = scaler.transform(feature_dataframe)
        sequence = scaled_features.reshape(1, WINDOW_SIZE, len(FEATURE_COLUMNS))
        model = tf.keras.models.load_model(MODEL_PATH)
        predicted_rul = float(model.predict(sequence, verbose=0).ravel()[0])
        return {"predicted_rul": predicted_rul, "unit": "cycles"}
    except Exception as error:
        logging.exception("RUL prediction failed")
        raise HTTPException(status_code=500, detail=f"RUL prediction failed: {error}") from error


if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8080)
