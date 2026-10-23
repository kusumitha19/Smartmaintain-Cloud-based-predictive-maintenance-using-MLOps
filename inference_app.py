"""Production-only FastAPI entrypoint for SMARTMAINTAIN RUL inference."""

import logging
import os
import pickle
import tempfile
import time
import zipfile
from typing import List

import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, make_asgi_app
from uvicorn import run as app_run


HTTP_REQUESTS = Counter(
    "smartmaintain_http_requests_total",
    "Total number of HTTP requests.",
    ["method", "endpoint", "status_code"],
)
HTTP_REQUEST_DURATION = Histogram(
    "smartmaintain_http_request_duration_seconds",
    "HTTP request processing duration in seconds.",
    ["method", "endpoint"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_model", "model.keras")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "final_model", "preprocessing.pkl")
TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "CMAPSS", "test_FD001.txt")
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


ENGINE_TEST_COLUMNS = [
    "unit_id",
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

ENGINE_FEATURE_COLUMNS = [
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


def load_production_model(model_path: str) -> tf.keras.Model:
    """Load the verified production LSTM model from the .keras archive."""
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        pass

    import h5py
    import tempfile
    import zipfile

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(30, 25)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="mse",
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(model_path, "r") as archive:
            archive.extract("model.weights.h5", path=temp_dir)

        weights_path = os.path.join(temp_dir, "model.weights.h5")

        with h5py.File(weights_path, "r") as f:

            def get_weights(path):
                """Read a dataset from the Keras .keras HDF5 archive."""
                parts = path.split("/")

                # Some Keras objects are stored as complete root-level
                # keys containing backslashes, for example:
                # layers\lstm\cell
                # layers\dense
                if len(parts) >= 3:
                    root_key = "\\".join(parts[:3])

                    if root_key in f:
                        obj = f[root_key]

                        for part in parts[3:]:
                            obj = obj[part]

                        return obj[:]

                # Dense layers are stored as:
                # layers\dense -> vars -> 0/1
                root_key = "\\".join(parts[:2])
                obj = f[root_key]

                for part in parts[2:]:
                    obj = obj[part]

                return obj[:]

            # First LSTM
            lstm1_weights = [
                get_weights("layers/lstm/cell/vars/0"),
                get_weights("layers/lstm/cell/vars/1"),
                get_weights("layers/lstm/cell/vars/2"),
            ]

            # Second LSTM
            lstm2_weights = [
                get_weights("layers/lstm_1/cell/vars/0"),
                get_weights("layers/lstm_1/cell/vars/1"),
                get_weights("layers/lstm_1/cell/vars/2"),
            ]

            # Dense layers
            dense1_weights = [
                get_weights("layers/dense/vars/0"),
                get_weights("layers/dense/vars/1"),
            ]

            dense2_weights = [
                get_weights("layers/dense_1/vars/0"),
                get_weights("layers/dense_1/vars/1"),
            ]

        model.layers[0].set_weights(lstm1_weights)
        model.layers[2].set_weights(lstm2_weights)
        model.layers[4].set_weights(dense1_weights)
        model.layers[5].set_weights(dense2_weights)

    return model
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def record_http_metrics(request, call_next):
    start_time = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")
        HTTP_REQUESTS.labels(request.method, endpoint, str(status_code)).inc()
        HTTP_REQUEST_DURATION.labels(request.method, endpoint).observe(
            time.perf_counter() - start_time
        )


@app.get("/", tags=["authentication"])
async def index():
    """Serve the SMARTMAINTAIN frontend."""
    return FileResponse("templates/index.html")


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
        model = load_production_model(MODEL_PATH)
        predicted_rul = float(model.predict(sequence, verbose=0).ravel()[0])
        return {"predicted_rul": predicted_rul, "unit": "cycles"}
    except Exception as error:
        logging.exception("RUL prediction failed")
        raise HTTPException(status_code=500, detail=f"RUL prediction failed: {error}") from error


@app.post("/predict_engine_rul")
async def predict_engine_rul(request: dict):
    """Predict RUL for a specific engine using its latest 30 cycles."""
    if not isinstance(request, dict):
        raise HTTPException(
            status_code=400,
            detail="Request body must be a JSON object with an integer 'engine_id'.",
        )

    engine_id = request.get("engine_id")
    if isinstance(engine_id, bool) or not isinstance(engine_id, int):
        raise HTTPException(
            status_code=400,
            detail="engine_id must be an integer.",
        )

    if not os.path.isfile(TEST_DATA_PATH):
        raise HTTPException(
            status_code=503,
            detail=f"Test data file not found: {TEST_DATA_PATH}.",
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
        engine_data = pd.read_csv(
            TEST_DATA_PATH,
            sep=r"\s+",
            header=None,
            names=ENGINE_TEST_COLUMNS,
            engine="python",
        )
    except Exception as error:
        logging.exception("Failed to load engine test data")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to load engine test data: {error}",
        ) from error

    engine_rows = engine_data[engine_data["unit_id"] == engine_id].sort_values("cycle").reset_index(drop=True)
    if engine_rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Engine ID {engine_id} not found in test data.",
        )

    if len(engine_rows) < WINDOW_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Engine {engine_id} has fewer than {WINDOW_SIZE} cycles.",
        )

    latest_window = engine_rows.iloc[-WINDOW_SIZE:][ENGINE_FEATURE_COLUMNS].astype(float)
    try:
        with open(PREPROCESSOR_PATH, "rb") as preprocessing_file:
            scaler = pickle.load(preprocessing_file)
        scaled_features = scaler.transform(latest_window.to_numpy())
        sequence = scaled_features.reshape(1, WINDOW_SIZE, len(ENGINE_FEATURE_COLUMNS))
        model = load_production_model(MODEL_PATH)
        predicted_rul = float(model.predict(sequence, verbose=0).ravel()[0])
    except Exception as error:
        logging.exception("Engine RUL prediction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Engine RUL prediction failed: {error}",
        ) from error

    return {
        "engine_id": int(engine_id),
        "latest_cycle": int(engine_rows["cycle"].iloc[-1]),
        "window_size": WINDOW_SIZE,
        "predicted_rul": predicted_rul,
    }


app.mount("/metrics", make_asgi_app())


if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8080)
