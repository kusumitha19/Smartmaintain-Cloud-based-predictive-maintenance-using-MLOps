
"""
SMARTMAINTAIN: XGBoost RUL Regression Experiment

Uses NASA C-MAPSS FD001 data.
Creates one feature vector per engine from its latest 30 observed cycles.
Trains XGBoost and evaluates on the same 100 test engines as the LSTM.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Project paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "CMAPSS"
OUTPUT_DIR = ROOT / "experiment_results" / "xgboost"

TRAIN_FILE = DATA_DIR / "train_FD001.txt"
TEST_FILE = DATA_DIR / "test_FD001.txt"
RUL_FILE = DATA_DIR / "RUL_FD001.txt"

WINDOW_SIZE = 30
RUL_CAP = 125
RANDOM_STATE = 42

RAW_COLUMNS = (
    ["unit_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

FEATURE_COLUMNS = RAW_COLUMNS[2:]


def read_cmapss_file(path):
    """Read a whitespace-separated C-MAPSS file."""
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=RAW_COLUMNS,
        engine="python",
    )


def add_rul_to_training_data(df):
    """Calculate capped RUL for every training cycle."""
    result = df.copy()
    max_cycles = result.groupby("unit_id")["cycle"].transform("max")
    result["RUL"] = (max_cycles - result["cycle"]).clip(upper=RUL_CAP)
    return result


def create_window_features(engine_data):
    """
    Convert an engine's latest 30 cycles into tabular features.
    Uses the same 25 input features as the LSTM.
    """
    engine_data = engine_data.sort_values("cycle")
    window = engine_data.tail(WINDOW_SIZE)

    if len(window) < WINDOW_SIZE:
        raise ValueError(
            f"Engine {engine_data['unit_id'].iloc[0]} has "
            f"only {len(window)} cycles; {WINDOW_SIZE} are required."
        )

    features = {}

    for column in FEATURE_COLUMNS:
        values = window[column].to_numpy(dtype=np.float64)

        features[f"{column}_latest"] = values[-1]
        features[f"{column}_mean"] = values.mean()
        features[f"{column}_std"] = values.std(ddof=0)
        features[f"{column}_trend"] = np.polyfit(
            np.arange(WINDOW_SIZE), values, 1
        )[0]

    return features


def main():
    for path in (TRAIN_FILE, TEST_FILE, RUL_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset file not found: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading C-MAPSS FD001 data...")
    train_df = add_rul_to_training_data(read_cmapss_file(TRAIN_FILE))
    test_df = read_cmapss_file(TEST_FILE)
    true_test_rul = pd.read_csv(RUL_FILE, header=None).iloc[:, 0].to_numpy()

    train_rows = []
    for engine_id, engine_data in train_df.groupby("unit_id"):
        # Train on windows ending at randomly selected observed cycles.
        # The target is the capped RUL at each selected ending cycle.
        engine_data = engine_data.sort_values("cycle")

        if len(engine_data) < WINDOW_SIZE:
            continue

        # Use all eligible window-ending cycles for training.
        for end_position in range(WINDOW_SIZE, len(engine_data) + 1):
            window = engine_data.iloc[:end_position].tail(WINDOW_SIZE)
            feature_row = create_window_features(window)
            feature_row["RUL"] = float(window["RUL"].iloc[-1])
            train_rows.append(feature_row)

    train_features = pd.DataFrame(train_rows)
    X_train = train_features.drop(columns=["RUL"])
    y_train = train_features["RUL"]

    # C-MAPSS RUL file contains the true RUL after the last observed test cycle.
    test_rows = []
    test_engine_ids = []

    for engine_id, engine_data in test_df.groupby("unit_id", sort=True):
        test_rows.append(create_window_features(engine_data))
        test_engine_ids.append(int(engine_id))

    X_test = pd.DataFrame(test_rows, columns=X_train.columns)

    if len(test_engine_ids) != len(true_test_rul):
        raise ValueError(
            f"Test engine count ({len(test_engine_ids)}) does not match "
            f"RUL label count ({len(true_test_rul)})."
        )

    y_test = np.minimum(true_test_rul.astype(np.float64), RUL_CAP)

    print(f"Training windows: {len(X_train)}")
    print(f"Training features: {X_train.shape[1]}")
    print(f"Test engines: {len(X_test)}")

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )

    print("Training XGBoost...")
    model.fit(X_train, y_train)

    predictions = np.clip(model.predict(X_test), 0, RUL_CAP)

    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))

    metrics = {
        "model": "XGBoost Regressor",
        "dataset": "NASA C-MAPSS FD001",
        "window_size": WINDOW_SIZE,
        "rul_cap": RUL_CAP,
        "test_engine_count": len(test_engine_ids),
        "training_window_count": len(X_train),
        "feature_count": X_train.shape[1],
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "hyperparameters": model.get_params(),
    }

    predictions_df = pd.DataFrame(
        {
            "engine_id": test_engine_ids,
            "actual_rul": y_test,
            "predicted_rul": predictions,
            "absolute_error": np.abs(y_test - predictions),
        }
    )

    predictions_df.to_csv(
        OUTPUT_DIR / "xgboost_test_predictions.csv", index=False
    )
    (OUTPUT_DIR / "xgboost_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str),
        encoding="utf-8",
    )
    model.save_model(str(OUTPUT_DIR / "xgboost_model.json"))

    print("\nXGBOOST TEST RESULTS")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")
    print(f"\nResults saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
