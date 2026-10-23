
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_DIR = Path("data/CMAPSS")
OUTPUT_DIR = Path("experiment_results/random_forest")

RUL_CAP = 125
WINDOW_SIZE = 30
RANDOM_STATE = 42

RAW_COLUMNS = (
    ["unit_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Use the same input features as the LSTM:
# cycle, 3 operating settings, and 21 sensors.
FEATURE_COLUMNS = RAW_COLUMNS[1:]


# --------------------------------------------------
# 2. Load NASA C-MAPSS FD001 data
# --------------------------------------------------

def load_data():
    train = pd.read_csv(
        DATA_DIR / "train_FD001.txt",
        sep=r"\s+",
        header=None,
    )

    test = pd.read_csv(
        DATA_DIR / "test_FD001.txt",
        sep=r"\s+",
        header=None,
    )

    rul = pd.read_csv(
        DATA_DIR / "RUL_FD001.txt",
        sep=r"\s+",
        header=None,
    )

    train = train.iloc[:, :26]
    test = test.iloc[:, :26]

    train.columns = RAW_COLUMNS
    test.columns = RAW_COLUMNS

    rul_values = rul.iloc[:, 0].to_numpy(dtype=float)

    return train, test, rul_values


# --------------------------------------------------
# 3. Create statistical features from each window
# --------------------------------------------------

def extract_window_features(window):
    features = []

    for column in FEATURE_COLUMNS:
        values = window[column].to_numpy(dtype=float)

        features.extend([
            values[-1],             # Latest value
            values.mean(),          # Mean
            values.std(ddof=0),     # Standard deviation
            values[-1] - values[0], # Change across the window
        ])

    return features


# --------------------------------------------------
# 4. Prepare training examples
# --------------------------------------------------

def prepare_training_data(train):
    X = []
    y = []

    for engine_id, engine_data in train.groupby("unit_id"):
        engine_data = engine_data.sort_values("cycle").reset_index(drop=True)

        total_cycles = len(engine_data)

        for end in range(WINDOW_SIZE, total_cycles + 1):
            window = engine_data.iloc[end - WINDOW_SIZE:end]

            # RUL at the last cycle of this window.
            rul_value = total_cycles - end
            rul_value = min(rul_value, RUL_CAP)

            X.append(extract_window_features(window))
            y.append(rul_value)

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


# --------------------------------------------------
# 5. Prepare one test example per engine
# --------------------------------------------------

def prepare_test_data(test, rul_values):
    X = []
    y = []
    engine_ids = []
    latest_cycles = []

    engines = sorted(test["unit_id"].unique())

    if len(engines) != len(rul_values):
        raise ValueError(
            "The number of test engines does not match "
            "the number of RUL labels."
        )

    for engine_id in engines:
        engine_data = test[test["unit_id"] == engine_id]
        engine_data = engine_data.sort_values("cycle").reset_index(drop=True)

        if len(engine_data) < WINDOW_SIZE:
            raise ValueError(
                f"Engine {engine_id} has fewer than "
                f"{WINDOW_SIZE} cycles."
            )

        window = engine_data.tail(WINDOW_SIZE)

        X.append(extract_window_features(window))
        y.append(min(float(rul_values[int(engine_id) - 1]), RUL_CAP))
        engine_ids.append(int(engine_id))
        latest_cycles.append(int(engine_data["cycle"].iloc[-1]))

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        engine_ids,
        latest_cycles,
    )


# --------------------------------------------------
# 6. Train and evaluate Random Forest
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading NASA C-MAPSS FD001 dataset...")
    train, test, rul_values = load_data()

    print("Preparing training windows...")
    X_train, y_train = prepare_training_data(train)

    print("Preparing test engines...")
    X_test, y_test, engine_ids, latest_cycles = prepare_test_data(
        test, rul_values
    )

    print(f"Training examples: {len(X_train)}")
    print(f"Test engines: {len(X_test)}")
    print(f"Features per example: {X_train.shape[1]}")

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("Training Random Forest...")
    model.fit(X_train, y_train)

    print("Evaluating on the official test engines...")
    predictions = np.clip(model.predict(X_test), 0, None)

    metrics = {
        "model": "Random Forest",
        "dataset": "NASA C-MAPSS FD001",
        "test_engine_count": int(len(y_test)),
        "training_window_count": int(len(X_train)),
        "window_size": WINDOW_SIZE,
        "rul_cap": RUL_CAP,
        "feature_count": int(X_train.shape[1]),
        "n_estimators": 300,
        "max_depth": 20,
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2": float(r2_score(y_test, predictions)),
    }

    predictions_df = pd.DataFrame({
        "engine_id": engine_ids,
        "latest_cycle": latest_cycles,
        "actual_rul": y_test,
        "predicted_rul": predictions,
    })

    predictions_df.to_csv(
        OUTPUT_DIR / "random_forest_test_predictions.csv",
        index=False,
    )

    with open(OUTPUT_DIR / "random_forest_metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    # Save the model for reproducibility.
    import joblib

    joblib.dump(model, OUTPUT_DIR / "random_forest_model.joblib")

    print("\nRandom Forest Results")
    print("---------------------")
    print(json.dumps(metrics, indent=4))
    print(f"\nResults saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
