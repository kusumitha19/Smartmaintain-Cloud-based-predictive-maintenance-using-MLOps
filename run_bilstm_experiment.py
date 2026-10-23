
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Bidirectional,
    LSTM,
    Dropout,
    Dense,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

SEED = 42
WINDOW_SIZE = 30
RUL_CAP = 125
BATCH_SIZE = 128
EPOCHS = 50

DATA_DIR = Path("data/CMAPSS")
OUTPUT_DIR = Path("experiment_results/bilstm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

RAW_COLUMNS = (
    ["unit_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

FEATURE_COLUMNS = RAW_COLUMNS[1:]


# --------------------------------------------------
# 2. Load FD001 data
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

    train = train.iloc[:, :26].copy()
    test = test.iloc[:, :26].copy()

    train.columns = RAW_COLUMNS
    test.columns = RAW_COLUMNS

    return train, test, rul.iloc[:, 0].to_numpy(dtype=np.float32)


# --------------------------------------------------
# 3. Add capped RUL labels to training data
# --------------------------------------------------

def add_training_rul(train):
    train = train.copy()

    last_cycle = train.groupby("unit_id")["cycle"].max()
    train["RUL"] = (
        train["unit_id"].map(last_cycle) - train["cycle"]
    ).clip(upper=RUL_CAP)

    return train


# --------------------------------------------------
# 4. Fit scaler on training data only
# --------------------------------------------------

def scale_data(train, test):
    scaler = MinMaxScaler()

    train.loc[:, FEATURE_COLUMNS] = scaler.fit_transform(
        train[FEATURE_COLUMNS]
    )

    test.loc[:, FEATURE_COLUMNS] = scaler.transform(
        test[FEATURE_COLUMNS]
    )

    return train, test, scaler


# --------------------------------------------------
# 5. Create training sequences
# --------------------------------------------------

def make_training_sequences(train):
    X, y = [], []

    for _, engine in train.groupby("unit_id", sort=True):
        engine = engine.sort_values("cycle")

        values = engine[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        labels = engine["RUL"].to_numpy(dtype=np.float32)

        for end in range(WINDOW_SIZE, len(engine) + 1):
            X.append(values[end - WINDOW_SIZE:end])
            y.append(labels[end - 1])

    return np.asarray(X, dtype=np.float32), np.asarray(
        y, dtype=np.float32
    )


# --------------------------------------------------
# 6. Create one test sequence per engine
# --------------------------------------------------

def make_test_sequences(test, rul_values):
    X, y, engine_ids, latest_cycles = [], [], [], []

    engines = sorted(test["unit_id"].unique())

    if len(engines) != len(rul_values):
        raise ValueError(
            "Test engine count does not match RUL label count."
        )

    for engine_id in engines:
        engine = test[test["unit_id"] == engine_id].sort_values("cycle")

        if len(engine) < WINDOW_SIZE:
            raise ValueError(
                f"Engine {engine_id} has fewer than "
                f"{WINDOW_SIZE} cycles."
            )

        sequence = engine[FEATURE_COLUMNS].tail(WINDOW_SIZE).to_numpy(
            dtype=np.float32
        )

        X.append(sequence)
        y.append(min(float(rul_values[int(engine_id) - 1]), RUL_CAP))
        engine_ids.append(int(engine_id))
        latest_cycles.append(int(engine["cycle"].iloc[-1]))

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        engine_ids,
        latest_cycles,
    )


# --------------------------------------------------
# 7. Build BiLSTM model
# --------------------------------------------------

def build_model():
    model = Sequential([
        Input(shape=(WINDOW_SIZE, len(FEATURE_COLUMNS))),

        Bidirectional(
            LSTM(64, return_sequences=True)
        ),
        Dropout(0.2),

        Bidirectional(LSTM(32)),
        Dropout(0.2),

        Dense(32, activation="relu"),
        Dense(1),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )

    return model


# --------------------------------------------------
# 8. Train and evaluate
# --------------------------------------------------

def main():
    print("Loading NASA C-MAPSS FD001...")
    train, test, rul_values = load_data()

    train = add_training_rul(train)

    # Match the feature scaling approach used by the
    # production LSTM: fit only on training data.
    train, test, scaler = scale_data(train, test)

    print("Creating training sequences...")
    X_train, y_train = make_training_sequences(train)

    print("Creating test sequences...")
    X_test, y_test, engine_ids, latest_cycles = make_test_sequences(
        test, rul_values
    )

    print("Training sequences:", len(X_train))
    print("Test engines:", len(X_test))
    print("Input shape:", X_train.shape[1:])

    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("Training BiLSTM...")
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    print("Evaluating on the 100 test engines...")
    predictions = model.predict(X_test, verbose=0).reshape(-1)
    predictions = np.clip(predictions, 0, None)

    metrics = {
        "model": "BiLSTM",
        "dataset": "NASA C-MAPSS FD001",
        "test_engine_count": int(len(y_test)),
        "training_window_count": int(len(X_train)),
        "window_size": WINDOW_SIZE,
        "rul_cap": RUL_CAP,
        "feature_count": len(FEATURE_COLUMNS),
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2": float(r2_score(y_test, predictions)),
        "epochs_trained": len(history.history["loss"]),
    }

    predictions_df = pd.DataFrame({
        "engine_id": engine_ids,
        "latest_cycle": latest_cycles,
        "actual_rul": y_test,
        "predicted_rul": predictions,
    })

    predictions_df.to_csv(
        OUTPUT_DIR / "bilstm_test_predictions.csv",
        index=False,
    )

    with open(OUTPUT_DIR / "bilstm_metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    model.save(OUTPUT_DIR / "bilstm_model.keras")

    # Save the scaler for reproducibility of this experiment.
    import joblib
    joblib.dump(scaler, OUTPUT_DIR / "bilstm_scaler.joblib")

    print("\nBiLSTM Results")
    print("--------------")
    print(json.dumps(metrics, indent=4))
    print("\nResults saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
