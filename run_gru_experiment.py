"""Train an isolated GRU comparison experiment for SMARTMAINTAIN FD001.

This script intentionally reads the retained 30x25 pipeline payloads and never
writes to final_model or to the production LSTM artifacts.
"""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
PRODUCTION_MODEL = ROOT / "final_model" / "model.keras"
PRODUCTION_SCALER = ROOT / "final_model" / "preprocessing.pkl"
PIPELINE_ARTIFACT = ROOT / "Artifacts" / "09_22_2026_20_20_32"
TRAIN_PAYLOAD = PIPELINE_ARTIFACT / "data_transformation" / "transformed" / "train.npy"
TEST_PAYLOAD = PIPELINE_ARTIFACT / "data_transformation" / "transformed" / "test.npy"
TRAIN_DATA = PIPELINE_ARTIFACT / "data_ingestion" / "ingested" / "train.csv"
TEST_DATA = PIPELINE_ARTIFACT / "data_ingestion" / "ingested" / "test.csv"
RESULTS_DIR = ROOT / "experiment_results" / "gru"

RUL_CAP = 125
WINDOW_SIZE = 30
FEATURE_COUNT = 25


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_payload(path: Path) -> dict:
    payload = np.load(path, allow_pickle=True)
    return payload.item() if hasattr(payload, "item") else payload


def capped_training_targets(train_dataframe: pd.DataFrame) -> np.ndarray:
    """Produce cap-125 labels in the retained stride-1 sequence order."""
    targets: list[float] = []
    dataframe = train_dataframe.sort_values(["unit_id", "cycle"])
    for _, unit_dataframe in dataframe.groupby("unit_id", sort=False):
        rul = unit_dataframe["RUL"].clip(upper=RUL_CAP).to_numpy()
        for end_index in range(WINDOW_SIZE, len(unit_dataframe) + 1):
            targets.append(rul[end_index - 1])
    return np.asarray(targets, dtype=np.float32)


def build_gru() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(WINDOW_SIZE, FEATURE_COUNT)),
            tf.keras.layers.GRU(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GRU(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def metrics(y_true: np.ndarray, y_predicted: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_predicted)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_predicted))),
        "r2": float(r2_score(y_true, y_predicted)),
    }


def main() -> None:
    model_hash_before = sha256(PRODUCTION_MODEL)
    scaler_hash_before = sha256(PRODUCTION_SCALER)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_payload = load_payload(TRAIN_PAYLOAD)
    test_payload = load_payload(TEST_PAYLOAD)
    x_train = train_payload["X"]
    x_test = test_payload["X"]
    y_train = capped_training_targets(pd.read_csv(TRAIN_DATA))
    y_test = test_payload["y"].astype(np.float32)

    if x_train.shape != (len(y_train), WINDOW_SIZE, FEATURE_COUNT):
        raise ValueError(f"Unexpected training shape: {x_train.shape}, labels: {y_train.shape}")
    if x_test.shape != (100, WINDOW_SIZE, FEATURE_COUNT):
        raise ValueError(f"Unexpected test shape: {x_test.shape}")

    model = build_gru()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=3, factor=0.5
        ),
    ]
    history = model.fit(
        x_train,
        y_train,
        epochs=30,
        batch_size=64,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=2,
    )

    train_predictions = model.predict(x_train, verbose=0).ravel()
    test_predictions = model.predict(x_test, verbose=0).ravel()
    train_metrics = metrics(y_train, train_predictions)
    test_metrics = metrics(y_test, test_predictions)
    model.save(RESULTS_DIR / "gru_model.keras")

    test_data = pd.read_csv(TEST_DATA).sort_values(["unit_id", "cycle"])
    unit_ids = test_data.groupby("unit_id", sort=False)["unit_id"].last().to_numpy()
    predictions = pd.DataFrame(
        {
            "unit_id": unit_ids,
            "actual_RUL": y_test,
            "predicted_RUL": test_predictions,
        }
    )
    predictions["signed_error"] = predictions["predicted_RUL"] - predictions["actual_RUL"]
    predictions["absolute_error"] = predictions["signed_error"].abs()
    predictions.to_csv(RESULTS_DIR / "gru_test_predictions.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, test_predictions, alpha=0.75)
    bounds = [min(y_test.min(), test_predictions.min()), max(y_test.max(), test_predictions.max())]
    plt.plot(bounds, bounds, "r--", label="Ideal")
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title("GRU: Actual vs Predicted RUL (FD001)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "gru_actual_vs_predicted_rul.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(predictions["signed_error"], bins=20, edgecolor="black")
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("Prediction Error (predicted - actual RUL)")
    plt.ylabel("Number of Engines")
    plt.title("GRU: Prediction Error Distribution (FD001)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "gru_prediction_error_distribution.png", dpi=150)
    plt.close()

    mlflow.set_experiment("SMARTMAINTAIN_GRU_FD001")
    with mlflow.start_run(run_name="gru_fd001_cap125_window30") as run:
        mlflow.log_params(
            {
                "model_type": "GRU",
                "dataset": "C-MAPSS FD001",
                "rul_cap": RUL_CAP,
                "window_size": WINDOW_SIZE,
                "feature_count": FEATURE_COUNT,
                "gru_units_1": 64,
                "gru_units_2": 32,
                "dropout_rate": 0.2,
                "dense_units": 32,
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "loss": "mse",
                "metric": "mae",
                "epochs_requested": 30,
                "batch_size": 64,
                "validation_split": 0.2,
                "early_stopping_patience": 5,
                "reduce_lr_patience": 3,
                "reduce_lr_factor": 0.5,
            }
        )
        mlflow.log_metrics(
            {
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
            }
        )
        mlflow.tensorflow.log_model(model, artifact_path="gru_model")
        run_id = run.info.run_id

    summary = {
        "experiment": "GRU_FD001_cap125_window30_25features",
        "model_type": "GRU",
        "dataset": "C-MAPSS FD001",
        "rul_cap": RUL_CAP,
        "window_size": WINDOW_SIZE,
        "feature_count": FEATURE_COUNT,
        "training_sequence_count": int(len(y_train)),
        "test_sequence_shape": list(x_test.shape),
        "test_engine_count": int(len(y_test)),
        "epochs_completed": len(history.history["loss"]),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "mlflow_experiment": "SMARTMAINTAIN_GRU_FD001",
        "mlflow_run_id": run_id,
        "production_lstm_model_sha256_before": model_hash_before,
        "production_lstm_model_sha256_after": sha256(PRODUCTION_MODEL),
        "production_scaler_sha256_before": scaler_hash_before,
        "production_scaler_sha256_after": sha256(PRODUCTION_SCALER),
    }
    if (
        summary["production_lstm_model_sha256_before"]
        != summary["production_lstm_model_sha256_after"]
        or summary["production_scaler_sha256_before"]
        != summary["production_scaler_sha256_after"]
    ):
        raise RuntimeError("A frozen production artifact changed during the GRU experiment.")
    (RESULTS_DIR / "gru_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
