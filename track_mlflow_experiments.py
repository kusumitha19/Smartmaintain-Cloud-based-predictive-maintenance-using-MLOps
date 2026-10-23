
import mlflow
from pathlib import Path

# Keep MLflow experiment data inside the project.
TRACKING_DIR = Path("mlruns").resolve()
TRACKING_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(TRACKING_DIR.as_uri())
mlflow.set_experiment("SmartMaintain-Model-Comparison")

# Previously recorded evaluation results.
# These are existing results, not new training runs.
experiments = {
    "LSTM": {
        "MAE": 11.3076,
        "RMSE": 14.7436,
        "R2": 0.8741,
    },
    "GRU": {
        "MAE": 11.7537,
        "RMSE": 16.5598,
        "R2": 0.8412,
    },
    "CNN-LSTM": {
        "MAE": 12.2434,
        "RMSE": 17.0503,
        "R2": 0.8317,
    },
    "Random-Forest": {
        "MAE": 12.6016482238,
        "RMSE": 16.5087827626,
        "R2": 0.8302849343,
    },
    "BiLSTM": {
        "MAE": 11.6802453995,
        "RMSE": 13.5290237957,
        "R2": 0.8860214353,
    },
}

for model_name, metrics in experiments.items():
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)
        mlflow.log_param("dataset", "NASA-C-MAPSS-FD001")
        mlflow.log_param("window_size", 30)
        mlflow.log_param("rul_cap", 125)
        mlflow.log_param("test_engine_count", 100)

        mlflow.log_metrics(metrics)

        print(f"Logged {model_name}: {metrics}")

print("\nAll existing evaluation metrics logged to MLflow.")
print("Tracking directory:", TRACKING_DIR)
print("Experiment: SmartMaintain-Model-Comparison")
