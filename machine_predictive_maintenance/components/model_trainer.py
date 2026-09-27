import json
import os
import sys

import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from machine_predictive_maintenance.components.data_transformation import FEATURE_COLUMNS
from machine_predictive_maintenance.constant.training_pipeline import (
    MLFLOW_EXPERIMENT_NAME,
    MODEL_FILE_NAME,
    SAVED_MODEL_DIR,
    VALIDATION_RATIO,
    WINDOW_SIZE,
)
from machine_predictive_maintenance.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    RegressionMetricArtifact,
)
from machine_predictive_maintenance.entity.config_entity import ModelTrainerConfig
from machine_predictive_maintenance.exception.exception import MachinePredictiveMaintenanceException
from machine_predictive_maintenance.logging.logger import logging
from machine_predictive_maintenance.utils.main_utils.utils import load_object

class ModelTrainer:
    """Train and evaluate an LSTM regressor for C-MAPSS RUL prediction."""

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
            self.training_history_file_path = os.path.join(
                self.model_trainer_config.model_trainer_dir, "training_history.json"
            )
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    @staticmethod
    def build_model(model_trainer_config: ModelTrainerConfig) -> tf.keras.Model:
        """Build the LSTM regressor for 30-cycle, 25-feature C-MAPSS windows."""
        try:
            feature_count = len(FEATURE_COLUMNS)
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(WINDOW_SIZE, feature_count)),
                    tf.keras.layers.LSTM(
                        model_trainer_config.lstm_units, return_sequences=True
                    ),
                    tf.keras.layers.Dropout(model_trainer_config.dropout_rate),
                    tf.keras.layers.LSTM(model_trainer_config.lstm_units // 2),
                    tf.keras.layers.Dropout(model_trainer_config.dropout_rate),
                    tf.keras.layers.Dense(32, activation="relu"),
                    tf.keras.layers.Dense(1),
                ]
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=model_trainer_config.learning_rate
                ),
                loss="mse",
                metrics=["mae"],
            )
            return model
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    @staticmethod
    def calculate_regression_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> RegressionMetricArtifact:
        try:
            return RegressionMetricArtifact(
                mae=float(mean_absolute_error(y_true, y_pred)),
                rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
                r2=float(r2_score(y_true, y_pred)),
            )
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    @staticmethod
    def validate_input_shape(x_data: np.ndarray, dataset_name: str) -> None:
        try:
            expected_shape = (WINDOW_SIZE, len(FEATURE_COLUMNS))
            if x_data.ndim != 3 or x_data.shape[1:] != expected_shape:
                raise ValueError(
                    f"{dataset_name} must have shape (samples, {WINDOW_SIZE}, "
                    f"{len(FEATURE_COLUMNS)}); received {x_data.shape}."
                )
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    def save_training_history(self, history: tf.keras.callbacks.History) -> None:
        try:
            serializable_history = {
                metric_name: [float(value) for value in values]
                for metric_name, values in history.history.items()
            }
            os.makedirs(os.path.dirname(self.training_history_file_path), exist_ok=True)
            with open(self.training_history_file_path, "w", encoding="utf-8") as history_file:
                json.dump(serializable_history, history_file, indent=2)
            logging.info("Saved training history to %s", self.training_history_file_path)
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    def track_mlflow(
        self,
        model: tf.keras.Model,
        train_metrics: RegressionMetricArtifact,
        test_metrics: RegressionMetricArtifact,
    ) -> None:
        try:
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            with mlflow.start_run():
                mlflow.log_params(
                    {
                        "epochs": self.model_trainer_config.epochs,
                        "batch_size": self.model_trainer_config.batch_size,
                        "learning_rate": self.model_trainer_config.learning_rate,
                        "lstm_units": self.model_trainer_config.lstm_units,
                        "dropout_rate": self.model_trainer_config.dropout_rate,
                    }
                )
                mlflow.log_metrics(
                    {
                        "train_mae": train_metrics.mae,
                        "train_rmse": train_metrics.rmse,
                        "train_r2": train_metrics.r2,
                        "test_mae": test_metrics.mae,
                        "test_rmse": test_metrics.rmse,
                        "test_r2": test_metrics.r2,
                    }
                )
                mlflow.tensorflow.log_model(model, artifact_path="model")
                logging.info("Logged LSTM model, parameters, and metrics to MLflow")
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    def train_model(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> ModelTrainerArtifact:
        try:
            self.validate_input_shape(x_train, "X_train")
            self.validate_input_shape(x_test, "X_test")

            model = self.build_model(self.model_trainer_config)
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
                epochs=self.model_trainer_config.epochs,
                batch_size=self.model_trainer_config.batch_size,
                validation_split=VALIDATION_RATIO,
                callbacks=callbacks,
                verbose=1,
            )
            self.save_training_history(history)

            train_predictions = model.predict(x_train, verbose=0).ravel()
            test_predictions = model.predict(x_test, verbose=0).ravel()
            train_metrics = self.calculate_regression_metrics(y_train, train_predictions)
            test_metrics = self.calculate_regression_metrics(y_test, test_predictions)

            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_file_path),
                exist_ok=True,
            )
            model.save(self.model_trainer_config.trained_model_file_path)
            os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
            model.save(os.path.join(SAVED_MODEL_DIR, MODEL_FILE_NAME))
            logging.info("Saved trained LSTM model to configured and final-model paths")

            self.track_mlflow(model, train_metrics, test_metrics)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metrics,
                test_metric_artifact=test_metrics,
            )
            logging.info("Model trainer artifact: %s", model_trainer_artifact)
            return model_trainer_artifact
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_payload = load_object(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_payload = load_object(
                self.data_transformation_artifact.transformed_test_file_path
            )
            if not {"X", "y"}.issubset(train_payload) or not {"X", "y"}.issubset(
                test_payload
            ):
                raise ValueError("Transformed sequence files must contain 'X' and 'y' keys.")

            return self.train_model(
                train_payload["X"],
                train_payload["y"],
                test_payload["X"],
                test_payload["y"],
            )
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)
