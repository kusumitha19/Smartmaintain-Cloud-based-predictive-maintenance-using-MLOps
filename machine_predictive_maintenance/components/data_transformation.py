import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from machine_predictive_maintenance.constant.training_pipeline import (
    WINDOW_SIZE,
    WINDOW_STRIDE,
)
from machine_predictive_maintenance.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from machine_predictive_maintenance.entity.config_entity import DataTransformationConfig
from machine_predictive_maintenance.exception.exception import MachinePredictiveMaintenanceException
from machine_predictive_maintenance.logging.logger import logging
from machine_predictive_maintenance.utils.main_utils.utils import save_object


# Keep feature selection centralized so low-variance sensors can be removed later.
FEATURE_COLUMNS = [
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]


class DataTransformation:
    """Create scaled, per-engine C-MAPSS sequences for LSTM RUL prediction."""

    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    @staticmethod
    def sort_by_engine_cycle(dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            return dataframe.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    @staticmethod
    def create_training_sequences(
        dataframe: pd.DataFrame, scaled_features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create stride-based windows without crossing engine boundaries."""
        try:
            sequences = []
            targets = []

            for _, unit_dataframe in dataframe.groupby("unit_id", sort=False):
                unit_features = scaled_features[unit_dataframe.index]
                unit_rul = unit_dataframe["RUL"].to_numpy()

                for start_index in range(
                    0, len(unit_dataframe) - WINDOW_SIZE + 1, WINDOW_STRIDE
                ):
                    end_index = start_index + WINDOW_SIZE
                    sequences.append(unit_features[start_index:end_index])
                    targets.append(unit_rul[end_index - 1])

            if not sequences:
                raise ValueError(
                    f"No training windows could be created with WINDOW_SIZE={WINDOW_SIZE}."
                )

            return np.stack(sequences), np.asarray(targets)
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    @staticmethod
    def create_test_sequences(
        dataframe: pd.DataFrame, scaled_features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create one final window and target for each C-MAPSS test engine."""
        try:
            sequences = []
            targets = []

            for unit_id, unit_dataframe in dataframe.groupby("unit_id", sort=False):
                if len(unit_dataframe) < WINDOW_SIZE:
                    raise ValueError(
                        f"Test engine {unit_id} has {len(unit_dataframe)} cycles; "
                        f"at least WINDOW_SIZE={WINDOW_SIZE} cycles are required."
                    )

                final_rul = unit_dataframe["RUL"].iloc[-1]
                if pd.isna(final_rul):
                    raise ValueError(
                        f"Test engine {unit_id} is missing RUL on its final cycle."
                    )

                unit_features = scaled_features[unit_dataframe.index]
                sequences.append(unit_features[-WINDOW_SIZE:])
                targets.append(final_rul)

            if not sequences:
                raise ValueError("No test sequences were created.")

            return np.stack(sequences), np.asarray(targets)
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting C-MAPSS LSTM data transformation")
            train_dataframe = self.sort_by_engine_cycle(
                self.read_data(self.data_validation_artifact.valid_train_file_path)
            )
            test_dataframe = self.sort_by_engine_cycle(
                self.read_data(self.data_validation_artifact.valid_test_file_path)
            )

            scaler = StandardScaler()
            scaled_train_features = scaler.fit_transform(train_dataframe[FEATURE_COLUMNS])
            scaled_test_features = scaler.transform(test_dataframe[FEATURE_COLUMNS])
            logging.info("Fitted StandardScaler on training features and transformed test features")

            x_train, y_train = self.create_training_sequences(
                train_dataframe, scaled_train_features
            )
            x_test, y_test = self.create_test_sequences(
                test_dataframe, scaled_test_features
            )

            save_object(
                self.data_transformation_config.transformed_train_file_path,
                {"X": x_train, "y": y_train},
            )
            save_object(
                self.data_transformation_config.transformed_test_file_path,
                {"X": x_test, "y": y_test},
            )
            save_object(
                self.data_transformation_config.transformed_object_file_path, scaler
            )
            logging.info(
                "Saved C-MAPSS sequence data: X_train=%s, y_train=%s, X_test=%s, y_test=%s",
                x_train.shape,
                y_train.shape,
                x_test.shape,
                y_test.shape,
            )

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)
