import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from machine_predictive_maintenance.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from machine_predictive_maintenance.entity.config_entity import DataValidationConfig
from machine_predictive_maintenance.exception.exception import MachinePredictiveMaintenanceException
from machine_predictive_maintenance.logging.logger import logging
from machine_predictive_maintenance.utils.main_utils.utils import write_yaml_file


CMAPSS_COLUMNS = [
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
    "RUL",
]

DRIFT_COLUMNS = [
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    *[f"sensor_{index}" for index in range(1, 22)],
]


class DataValidation:
    """Validate C-MAPSS FD001 data prepared for RUL prediction."""

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    @staticmethod
    def validate_columns(dataframe: pd.DataFrame, dataset_name: str) -> list[str]:
        """Require the C-MAPSS columns and their expected order."""
        try:
            if dataframe.columns.tolist() == CMAPSS_COLUMNS:
                return []
            return [
                f"{dataset_name} columns must exactly match the expected C-MAPSS column names and order."
            ]
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    @staticmethod
    def validate_numeric_data(dataframe: pd.DataFrame, dataset_name: str) -> list[str]:
        """Ensure every C-MAPSS field is represented as numeric data."""
        try:
            non_numeric_columns = [
                column
                for column in CMAPSS_COLUMNS
                if column in dataframe.columns
                and not pd.api.types.is_numeric_dtype(dataframe[column])
            ]
            if non_numeric_columns:
                return [
                    f"{dataset_name} contains non-numeric columns: {non_numeric_columns}."
                ]
            return []
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    @staticmethod
    def validate_common_values(dataframe: pd.DataFrame, dataset_name: str) -> list[str]:
        """Validate required non-null C-MAPSS fields and engine/cycle values."""
        try:
            errors = []
            required_columns = [column for column in CMAPSS_COLUMNS if column != "RUL"]
            missing_columns = [
                column for column in required_columns if dataframe[column].isna().any()
            ]
            if missing_columns:
                errors.append(
                    f"{dataset_name} contains missing values in required columns: {missing_columns}."
                )

            if not dataframe.empty:
                unit_ids = dataframe["unit_id"]
                if (unit_ids <= 0).any() or not np.equal(unit_ids % 1, 0).all():
                    errors.append(f"{dataset_name} contains invalid unit_id values.")

                cycles = dataframe["cycle"]
                if (cycles <= 0).any() or not np.equal(cycles % 1, 0).all():
                    errors.append(f"{dataset_name} contains invalid cycle values.")

            return errors
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    def validate_training_data(self, dataframe: pd.DataFrame) -> list[str]:
        try:
            errors = self.validate_columns(dataframe, "Training data")
            if errors:
                return errors

            errors.extend(self.validate_numeric_data(dataframe, "Training data"))
            if errors:
                return errors
            errors.extend(self.validate_common_values(dataframe, "Training data"))
            if dataframe["RUL"].isna().any():
                errors.append("Training data contains missing RUL values.")
            elif (dataframe["RUL"] < 0).any():
                errors.append("Training data contains negative RUL values.")
            return errors
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    def validate_test_data(self, dataframe: pd.DataFrame) -> list[str]:
        try:
            errors = self.validate_columns(dataframe, "Test data")
            if errors:
                return errors

            errors.extend(self.validate_numeric_data(dataframe, "Test data"))
            if errors:
                return errors
            errors.extend(self.validate_common_values(dataframe, "Test data"))

            final_cycles = dataframe.groupby("unit_id")["cycle"].transform("max")
            is_final_cycle = dataframe["cycle"].eq(final_cycles)
            if dataframe.loc[is_final_cycle, "RUL"].isna().any():
                errors.append("Test data is missing RUL values on one or more final engine cycles.")
            if dataframe.loc[~is_final_cycle, "RUL"].notna().any():
                errors.append("Test data contains RUL values before an engine's final cycle.")

            populated_rul = dataframe.loc[dataframe["RUL"].notna(), "RUL"]
            if (populated_rul < 0).any():
                errors.append("Test data contains negative RUL values.")
            return errors
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    def detect_dataset_drift(
        self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05
    ) -> bool:
        """Run KS drift checks on operating settings and sensor values only."""
        try:
            status = True
            report = {}

            for column in DRIFT_COLUMNS:
                train_values = base_df[column].dropna()
                test_values = current_df[column].dropna()
                result = ks_2samp(train_values, test_values)
                drift_detected = result.pvalue < threshold
                status = status and not drift_detected
                report[column] = {
                    "p_value": float(result.pvalue),
                    "drift_status": bool(drift_detected),
                }

            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content=report,
            )
            logging.info("Saved C-MAPSS drift report to %s", self.data_validation_config.drift_report_file_path)
            return status
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting C-MAPSS data validation")
            train_dataframe = self.read_data(
                self.data_ingestion_artifact.trained_file_path
            )
            test_dataframe = self.read_data(self.data_ingestion_artifact.test_file_path)

            validation_errors = self.validate_training_data(train_dataframe)
            validation_errors.extend(self.validate_test_data(test_dataframe))
            if validation_errors:
                raise ValueError("\n".join(validation_errors))

            drift_status = self.detect_dataset_drift(train_dataframe, test_dataframe)
            logging.info("C-MAPSS sensor drift status: %s", drift_status)

            os.makedirs(
                os.path.dirname(self.data_validation_config.valid_train_file_path),
                exist_ok=True,
            )
            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header=True
            )
            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )
            logging.info("Saved validated C-MAPSS train and test datasets")

            return DataValidationArtifact(
                validation_status=True,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)
