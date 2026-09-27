import os
import sys

import numpy as np
import pandas as pd

from machine_predictive_maintenance.constant.training_pipeline import RUL_CAP
from machine_predictive_maintenance.entity.artifact_entity import DataIngestionArtifact
from machine_predictive_maintenance.entity.config_entity import DataIngestionConfig
from machine_predictive_maintenance.exception.exception import MachinePredictiveMaintenanceException
from machine_predictive_maintenance.logging.logger import logging


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
]


class DataIngestion:
    """Load and prepare the NASA C-MAPSS FD001 data for RUL prediction."""

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    @staticmethod
    def read_cmapss_data(file_path: str) -> pd.DataFrame:
        """Read a whitespace-separated C-MAPSS train or test file without a header."""
        try:
            dataframe = pd.read_csv(
                file_path,
                sep=r"\s+",
                header=None,
                names=CMAPSS_COLUMNS,
            )
            logging.info("Loaded C-MAPSS data from %s", file_path)
            return dataframe
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    @staticmethod
    def read_rul_data(file_path: str) -> pd.Series:
        """Read the provided FD001 test-engine RUL values in unit-id order."""
        try:
            rul_values = pd.read_csv(file_path, sep=r"\s+", header=None).iloc[:, 0]
            logging.info("Loaded C-MAPSS test RUL values from %s", file_path)
            return rul_values
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    @staticmethod
    def add_training_rul(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Calculate capped RUL for each training row within its engine trajectory."""
        try:
            prepared_dataframe = dataframe.copy()
            max_cycle = prepared_dataframe.groupby("unit_id")["cycle"].transform("max")
            prepared_dataframe["RUL"] = (max_cycle - prepared_dataframe["cycle"]).clip(upper=RUL_CAP)
            logging.info("Calculated capped training RUL with cap=%s", RUL_CAP)
            return prepared_dataframe
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    @staticmethod
    def add_test_final_cycle_rul(dataframe: pd.DataFrame, rul_values: pd.Series) -> pd.DataFrame:
        """Store supplied RUL only on each test engine's final observed cycle."""
        try:
            prepared_dataframe = dataframe.copy()
            unit_ids = prepared_dataframe["unit_id"].drop_duplicates().sort_values().to_numpy()

            if len(unit_ids) != len(rul_values):
                raise ValueError(
                    "The number of test engine units does not match the number of supplied RUL values."
                )

            rul_by_unit = dict(zip(unit_ids, rul_values.to_numpy()))
            final_cycles = prepared_dataframe.groupby("unit_id")["cycle"].transform("max")
            prepared_dataframe["RUL"] = np.nan
            is_final_cycle = prepared_dataframe["cycle"].eq(final_cycles)
            prepared_dataframe.loc[is_final_cycle, "RUL"] = prepared_dataframe.loc[
                is_final_cycle, "unit_id"
            ].map(rul_by_unit)
            logging.info("Attached supplied RUL values to final test-engine cycles")
            return prepared_dataframe
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    @staticmethod
    def save_dataframe(dataframe: pd.DataFrame, file_path: str) -> None:
        """Save a prepared C-MAPSS dataframe as CSV, creating its parent directory."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            dataframe.to_csv(file_path, index=False, header=True)
            logging.info("Saved prepared data to %s", file_path)
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """Load C-MAPSS FD001 files without splitting engine trajectories."""
        try:
            train_dataframe = self.read_cmapss_data(
                self.data_ingestion_config.raw_train_file_path
            )
            train_dataframe = self.add_training_rul(train_dataframe)
            self.save_dataframe(
                train_dataframe, self.data_ingestion_config.feature_store_file_path
            )
            self.save_dataframe(
                train_dataframe, self.data_ingestion_config.training_file_path
            )

            test_dataframe = self.read_cmapss_data(
                self.data_ingestion_config.raw_test_file_path
            )
            test_rul_values = self.read_rul_data(
                self.data_ingestion_config.raw_rul_file_path
            )
            test_dataframe = self.add_test_final_cycle_rul(test_dataframe, test_rul_values)
            self.save_dataframe(
                test_dataframe, self.data_ingestion_config.testing_file_path
            )

            return DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)
