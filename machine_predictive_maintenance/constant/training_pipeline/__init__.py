import os

# =========================
# General Pipeline Constants
# =========================

TARGET_COLUMN = "RUL"

PIPELINE_NAME: str = "SMARTMAINTAIN"

ARTIFACT_DIR: str = "Artifacts"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

SAVED_MODEL_DIR = os.path.join("final_model")

MODEL_FILE_NAME = "model.keras"


# =========================
# C-MAPSS Dataset Constants
# =========================

CMAPSS_DATA_DIR: str = os.path.join("data", "CMAPSS")

CMAPSS_DATASET: str = "FD001"

CMAPSS_TRAIN_FILE: str = "train_FD001.txt"

CMAPSS_TEST_FILE: str = "test_FD001.txt"

CMAPSS_RUL_FILE: str = "RUL_FD001.txt"


# =========================
# Data Ingestion
# =========================

DATA_INGESTION_DIR_NAME: str = "data_ingestion"

DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

DATA_INGESTION_INGESTED_DIR: str = "ingested"

TRAIN_FILE_NAME: str = "train.csv"

TEST_FILE_NAME: str = "test.csv"


# =========================
# Data Validation
# =========================

DATA_VALIDATION_DIR_NAME: str = "data_validation"

DATA_VALIDATION_VALID_DIR: str = "validated"

DATA_VALIDATION_INVALID_DIR: str = "invalid"

DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"

DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"


# =========================
# Data Transformation
# =========================

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"

DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"

DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"

DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"

DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"


# =========================
# RUL / LSTM Configuration
# =========================

RUL_CAP: int = 125

WINDOW_SIZE: int = 30

WINDOW_STRIDE: int = 1

VALIDATION_RATIO: float = 0.2

RANDOM_STATE: int = 42


# =========================
# LSTM Model Configuration
# =========================

MODEL_TRAINER_DIR_NAME: str = "model_trainer"

MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"

MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.keras"

LSTM_UNITS: int = 64

DROPOUT_RATE: float = 0.2

LEARNING_RATE: float = 0.001

EPOCHS: int = 30

BATCH_SIZE: int = 64


# =========================
# MLflow
# =========================

MLFLOW_EXPERIMENT_NAME: str = "SMARTMAINTAIN_RUL_FD001"
