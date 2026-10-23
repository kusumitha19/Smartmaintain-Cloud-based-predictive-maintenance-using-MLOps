
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = Path("data/CMAPSS")
OUTPUT_DIR = Path("experiment_results/drift_monitoring")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_COLUMNS = (
    ["unit_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

FEATURE_COLUMNS = RAW_COLUMNS[1:]


# --------------------------------------------------
# Load sensor data
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

    train = train.iloc[:, :26].copy()
    test = test.iloc[:, :26].copy()

    train.columns = RAW_COLUMNS
    test.columns = RAW_COLUMNS

    return train, test


# --------------------------------------------------
# Run drift report
# --------------------------------------------------

def main():
    print("Loading NASA C-MAPSS FD001 data...")
    train, test = load_data()

    # Reference: training sensor observations.
    reference = train[FEATURE_COLUMNS].copy()

    # Current: official test sensor observations.
    current = test[FEATURE_COLUMNS].copy()

    print("Reference rows:", len(reference))
    print("Current rows:", len(current))
    print("Features monitored:", len(FEATURE_COLUMNS))

    report = Report([
        DataDriftPreset(),
    ])

    print("Generating Evidently drift report...")
    snapshot = report.run(
        reference_data=reference,
        current_data=current,
    )

    report_path = OUTPUT_DIR / "data_drift_report.html"
    snapshot.save_html(str(report_path))

    print("\nDrift analysis completed.")
    print("HTML report:", report_path.resolve())


if __name__ == "__main__":
    main()
