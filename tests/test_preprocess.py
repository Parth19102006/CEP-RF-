from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocess import (
    CONSTANT_FEATURES_TO_REMOVE,
    TARGET_COLUMN,
    apply_numeric_cleaning,
    discover_raw_files,
    fit_training_medians,
    validate_schema,
)


class PreprocessTests(unittest.TestCase):
    def make_frame(self) -> pd.DataFrame:
        data = {
            "Protocol": [6, 17, 6],
            "Flow Duration": [10, 20, 30],
            TARGET_COLUMN: ["Benign", "DrDoS_DNS", "BENIGN"],
        }
        for column in CONSTANT_FEATURES_TO_REMOVE:
            data[column] = [0, 0, 0]
        return pd.DataFrame(data)

    def test_discover_raw_files_requires_known_train_test_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            self.make_frame().to_parquet(raw_dir / "LDAP-training.parquet", index=False)
            self.make_frame().to_parquet(raw_dir / "LDAP-testing.parquet", index=False)

            files = discover_raw_files(raw_dir)

        self.assertEqual(["testing", "training"], sorted(item.split for item in files))

    def test_validate_schema_removes_only_phase2_constants(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            self.make_frame().to_parquet(raw_dir / "LDAP-training.parquet", index=False)
            self.make_frame().to_parquet(raw_dir / "LDAP-testing.parquet", index=False)
            files = discover_raw_files(raw_dir)

            original_columns, feature_columns, _ = validate_schema(files)

        self.assertEqual(len(original_columns), 15)
        self.assertEqual(feature_columns, ["Protocol", "Flow Duration"])

    def test_imputation_uses_training_medians_and_encodes_binary_target(self) -> None:
        train_df = pd.DataFrame(
            {
                "Protocol": [6.0, 17.0, 6.0],
                "Flow Duration": [1.0, np.inf, np.nan],
                TARGET_COLUMN: ["Benign", "DrDoS_DNS", "Syn"],
            }
        )
        test_df = pd.DataFrame(
            {
                "Protocol": [17.0],
                "Flow Duration": [-np.inf],
                TARGET_COLUMN: ["Benign"],
            }
        )
        feature_columns = ["Protocol", "Flow Duration"]

        medians = fit_training_medians(train_df, feature_columns)
        cleaned = apply_numeric_cleaning(test_df, feature_columns, medians)

        self.assertEqual(medians["Flow Duration"], 1.0)
        self.assertEqual(cleaned.loc[0, "Flow Duration"], 1.0)
        self.assertEqual(cleaned.loc[0, "Label_binary"], 0)


if __name__ == "__main__":
    unittest.main()
