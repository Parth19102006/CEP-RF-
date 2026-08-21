"""Build Phase 3 ML-ready CIC-DDoS2019 preprocessing artifacts.

The pipeline preserves the dataset's existing train/test organization, removes
only the 12 globally constant Phase 2 features, keeps all remaining numeric
features including Protocol, and fits missing/non-finite value handling on the
training split only. Raw files under data/raw/ are never modified.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "phase3"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "preprocessing"

TARGET_COLUMN = "Label"
ENCODED_TARGET_COLUMN = "Label_binary"
SPLIT_PATTERN = re.compile(r"^(?P<family>.+)-(?P<split>training|testing)$", re.IGNORECASE)

CONSTANT_FEATURES_TO_REMOVE: tuple[str, ...] = (
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "FIN Flag Count",
    "PSH Flag Count",
    "ECE Flag Count",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
)


@dataclass(frozen=True)
class RawFile:
    """Raw Parquet file and its split from the dataset organization."""

    path: Path
    split: str
    attack_family_from_filename: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Create Phase 3 preprocessing artifacts for CIC-DDoS2019."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory containing raw CIC-DDoS2019 Parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for processed train/test Parquet files.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for preprocessing metadata and CSV reports.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    """Resolve relative paths from the project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    """Return a path relative to project root when possible."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def parse_dataset_name(path: Path) -> tuple[str, str]:
    """Infer file family and dataset split from the established filename pattern."""
    match = SPLIT_PATTERN.match(path.stem)
    if not match:
        return path.stem, "unknown"
    return match.group("family"), match.group("split").lower()


def discover_raw_files(raw_dir: Path) -> list[RawFile]:
    """Discover raw Parquet files and validate known train/test split names."""
    files: list[RawFile] = []
    for path in sorted(raw_dir.glob("*.parquet")):
        family, split = parse_dataset_name(path)
        if split not in {"training", "testing"}:
            raise ValueError(f"Cannot determine train/test split for {path.name}")
        files.append(RawFile(path=path, split=split, attack_family_from_filename=family))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {raw_dir}")
    if not any(item.split == "training" for item in files):
        raise ValueError("No training Parquet files were found.")
    if not any(item.split == "testing" for item in files):
        raise ValueError("No testing Parquet files were found.")
    return files


def read_schema(path: Path) -> list[tuple[str, str]]:
    """Read a Parquet schema as ordered column name/type pairs."""
    return [(field.name, str(field.type)) for field in pq.ParquetFile(path).schema_arrow]


def is_numeric_type(type_name: str) -> bool:
    """Return True for Arrow numeric type names used by the raw dataset."""
    return type_name.startswith(("int", "uint", "float", "double", "halffloat"))


def validate_schema(raw_files: list[RawFile]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Validate consistent columns and return original columns and selected features."""
    reference_schema = read_schema(raw_files[0].path)
    original_columns = [column for column, _ in reference_schema]
    reference_types = dict(reference_schema)
    schema_rows: list[dict[str, Any]] = []

    if TARGET_COLUMN not in original_columns:
        raise ValueError(f"Target column {TARGET_COLUMN!r} was not found.")

    missing_constants = [
        column for column in CONSTANT_FEATURES_TO_REMOVE if column not in original_columns
    ]
    if missing_constants:
        raise ValueError(
            "Expected Phase 2 constant features are missing: "
            + ", ".join(missing_constants)
        )

    for raw_file in raw_files:
        schema = read_schema(raw_file.path)
        columns = [column for column, _ in schema]
        type_by_column = dict(schema)
        missing_columns = [column for column in original_columns if column not in columns]
        extra_columns = [column for column in columns if column not in original_columns]
        type_mismatches = [
            column
            for column in original_columns
            if column in type_by_column and type_by_column[column] != reference_types[column]
        ]
        incompatible_type_mismatches = [
            column
            for column in type_mismatches
            if column != TARGET_COLUMN
            and not (
                is_numeric_type(reference_types[column])
                and is_numeric_type(type_by_column[column])
            )
        ]
        schema_rows.append(
            {
                "file_name": raw_file.path.name,
                "split": raw_file.split,
                "column_count": len(columns),
                "missing_columns": "; ".join(missing_columns),
                "extra_columns": "; ".join(extra_columns),
                "numeric_type_coercions": "; ".join(
                    column
                    for column in type_mismatches
                    if column not in incompatible_type_mismatches
                ),
                "incompatible_type_mismatches": "; ".join(incompatible_type_mismatches),
            }
        )
        if missing_columns or extra_columns or incompatible_type_mismatches:
            raise ValueError(f"Schema mismatch detected in {raw_file.path.name}")

    feature_columns = [
        column
        for column, type_name in reference_schema
        if column != TARGET_COLUMN
        and column not in CONSTANT_FEATURES_TO_REMOVE
        and is_numeric_type(type_name)
    ]
    if "Protocol" not in feature_columns:
        raise ValueError("Protocol must be retained as a numeric feature.")

    return original_columns, feature_columns, schema_rows


def count_missing_and_inf(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, int]:
    """Count NaN and positive/negative infinity in feature columns."""
    features = df[feature_columns]
    numeric = features.to_numpy(dtype=np.float64, copy=False)
    return {
        "nan": int(np.isnan(numeric).sum()),
        "pos_inf": int(np.isposinf(numeric).sum()),
        "neg_inf": int(np.isneginf(numeric).sum()),
    }


def load_split_frame(raw_files: list[RawFile], split: str, columns: list[str]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Load all files for one split, preserving label values and file provenance in reports only."""
    frames: list[pd.DataFrame] = []
    file_rows: list[dict[str, Any]] = []
    for raw_file in [item for item in raw_files if item.split == split]:
        frame = pd.read_parquet(raw_file.path, columns=columns)
        frames.append(frame)
        file_rows.append(
            {
                "file_name": raw_file.path.name,
                "relative_path": project_relative(raw_file.path),
                "split": split,
                "attack_family_from_filename": raw_file.attack_family_from_filename,
                "row_count": int(len(frame)),
            }
        )
    combined = pd.concat(frames, ignore_index=True)
    return combined, file_rows


def build_label_distribution(df: pd.DataFrame, split: str) -> list[dict[str, Any]]:
    """Summarize exact labels and binary target values for one split."""
    total = len(df)
    rows: list[dict[str, Any]] = []
    for label, count in df[TARGET_COLUMN].astype(str).value_counts().sort_index().items():
        rows.append(
            {
                "split": split,
                "label": label,
                "binary_target": int(label.strip().lower() != "benign"),
                "row_count": int(count),
                "percentage": round((int(count) / total * 100) if total else 0, 6),
            }
        )
    return rows


def fit_training_medians(train_df: pd.DataFrame, feature_columns: list[str]) -> dict[str, float]:
    """Fit deterministic median imputation values on training data only."""
    sanitized = train_df[feature_columns].replace([np.inf, -np.inf], np.nan)
    medians = sanitized.median(axis=0, skipna=True, numeric_only=True)
    return {
        column: float(0.0 if pd.isna(medians[column]) else medians[column])
        for column in feature_columns
    }


def apply_numeric_cleaning(
    df: pd.DataFrame,
    feature_columns: list[str],
    medians: dict[str, float],
) -> pd.DataFrame:
    """Replace non-finite values and cast features to a consistent float64 representation."""
    cleaned = df.copy()
    features = cleaned[feature_columns].replace([np.inf, -np.inf], np.nan)
    cleaned[feature_columns] = features.fillna(value=medians).astype("float64")
    cleaned[ENCODED_TARGET_COLUMN] = (
        cleaned[TARGET_COLUMN].astype(str).str.strip().str.lower().ne("benign").astype("int8")
    )
    return cleaned[feature_columns + [TARGET_COLUMN, ENCODED_TARGET_COLUMN]]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as CSV, preserving headers when rows are present."""
    pd.DataFrame(rows).to_csv(path, index=False)


def build_metadata(
    *,
    raw_files: list[RawFile],
    original_columns: list[str],
    feature_columns: list[str],
    training_medians: dict[str, float],
    split_rows: dict[str, int],
    label_distribution: list[dict[str, Any]],
    missing_inf_before: dict[str, dict[str, int]],
    missing_inf_after: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Build machine-readable preprocessing metadata."""
    return {
        "phase": "Phase 3 preprocessing",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "CIC-DDoS2019",
        "input_files": [
            {
                "file_name": item.path.name,
                "relative_path": project_relative(item.path),
                "split": item.split,
                "attack_family_from_filename": item.attack_family_from_filename,
            }
            for item in raw_files
        ],
        "target_column": TARGET_COLUMN,
        "encoded_target_column": ENCODED_TARGET_COLUMN,
        "target_encoding": {
            "Benign": 0,
            "all_non_benign_labels": 1,
            "note": "Exact Label values are preserved in the Label column; binary encoding is for DDoS baseline modeling.",
        },
        "original_column_count": len(original_columns),
        "original_feature_count_excluding_target": len(original_columns) - 1,
        "removed_constant_features": list(CONSTANT_FEATURES_TO_REMOVE),
        "removed_constant_feature_count": len(CONSTANT_FEATURES_TO_REMOVE),
        "final_feature_count": len(feature_columns),
        "final_feature_names": feature_columns,
        "protocol_retained": "Protocol" in feature_columns,
        "correlated_pairs_removed": False,
        "scaling_applied": False,
        "scaling_decision": "No standardization/scaling applied because Random Forest does not require feature scaling.",
        "missing_inf_handling": {
            "fit_scope": "training split only",
            "decision": "Replace +Inf and -Inf with NaN, then impute NaN using training medians per feature; all-missing training medians fall back to 0.0.",
            "training_medians": training_medians,
        },
        "rows_processed": split_rows,
        "label_distribution": label_distribution,
        "missing_inf_counts_before": missing_inf_before,
        "missing_inf_counts_after": missing_inf_after,
        "raw_data_modified": False,
    }


def run_preprocessing(raw_dir: Path, output_dir: Path, report_dir: Path) -> dict[str, Any]:
    """Run the complete preprocessing pipeline and write outputs."""
    raw_files = discover_raw_files(raw_dir)
    original_columns, feature_columns, schema_rows = validate_schema(raw_files)
    read_columns = feature_columns + [TARGET_COLUMN]

    train_df, train_file_rows = load_split_frame(raw_files, "training", read_columns)
    test_df, test_file_rows = load_split_frame(raw_files, "testing", read_columns)

    missing_inf_before = {
        "training": count_missing_and_inf(train_df, feature_columns),
        "testing": count_missing_and_inf(test_df, feature_columns),
    }
    label_distribution = build_label_distribution(train_df, "training") + build_label_distribution(
        test_df,
        "testing",
    )

    training_medians = fit_training_medians(train_df, feature_columns)
    train_clean = apply_numeric_cleaning(train_df, feature_columns, training_medians)
    test_clean = apply_numeric_cleaning(test_df, feature_columns, training_medians)

    missing_inf_after = {
        "training": count_missing_and_inf(train_clean, feature_columns),
        "testing": count_missing_and_inf(test_clean, feature_columns),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    train_clean.to_parquet(output_dir / "train.parquet", index=False)
    test_clean.to_parquet(output_dir / "test.parquet", index=False)

    split_rows = {"training": int(len(train_clean)), "testing": int(len(test_clean))}
    metadata = build_metadata(
        raw_files=raw_files,
        original_columns=original_columns,
        feature_columns=feature_columns,
        training_medians=training_medians,
        split_rows=split_rows,
        label_distribution=label_distribution,
        missing_inf_before=missing_inf_before,
        missing_inf_after=missing_inf_after,
    )
    (report_dir / "preprocessing_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    file_rows = train_file_rows + test_file_rows
    write_csv(report_dir / "input_file_summary.csv", file_rows)
    write_csv(report_dir / "schema_validation.csv", schema_rows)
    write_csv(report_dir / "label_distribution.csv", label_distribution)
    write_csv(
        report_dir / "feature_list.csv",
        [{"feature_index": index, "feature_name": column} for index, column in enumerate(feature_columns, start=1)],
    )
    write_csv(
        report_dir / "missing_inf_summary.csv",
        [
            {"split": split, "stage": stage, **counts}
            for stage, by_split in (("before", missing_inf_before), ("after", missing_inf_after))
            for split, counts in by_split.items()
        ],
    )
    write_csv(
        report_dir / "preprocessing_summary.csv",
        [
            {"metric": "input_file_count", "value": len(raw_files)},
            {"metric": "training_row_count", "value": split_rows["training"]},
            {"metric": "testing_row_count", "value": split_rows["testing"]},
            {"metric": "original_column_count", "value": len(original_columns)},
            {"metric": "original_feature_count_excluding_target", "value": len(original_columns) - 1},
            {"metric": "removed_constant_feature_count", "value": len(CONSTANT_FEATURES_TO_REMOVE)},
            {"metric": "final_feature_count", "value": len(feature_columns)},
            {"metric": "target_column", "value": TARGET_COLUMN},
            {"metric": "encoded_target_column", "value": ENCODED_TARGET_COLUMN},
            {"metric": "scaling_applied", "value": False},
            {"metric": "raw_data_modified", "value": False},
        ],
    )
    return metadata


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    raw_dir = resolve_project_path(args.raw_dir)
    output_dir = resolve_project_path(args.output_dir)
    report_dir = resolve_project_path(args.report_dir)

    metadata = run_preprocessing(raw_dir=raw_dir, output_dir=output_dir, report_dir=report_dir)
    print("=" * 80)
    print("CIC-DDOS2019 PHASE 3 PREPROCESSING")
    print("=" * 80)
    print(f"Raw directory: {project_relative(raw_dir)}")
    print(f"Processed output directory: {project_relative(output_dir)}")
    print(f"Report directory: {project_relative(report_dir)}")
    print(f"Input files: {len(metadata['input_files'])}")
    print(f"Rows: training={metadata['rows_processed']['training']}, testing={metadata['rows_processed']['testing']}")
    print(f"Original features excluding target: {metadata['original_feature_count_excluding_target']}")
    print(f"Removed constant features: {metadata['removed_constant_feature_count']}")
    print(f"Final features: {metadata['final_feature_count']}")
    print("Protocol retained: yes")
    print("Scaling applied: no")
    print("Raw data was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
