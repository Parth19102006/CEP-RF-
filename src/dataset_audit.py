"""Inspect raw CIC-DDoS2019 Parquet metadata and labels for Phase 2.

This script reports file dimensions, schemas, label distributions, constant and
near-constant features, correlation redundancy, identifier/leakage risks, and a
feature taxonomy. It does not clean data, transform features, or modify any
files in data/raw/.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "dataset_audit"

SPLIT_PATTERN = re.compile(r"^(?P<family>.+)-(?P<split>training|testing)$", re.IGNORECASE)
LABEL_NAME_CANDIDATES = (
    "label",
    "target",
    "class",
    "attack",
    "category",
)
DDOS_LABELS = {
    "dns",
    "drdos_dns",
    "ldap",
    "drdos_ldap",
    "mssql",
    "drdos_mssql",
    "netbios",
    "drdos_netbios",
    "ntp",
    "drdos_ntp",
    "portmap",
    "drdos_portmap",
    "snmp",
    "drdos_snmp",
    "syn",
    "tftp",
    "udp",
    "drdos_udp",
    "udplag",
    "udp-lag",
    "webddos",
}

LEAKAGE_KEYWORD_RULES: tuple[tuple[str, str], ...] = (
    (r"\bflow[\s_-]?id\b", "flow_identifier"),
    (r"\bsource[\s_-]?ip\b", "source_ip"),
    (r"\bdest(ination)?[\s_-]?ip\b", "destination_ip"),
    (r"\btimestamp\b", "timestamp"),
    (r"\btime[\s_-]?stamp\b", "timestamp"),
    (r"\bmac[\s_-]?addr\b", "mac_address"),
    (r"\bhostname\b", "hostname"),
    (r"\bprotocol\b", "protocol_keyword"),
)
LEAKAGE_ALLOWLIST = {"Protocol"}
NEAR_CONSTANT_VARIANCE_THRESHOLD = 1e-12

FEATURE_TAXONOMY_RULES: tuple[tuple[str, str], ...] = (
    (r"^Protocol$", "protocol"),
    (r"Flow Duration|Flow IAT|Fwd IAT|Bwd IAT|Active |Idle ", "timing_and_inter_arrival"),
    (r"Packet Length|Avg Packet Size|Avg Fwd Segment Size|Avg Bwd Segment Size", "packet_size"),
    (r"Total Fwd Packets|Total Backward Packets|Subflow .* Packets|Fwd Act Data Packets", "packet_count"),
    (r"Bytes/s|Packets/s|Bulk Rate", "rate_and_throughput"),
    (r"Flags|Flag Count", "tcp_flags"),
    (r"Header Length|Win Bytes|Seg Size Min", "header_and_window"),
    (r"Subflow .* Bytes|Packets Length Total", "subflow_volume"),
    (r"Bulk", "bulk_transfer"),
    (r"Down/Up Ratio", "direction_ratio"),
    (r"^Label$", "target"),
)


@dataclass(frozen=True)
class SchemaInspection:
    """Schema metadata collected from one Parquet file."""

    file_name: str
    relative_path: str
    attack_family: str
    split: str
    row_count: int | None
    column_count: int | None
    schema: list[tuple[str, str]]
    read_status: str
    error: str | None = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect schemas and dimensions for raw CIC-DDoS2019 Parquet files."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory containing raw Parquet files. Defaults to data/raw/.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory where schema inspection CSV reports will be written.",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.95,
        help="Absolute Pearson correlation threshold for redundant feature pairs.",
    )
    parser.add_argument(
        "--correlation-sample-size",
        type=int,
        default=50_000,
        help="Maximum rows sampled for correlation analysis (0 = use all rows).",
    )
    parser.add_argument(
        "--skip-correlation",
        action="store_true",
        help="Skip correlation analysis (faster run).",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    """Resolve relative paths from the project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    """Return a readable path relative to the project root when possible."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def parse_dataset_name(path: Path) -> tuple[str, str]:
    """Infer attack family and train/test split from the filename only."""
    match = SPLIT_PATTERN.match(path.stem)
    if not match:
        return path.stem, "unknown"
    return match.group("family"), match.group("split").lower()


def find_parquet_files(raw_dir: Path) -> list[Path]:
    """Find Parquet files directly inside the raw data directory."""
    if not raw_dir.exists() or not raw_dir.is_dir():
        return []
    return sorted(raw_dir.glob("*.parquet"))


def inspect_parquet_schema(path: Path) -> SchemaInspection:
    """Read row count, column count, column names, and data types from one file."""
    attack_family, split = parse_dataset_name(path)
    try:
        parquet_file = pq.ParquetFile(path)
        schema = [(field.name, str(field.type)) for field in parquet_file.schema_arrow]
        return SchemaInspection(
            file_name=path.name,
            relative_path=project_relative(path),
            attack_family=attack_family,
            split=split,
            row_count=parquet_file.metadata.num_rows,
            column_count=len(schema),
            schema=schema,
            read_status="ok",
        )
    except Exception as exc:  # pragma: no cover - depends on local/corrupt files
        return SchemaInspection(
            file_name=path.name,
            relative_path=project_relative(path),
            attack_family=attack_family,
            split=split,
            row_count=None,
            column_count=None,
            schema=[],
            read_status="error",
            error=f"{type(exc).__name__}: {exc}",
        )


def identify_label_column(schema: list[tuple[str, str]]) -> str | None:
    """Identify the likely label column from column names without assuming case."""
    column_names = [column_name for column_name, _ in schema]
    lowered_to_original = {column_name.lower(): column_name for column_name in column_names}

    for candidate in LABEL_NAME_CANDIDATES:
        if candidate in lowered_to_original:
            return lowered_to_original[candidate]

    for column_name in column_names:
        normalized = column_name.lower().replace("_", " ").replace("-", " ")
        if any(candidate in normalized.split() for candidate in LABEL_NAME_CANDIDATES):
            return column_name

    return None


def classify_label_value(label: object) -> str:
    """Classify exact label values for reporting without renaming them."""
    if label is None:
        return "another_label"

    normalized = str(label).strip().lower()
    if normalized == "benign":
        return "benign_traffic"
    if normalized in DDOS_LABELS or "ddos" in normalized or normalized.startswith("drdos"):
        return "ddos_traffic"
    return "another_label"


def read_label_counts(path: Path, label_column: str) -> Counter[str]:
    """Read only the detected label column and count its exact values."""
    table = pq.read_table(path, columns=[label_column])
    labels = table.column(label_column).combine_chunks().to_pylist()
    return Counter("<NA>" if label is None else str(label) for label in labels)


def build_label_distribution(
    inspections: list[SchemaInspection],
    raw_dir: Path,
) -> list[dict[str, object]]:
    """Build one row per exact label value in each Parquet file."""
    rows: list[dict[str, object]] = []
    for item in inspections:
        label_column = identify_label_column(item.schema)
        if item.read_status != "ok" or label_column is None:
            rows.append(
                {
                    "file_name": item.file_name,
                    "label_column": label_column,
                    "label": None,
                    "row_count": 0,
                    "percentage": 0,
                    "label_category": "label_column_not_detected",
                    "read_status": item.read_status,
                    "error": item.error,
                }
            )
            continue

        try:
            counts = read_label_counts(raw_dir / item.file_name, label_column)
        except Exception as exc:  # pragma: no cover - depends on local/corrupt files
            rows.append(
                {
                    "file_name": item.file_name,
                    "label_column": label_column,
                    "label": None,
                    "row_count": 0,
                    "percentage": 0,
                    "label_category": "label_read_error",
                    "read_status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        total_rows = sum(counts.values())
        categories_in_file = sorted({classify_label_value(label) for label in counts})
        for label, count in sorted(counts.items()):
            rows.append(
                {
                    "file_name": item.file_name,
                    "label_column": label_column,
                    "label": label,
                    "row_count": count,
                    "percentage": round((count / total_rows * 100) if total_rows else 0, 6),
                    "label_category": classify_label_value(label),
                    "traffic_categories_in_file": "; ".join(categories_in_file),
                    "read_status": "ok",
                    "error": None,
                }
            )

    return rows


def build_label_summary(label_distribution: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build overall label counts across all files, preserving exact label values."""
    totals: dict[tuple[str | None, str], int] = {}
    for row in label_distribution:
        if row["read_status"] != "ok" or row["label"] is None:
            continue
        key = (str(row["label_column"]) if row["label_column"] is not None else None, str(row["label"]))
        totals[key] = totals.get(key, 0) + int(row["row_count"])

    grand_total = sum(totals.values())
    summary_rows: list[dict[str, object]] = []
    for (label_column, label), count in sorted(totals.items(), key=lambda item: item[0]):
        summary_rows.append(
            {
                "file_name": "ALL_FILES",
                "label_column": label_column,
                "label": label,
                "row_count": count,
                "percentage": round((count / grand_total * 100) if grand_total else 0, 6),
                "label_category": classify_label_value(label),
            }
        )

    return summary_rows


def build_label_file_mapping(
    inspections: list[SchemaInspection],
    label_distribution: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Build the requested label-to-file mapping with exact labels and counts."""
    split_by_file = {item.file_name: item.split for item in inspections}
    rows: list[dict[str, object]] = []
    for row in label_distribution:
        if row["read_status"] != "ok" or row["label"] is None:
            continue
        rows.append(
            {
                "file_name": row["file_name"],
                "split": split_by_file.get(str(row["file_name"]), "unknown"),
                "label": row["label"],
                "row_count": row["row_count"],
            }
        )
    return rows


def build_dataset_summary(inspections: list[SchemaInspection]) -> list[dict[str, object]]:
    """Build one summary row per Parquet file."""
    return [
        {
            "file_name": item.file_name,
            "relative_path": item.relative_path,
            "attack_family": item.attack_family,
            "split": item.split,
            "row_count": item.row_count,
            "column_count": item.column_count,
            "read_status": item.read_status,
            "error": item.error,
        }
        for item in inspections
    ]


def build_column_summary(inspections: list[SchemaInspection]) -> list[dict[str, object]]:
    """Build one row per column in each Parquet file."""
    rows: list[dict[str, object]] = []
    for item in inspections:
        for position, (column_name, data_type) in enumerate(item.schema, start=1):
            rows.append(
                {
                    "file_name": item.file_name,
                    "attack_family": item.attack_family,
                    "split": item.split,
                    "column_position": position,
                    "column_name": column_name,
                    "data_type": data_type,
                }
            )
    return rows


def build_schema_comparison(inspections: list[SchemaInspection]) -> list[dict[str, object]]:
    """Compare every file schema against the first successfully read file."""
    readable = [item for item in inspections if item.read_status == "ok"]
    if not readable:
        return []

    reference = readable[0]
    reference_schema = dict(reference.schema)
    reference_order = [column_name for column_name, _ in reference.schema]
    rows: list[dict[str, object]] = []

    for compared in readable[1:]:
        compared_schema = dict(compared.schema)

        for column_name in reference_order:
            reference_type = reference_schema[column_name]
            compared_type = compared_schema.get(column_name)
            if compared_type is None:
                rows.append(
                    {
                        "reference_file": reference.file_name,
                        "compared_file": compared.file_name,
                        "column_name": column_name,
                        "issue": "missing_column",
                        "reference_data_type": reference_type,
                        "compared_data_type": None,
                    }
                )
            elif compared_type != reference_type:
                rows.append(
                    {
                        "reference_file": reference.file_name,
                        "compared_file": compared.file_name,
                        "column_name": column_name,
                        "issue": "different_data_type",
                        "reference_data_type": reference_type,
                        "compared_data_type": compared_type,
                    }
                )

        for column_name, compared_type in compared_schema.items():
            if column_name not in reference_schema:
                rows.append(
                    {
                        "reference_file": reference.file_name,
                        "compared_file": compared.file_name,
                        "column_name": column_name,
                        "issue": "extra_column",
                        "reference_data_type": None,
                        "compared_data_type": compared_type,
                    }
                )

    return rows


def classify_type_difference(data_types: list[str]) -> str:
    """Classify whether a dtype difference stays within ints, floats, or another family."""
    unique_types = set(data_types)
    if all(data_type.startswith(("int", "uint")) for data_type in unique_types):
        return "integer_types"
    if all(
        data_type.startswith(("float", "double", "halffloat"))
        for data_type in unique_types
    ):
        return "floating_point_types"
    return "other_types"


def build_schema_type_differences(
    inspections: list[SchemaInspection],
) -> list[dict[str, object]]:
    """Group datatype-mismatch columns by dtype and file usage across all files."""
    readable = [item for item in inspections if item.read_status == "ok"]
    column_types: dict[str, dict[str, list[str]]] = {}

    for item in readable:
        for column_name, data_type in item.schema:
            column_types.setdefault(column_name, {}).setdefault(data_type, []).append(item.file_name)

    rows: list[dict[str, object]] = []
    for column_name in sorted(column_types):
        type_to_files = column_types[column_name]
        if len(type_to_files) <= 1:
            continue

        type_difference_category = classify_type_difference(list(type_to_files))
        for data_type in sorted(type_to_files):
            files = sorted(type_to_files[data_type])
            rows.append(
                {
                    "column_name": column_name,
                    "data_type": data_type,
                    "files_using_data_type": "; ".join(files),
                    "file_count": len(files),
                    "all_data_types_for_column": "; ".join(sorted(type_to_files)),
                    "type_difference_category": type_difference_category,
                }
            )

    return rows


def is_numeric_data_type(data_type: str) -> bool:
    """Return True when a Parquet/Arrow type string represents numeric data."""
    numeric_prefixes = ("int", "uint", "float", "double", "halffloat")
    return any(data_type.startswith(prefix) for prefix in numeric_prefixes)


def get_numeric_feature_columns(schema: list[tuple[str, str]], label_column: str | None) -> list[str]:
    """Return numeric model-feature column names, excluding the label column."""
    excluded = {label_column} if label_column else set()
    return [
        column_name
        for column_name, data_type in schema
        if column_name not in excluded and is_numeric_data_type(data_type)
    ]


def classify_feature_group(column_name: str) -> str:
    """Assign a feature to a taxonomy group using pattern rules."""
    for pattern, group in FEATURE_TAXONOMY_RULES:
        if re.search(pattern, column_name, flags=re.IGNORECASE):
            return group
    return "other"


def classify_leakage_reason(column_name: str) -> str | None:
    """Return a leakage keyword category when the column name matches a rule."""
    normalized = column_name.strip().lower()
    for pattern, reason in LEAKAGE_KEYWORD_RULES:
        if re.search(pattern, normalized):
            return reason
    return None


def read_numeric_column(path: Path, column_name: str) -> np.ndarray:
    """Read one numeric column from a Parquet file as a NumPy array."""
    table = pq.read_table(path, columns=[column_name])
    array = table.column(column_name).combine_chunks()
    return array.to_numpy(zero_copy_only=False)


def build_file_inventory(parquet_files: list[Path]) -> list[dict[str, object]]:
    """Build one inventory row per Parquet file with size and timestamp metadata."""
    rows: list[dict[str, object]] = []
    for index, path in enumerate(parquet_files, start=1):
        attack_family, split = parse_dataset_name(path)
        stat = path.stat()
        rows.append(
            {
                "file_index": index,
                "file_name": path.name,
                "relative_path": project_relative(path),
                "attack_family_from_filename": attack_family,
                "split_from_filename": split,
                "extension": path.suffix,
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 6),
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            }
        )
    return rows


def build_feature_taxonomy(inspections: list[SchemaInspection]) -> list[dict[str, object]]:
    """Build one taxonomy row per unique feature column."""
    readable = [item for item in inspections if item.read_status == "ok"]
    if not readable:
        return []

    reference = readable[0]
    label_column = identify_label_column(reference.schema)
    rows: list[dict[str, object]] = []
    for position, (column_name, data_type) in enumerate(reference.schema, start=1):
        rows.append(
            {
                "column_position": position,
                "column_name": column_name,
                "data_type": data_type,
                "feature_group": classify_feature_group(column_name),
                "is_model_feature": column_name != label_column and is_numeric_data_type(data_type),
                "is_label_column": column_name == label_column,
            }
        )
    return rows


def build_potential_leakage_columns(inspections: list[SchemaInspection]) -> list[dict[str, object]]:
    """Flag columns whose names resemble identifiers or leakage-prone fields."""
    rows: list[dict[str, object]] = []
    for item in inspections:
        if item.read_status != "ok":
            continue
        for column_name, _ in item.schema:
            reason = classify_leakage_reason(column_name)
            if reason is None:
                continue
            rows.append(
                {
                    "file_name": item.file_name,
                    "attack_family": item.attack_family,
                    "split": item.split,
                    "column_name": column_name,
                    "matched_rule": reason,
                    "is_allowlisted_legitimate_feature": column_name in LEAKAGE_ALLOWLIST,
                    "review_note": (
                        "Legitimate traffic feature; keyword match is a false positive."
                        if column_name in LEAKAGE_ALLOWLIST
                        else "Review before modeling; may cause leakage if values identify capture sessions."
                    ),
                }
            )
    return rows


def build_constant_feature_analysis(
    inspections: list[SchemaInspection],
    raw_dir: Path,
) -> list[dict[str, object]]:
    """Detect globally constant and per-file constant numeric features."""
    readable = [item for item in inspections if item.read_status == "ok"]
    if not readable:
        return []

    reference = readable[0]
    label_column = identify_label_column(reference.schema)
    feature_columns = get_numeric_feature_columns(reference.schema, label_column)

    global_tracker: dict[str, float | str] = {}
    rows: list[dict[str, object]] = []

    for item in readable:
        path = raw_dir / item.file_name
        file_feature_columns = get_numeric_feature_columns(item.schema, identify_label_column(item.schema))
        for column_name in file_feature_columns:
            try:
                values = read_numeric_column(path, column_name)
            except Exception as exc:  # pragma: no cover - depends on local/corrupt files
                rows.append(
                    {
                        "scope": "file",
                        "file_name": item.file_name,
                        "column_name": column_name,
                        "unique_count": None,
                        "constant_value": None,
                        "min_value": None,
                        "max_value": None,
                        "variance": None,
                        "is_constant": False,
                        "is_globally_constant": False,
                        "read_status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                unique_count = 0
                constant_value = None
                min_value = None
                max_value = None
                variance = 0.0
                is_constant = True
            else:
                unique_values = np.unique(finite_values)
                unique_count = int(unique_values.size)
                constant_value = float(unique_values[0]) if unique_count == 1 else None
                min_value = float(np.min(finite_values))
                max_value = float(np.max(finite_values))
                variance = float(np.var(finite_values))
                is_constant = unique_count <= 1

            if unique_count > 1:
                global_tracker[column_name] = "__NON_CONSTANT__"
            elif unique_count == 1:
                value = float(unique_values[0])
                current = global_tracker.get(column_name)
                if current == "__NON_CONSTANT__":
                    pass
                elif column_name not in global_tracker:
                    global_tracker[column_name] = value
                elif current != value:
                    global_tracker[column_name] = "__NON_CONSTANT__"

            rows.append(
                {
                    "scope": "file",
                    "file_name": item.file_name,
                    "column_name": column_name,
                    "unique_count": unique_count,
                    "constant_value": constant_value,
                    "min_value": min_value,
                    "max_value": max_value,
                    "variance": round(variance, 12),
                    "is_constant": is_constant,
                    "is_globally_constant": False,
                    "read_status": "ok",
                    "error": None,
                }
            )

    globally_constant: dict[str, float | None] = {}
    for column_name in feature_columns:
        tracked = global_tracker.get(column_name)
        if isinstance(tracked, float):
            globally_constant[column_name] = tracked
        else:
            globally_constant[column_name] = None

    global_rows: list[dict[str, object]] = []
    for column_name in feature_columns:
        constant_value = globally_constant.get(column_name)
        is_globally_constant = constant_value is not None
        global_rows.append(
            {
                "scope": "global",
                "file_name": "ALL_FILES",
                "column_name": column_name,
                "unique_count": 1 if is_globally_constant else None,
                "constant_value": constant_value,
                "min_value": constant_value,
                "max_value": constant_value,
                "variance": 0.0 if is_globally_constant else None,
                "is_constant": is_globally_constant,
                "is_globally_constant": is_globally_constant,
                "read_status": "ok",
                "error": None,
            }
        )

    for row in rows:
        if row["read_status"] == "ok":
            row["is_globally_constant"] = globally_constant.get(str(row["column_name"])) is not None

    return global_rows + rows


def build_near_constant_features(
    constant_feature_analysis: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Flag file-level features with extremely low variance but more than one unique value."""
    rows: list[dict[str, object]] = []
    for row in constant_feature_analysis:
        if row["scope"] != "file" or row["read_status"] != "ok":
            continue
        variance = row["variance"]
        unique_count = row["unique_count"]
        if variance is None or unique_count is None:
            continue
        if unique_count > 1 and isinstance(variance, (float, int)) and variance <= NEAR_CONSTANT_VARIANCE_THRESHOLD:
            rows.append(
                {
                    "file_name": row["file_name"],
                    "column_name": row["column_name"],
                    "unique_count": unique_count,
                    "variance": variance,
                    "min_value": row["min_value"],
                    "max_value": row["max_value"],
                    "note": "Near-constant numeric feature; review before feature selection.",
                }
            )
    return rows


def _sample_correlation_matrix(
    matrix: np.ndarray,
    sample_size: int,
    random_seed: int,
) -> np.ndarray:
    """Return a row sample of the feature matrix for correlation analysis."""
    if sample_size <= 0 or matrix.shape[0] <= sample_size:
        return matrix
    rng = np.random.default_rng(random_seed)
    indices = rng.choice(matrix.shape[0], size=sample_size, replace=False)
    return matrix[indices]


def build_correlation_analysis(
    inspections: list[SchemaInspection],
    raw_dir: Path,
    correlation_threshold: float,
    correlation_sample_size: int,
    random_seed: int = 42,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Compute feature correlations and return pair list, matrix rows, and metadata."""
    readable = [item for item in inspections if item.read_status == "ok"]
    if not readable:
        return [], [], {"status": "no_readable_files"}

    reference = readable[0]
    label_column = identify_label_column(reference.schema)
    feature_columns = get_numeric_feature_columns(reference.schema, label_column)
    if not feature_columns:
        return [], [], {"status": "no_numeric_features"}

    tables: list[pa.Table] = []
    total_rows = 0

    # Correlation requires a common numeric dtype across all files.
    # CIC-DDoS2019 contains schema-width differences such as int8 vs int16
    # for otherwise identical numeric features.
    correlation_schema = pa.schema(
        [(column_name, pa.float64()) for column_name in feature_columns]
    )

    for item in readable:
        path = raw_dir / item.file_name
        table = pq.read_table(
            path,
            columns=feature_columns,
        )

        # Normalize all numeric feature columns to float64 before concatenation.
        table = table.cast(correlation_schema)

        tables.append(table)
        total_rows += table.num_rows

    combined = pa.concat_tables(
        tables,
        promote_options="default",
    )

    matrix = combined.to_pandas().to_numpy(
        dtype=np.float64,
        copy=True,
    )
    sampled_matrix = _sample_correlation_matrix(matrix, correlation_sample_size, random_seed)

    if sampled_matrix.shape[0] < 2:
        return [], [], {"status": "insufficient_rows", "rows_used": sampled_matrix.shape[0]}

    corr = np.corrcoef(sampled_matrix, rowvar=False)
    np.fill_diagonal(corr, 1.0)

    pair_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    feature_count = len(feature_columns)

    for row_index, row_name in enumerate(feature_columns):
        matrix_row: dict[str, object] = {"feature": row_name}
        for col_index, col_name in enumerate(feature_columns):
            value = corr[row_index, col_index]
            matrix_row[col_name] = round(float(value), 6) if np.isfinite(value) else ""
        matrix_rows.append(matrix_row)

        for col_index in range(row_index + 1, feature_count):
            col_name = feature_columns[col_index]
            value = corr[row_index, col_index]
            if not np.isfinite(value):
                continue
            abs_value = abs(float(value))
            if abs_value >= correlation_threshold:
                pair_rows.append(
                    {
                        "feature_a": row_name,
                        "feature_b": col_name,
                        "pearson_r": round(float(value), 6),
                        "abs_pearson_r": round(abs_value, 6),
                        "feature_group_a": classify_feature_group(row_name),
                        "feature_group_b": classify_feature_group(col_name),
                        "redundancy_note": "Candidate pair for correlation pruning in Phase 5.",
                    }
                )

    metadata = {
        "status": "ok",
        "total_rows": total_rows,
        "rows_used_for_correlation": sampled_matrix.shape[0],
        "feature_count": feature_count,
        "correlation_threshold": correlation_threshold,
        "high_correlation_pair_count": len(pair_rows),
        "random_seed": random_seed,
    }
    return pair_rows, matrix_rows, metadata


def build_dataset_characterization_summary(
    dataset_summary: list[dict[str, object]],
    label_summary: list[dict[str, object]],
    constant_feature_analysis: list[dict[str, object]],
    correlation_metadata: dict[str, object],
    feature_taxonomy: list[dict[str, object]],
    potential_leakage_columns: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Build a paper-ready dataset characterization summary."""
    total_rows = sum(int(row["row_count"]) for row in dataset_summary if row["row_count"] is not None)
    global_constants = [
        row
        for row in constant_feature_analysis
        if row["scope"] == "global" and row["is_globally_constant"]
    ]
    model_features = [row for row in feature_taxonomy if row["is_model_feature"]]
    false_positive_leakage = [
        row for row in potential_leakage_columns if row["is_allowlisted_legitimate_feature"]
    ]
    true_leakage_flags = [
        row for row in potential_leakage_columns if not row["is_allowlisted_legitimate_feature"]
    ]
    benign_rows = next((row["row_count"] for row in label_summary if row["label"] == "Benign"), 0)
    attack_rows = total_rows - int(benign_rows) if total_rows else 0

    metrics: list[tuple[str, object]] = [
        ("parquet_file_count", len(dataset_summary)),
        ("total_flow_rows", total_rows),
        ("raw_feature_columns_including_label", len(feature_taxonomy)),
        ("numeric_model_feature_count", len(model_features)),
        ("unique_multiclass_labels", len(label_summary)),
        ("benign_row_count", benign_rows),
        ("attack_row_count", attack_rows),
        ("globally_constant_feature_count", len(global_constants)),
        ("near_constant_feature_pairs", "see near_constant_features.csv"),
        ("high_correlation_feature_pairs", correlation_metadata.get("high_correlation_pair_count", 0)),
        ("leakage_keyword_matches", len(potential_leakage_columns)),
        ("leakage_false_positives_allowlisted", len(false_positive_leakage)),
        ("leakage_flags_requiring_review", len(true_leakage_flags)),
        ("feature_taxonomy_groups", len({row["feature_group"] for row in feature_taxonomy})),
    ]
    return [{"metric": metric, "value": value} for metric, value in metrics]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write rows to CSV, preserving headers even when there are no rows."""
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(
    report_dir: Path,
    dataset_summary: list[dict[str, object]],
    column_summary: list[dict[str, object]],
    schema_comparison: list[dict[str, object]],
    schema_type_differences: list[dict[str, object]],
    label_distribution: list[dict[str, object]],
    label_summary: list[dict[str, object]],
    label_file_mapping: list[dict[str, object]],
    constant_feature_analysis: list[dict[str, object]],
    near_constant_features: list[dict[str, object]],
    correlation_pairs: list[dict[str, object]],
    correlation_matrix: list[dict[str, object]],
    potential_leakage_columns: list[dict[str, object]],
    feature_taxonomy: list[dict[str, object]],
    dataset_characterization: list[dict[str, object]],
) -> None:
    """Write Phase 2 reports."""
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        report_dir / "dataset_summary.csv",
        dataset_summary,
        [
            "file_name",
            "relative_path",
            "attack_family",
            "split",
            "row_count",
            "column_count",
            "read_status",
            "error",
        ],
    )
    write_csv(
        report_dir / "column_summary.csv",
        column_summary,
        [
            "file_name",
            "attack_family",
            "split",
            "column_position",
            "column_name",
            "data_type",
        ],
    )
    write_csv(
        report_dir / "schema_comparison.csv",
        schema_comparison,
        [
            "reference_file",
            "compared_file",
            "column_name",
            "issue",
            "reference_data_type",
            "compared_data_type",
        ],
    )
    write_csv(
        report_dir / "schema_type_differences.csv",
        schema_type_differences,
        [
            "column_name",
            "data_type",
            "files_using_data_type",
            "file_count",
            "all_data_types_for_column",
            "type_difference_category",
        ],
    )
    write_csv(
        report_dir / "label_distribution.csv",
        label_distribution,
        [
            "file_name",
            "label_column",
            "label",
            "row_count",
            "percentage",
            "label_category",
            "traffic_categories_in_file",
            "read_status",
            "error",
        ],
    )
    write_csv(
        report_dir / "label_summary.csv",
        label_summary,
        [
            "file_name",
            "label_column",
            "label",
            "row_count",
            "percentage",
            "label_category",
        ],
    )
    write_csv(
        report_dir / "label_file_mapping.csv",
        label_file_mapping,
        [
            "file_name",
            "split",
            "label",
            "row_count",
        ],
    )

    write_csv(
        report_dir / "constant_feature_analysis.csv",
        constant_feature_analysis,
        [
            "scope",
            "file_name",
            "column_name",
            "unique_count",
            "constant_value",
            "min_value",
            "max_value",
            "variance",
            "is_constant",
            "is_globally_constant",
            "read_status",
            "error",
        ],
    )

    write_csv(
        report_dir / "near_constant_features.csv",
        near_constant_features,
        [
            "file_name",
            "column_name",
            "unique_count",
            "variance",
            "min_value",
            "max_value",
            "note",
        ],
    )

    write_csv(
        report_dir / "high_correlation_pairs.csv",
        correlation_pairs,
        [
            "feature_a",
            "feature_b",
            "pearson_r",
            "abs_pearson_r",
            "feature_group_a",
            "feature_group_b",
            "redundancy_note",
        ],
    )

    correlation_matrix_fields = ["feature"]

    if correlation_matrix:
        correlation_matrix_fields.extend(
            str(row["feature"])
            for row in correlation_matrix
        )

    write_csv(
        report_dir / "correlation_matrix.csv",
        correlation_matrix,
        correlation_matrix_fields,
    )

    write_csv(
        report_dir / "potential_leakage_columns.csv",
        potential_leakage_columns,
        [
            "file_name",
            "attack_family",
            "split",
            "column_name",
            "matched_rule",
            "is_allowlisted_legitimate_feature",
            "review_note",
        ],
    )

    write_csv(
        report_dir / "feature_taxonomy.csv",
        feature_taxonomy,
        [
            "column_position",
            "column_name",
            "data_type",
            "feature_group",
            "is_model_feature",
            "is_label_column",
        ],
    )

    write_csv(
        report_dir / "dataset_characterization.csv",
        dataset_characterization,
        [
            "metric",
            "value",
        ],
    )


def print_label_results(
    label_distribution: list[dict[str, object]],
    label_summary: list[dict[str, object]],
    dataset_summary: list[dict[str, object]],
) -> None:
    """Print concise label-inspection results."""
    label_columns = sorted(
        {
            str(row["label_column"])
            for row in label_distribution
            if row["label_column"] is not None
        }
    )
    unique_labels = sorted(
        {
            str(row["label"])
            for row in label_distribution
            if row["read_status"] == "ok" and row["label"] is not None
        }
    )

    print("\nLabel inspection:")
    print(f"- Detected label column(s): {', '.join(label_columns) if label_columns else 'none'}")
    print(f"- Unique labels: {', '.join(unique_labels) if unique_labels else 'none'}")

    if len(label_columns) > 1:
        print("- Different label column names were detected across files.")

    rows_by_file: dict[str, list[dict[str, object]]] = {}
    total_rows_by_file = {
        str(row["file_name"]): row["row_count"] for row in dataset_summary
    }
    for row in label_distribution:
        if row["read_status"] == "ok" and row["label"] is not None:
            rows_by_file.setdefault(str(row["file_name"]), []).append(row)

    print("\nLabel-to-file mapping:")
    for file_name in sorted(rows_by_file):
        rows = rows_by_file[file_name]
        categories = sorted({str(row["label_category"]) for row in rows})
        if categories == ["benign_traffic"]:
            traffic_mix = "Benign traffic only"
        elif categories == ["ddos_traffic"]:
            traffic_mix = "Attack traffic only"
        elif "benign_traffic" in categories and "ddos_traffic" in categories:
            traffic_mix = "Both Benign and attack traffic"
        else:
            traffic_mix = "Another label category"

        labels = ", ".join(
            f"{row['label']}={row['row_count']}" for row in sorted(rows, key=lambda item: str(item["label"]))
        )
        print(
            f"- {file_name}: total_rows={total_rows_by_file.get(file_name)}, "
            f"{traffic_mix}; {labels}"
        )

    print("\nTotal samples per label across all files:")
    for row in label_summary:
        print(
            f"- {row['label']}: {row['row_count']} "
            f"({row['percentage']}%, {row['label_category']})"
        )


def print_results(
    raw_dir: Path,
    dataset_summary: list[dict[str, object]],
    schema_comparison: list[dict[str, object]],
    schema_type_differences: list[dict[str, object]],
    label_distribution: list[dict[str, object]],
    label_summary: list[dict[str, object]],
) -> None:
    """Print a concise human-readable report."""
    print("=" * 80)
    print("CIC-DDOS2019 PHASE 2 DATASET AUDIT")
    print("=" * 80)
    print(f"Raw directory: {project_relative(raw_dir)}")
    print(f"Parquet files inspected: {len(dataset_summary)}")

    issue_counts: dict[str, int] = {}
    for row in schema_comparison:
        issue = str(row["issue"])
        issue_counts[issue] = issue_counts.get(issue, 0) + 1

    print("\nSchema status:")
    print(f"- Schema comparison issues: {sum(issue_counts.values())}")
    for issue, count in sorted(issue_counts.items()):
        print(f"- {issue}: {count}")
    print(f"- Datatype-mismatch columns: {len({row['column_name'] for row in schema_type_differences})}")

    print_label_results(label_distribution, label_summary, dataset_summary)


def main() -> int:
    """Run the complete Phase 2 dataset audit."""
    args = parse_args()
    raw_dir = resolve_project_path(args.raw_dir)
    report_dir = resolve_project_path(args.report_dir)

    if not raw_dir.exists():
        print(f"Raw directory does not exist: {project_relative(raw_dir)}")
        print("No raw data was modified.")
        return 1

    if not raw_dir.is_dir():
        print(f"Raw path is not a directory: {project_relative(raw_dir)}")
        print("No raw data was modified.")
        return 1

    parquet_files = find_parquet_files(raw_dir)

    if not parquet_files:
        print(f"No Parquet files found in: {project_relative(raw_dir)}")
        print("No raw data was modified.")
        return 1

    # ------------------------------------------------------------------
    # 1. Basic dataset/schema inspection
    # ------------------------------------------------------------------
    inspections = [inspect_parquet_schema(path) for path in parquet_files]

    dataset_summary = build_dataset_summary(inspections)
    column_summary = build_column_summary(inspections)
    schema_comparison = build_schema_comparison(inspections)
    schema_type_differences = build_schema_type_differences(inspections)

    # ------------------------------------------------------------------
    # 2. Label inspection
    # ------------------------------------------------------------------
    label_distribution = build_label_distribution(
        inspections,
        raw_dir,
    )
    label_summary = build_label_summary(label_distribution)
    label_file_mapping = build_label_file_mapping(
        inspections,
        label_distribution,
    )

    # ------------------------------------------------------------------
    # 3. Constant / near-constant feature analysis
    # ------------------------------------------------------------------
    constant_feature_analysis = build_constant_feature_analysis(
        inspections,
        raw_dir,
    )

    near_constant_features = build_near_constant_features(
        constant_feature_analysis,
    )

    # ------------------------------------------------------------------
    # 4. Feature taxonomy
    # ------------------------------------------------------------------
    feature_taxonomy = build_feature_taxonomy(inspections)

    # ------------------------------------------------------------------
    # 5. Potential leakage / identifier analysis
    # ------------------------------------------------------------------
    potential_leakage_columns = build_potential_leakage_columns(
        inspections,
    )

    # ------------------------------------------------------------------
    # 6. Correlation / redundancy analysis
    # ------------------------------------------------------------------
    if args.skip_correlation:
        correlation_pairs = []
        correlation_matrix = []
        correlation_metadata = {
            "status": "skipped",
            "high_correlation_pair_count": 0,
            "correlation_threshold": args.correlation_threshold,
        }
    else:
        (
            correlation_pairs,
            correlation_matrix,
            correlation_metadata,
        ) = build_correlation_analysis(
            inspections,
            raw_dir,
            correlation_threshold=args.correlation_threshold,
            correlation_sample_size=args.correlation_sample_size,
            random_seed=42,
        )

    # ------------------------------------------------------------------
    # 7. Dataset characterization summary
    # ------------------------------------------------------------------
    dataset_characterization = build_dataset_characterization_summary(
        dataset_summary=dataset_summary,
        label_summary=label_summary,
        constant_feature_analysis=constant_feature_analysis,
        correlation_metadata=correlation_metadata,
        feature_taxonomy=feature_taxonomy,
        potential_leakage_columns=potential_leakage_columns,
    )

    # ------------------------------------------------------------------
    # 8. Write all Phase 2 reports
    # ------------------------------------------------------------------
    write_reports(
        report_dir=report_dir,
        dataset_summary=dataset_summary,
        column_summary=column_summary,
        schema_comparison=schema_comparison,
        schema_type_differences=schema_type_differences,
        label_distribution=label_distribution,
        label_summary=label_summary,
        label_file_mapping=label_file_mapping,
        constant_feature_analysis=constant_feature_analysis,
        near_constant_features=near_constant_features,
        correlation_pairs=correlation_pairs,
        correlation_matrix=correlation_matrix,
        potential_leakage_columns=potential_leakage_columns,
        feature_taxonomy=feature_taxonomy,
        dataset_characterization=dataset_characterization,
    )

    # ------------------------------------------------------------------
    # 9. Console summary
    # ------------------------------------------------------------------
    print_results(
        raw_dir=raw_dir,
        dataset_summary=dataset_summary,
        schema_comparison=schema_comparison,
        schema_type_differences=schema_type_differences,
        label_distribution=label_distribution,
        label_summary=label_summary,
    )

    print("\nPhase 2 analysis:")
    print(
        f"- Constant/feature-analysis rows: "
        f"{len(constant_feature_analysis)}"
    )
    print(
        f"- Near-constant feature/file pairs: "
        f"{len(near_constant_features)}"
    )
    print(
        f"- High-correlation feature pairs: "
        f"{len(correlation_pairs)}"
    )
    print(
        f"- Potential leakage/identifier matches: "
        f"{len(potential_leakage_columns)}"
    )
    print(
        f"- Feature taxonomy rows: "
        f"{len(feature_taxonomy)}"
    )
    print(
        f"- Numeric model features: "
        f"{sum(1 for row in feature_taxonomy if row['is_model_feature'])}"
    )

    print(
        f"\nReports written to: "
        f"{project_relative(report_dir)}"
    )
    print("Raw data was not modified.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
