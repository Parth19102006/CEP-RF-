"""Inspect raw CIC-DDoS2019 Parquet metadata and labels for Phase 2.

This script reports file dimensions, schemas, and label distributions only. It
does not clean data, transform features, or modify any files in data/raw/.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import re
from dataclasses import dataclass
from pathlib import Path

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
    """Run schema inspection only."""
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
    inspections = [inspect_parquet_schema(path) for path in parquet_files]
    dataset_summary = build_dataset_summary(inspections)
    column_summary = build_column_summary(inspections)
    schema_comparison = build_schema_comparison(inspections)
    schema_type_differences = build_schema_type_differences(inspections)
    label_distribution = build_label_distribution(inspections, raw_dir)
    label_summary = build_label_summary(label_distribution)
    label_file_mapping = build_label_file_mapping(inspections, label_distribution)

    write_reports(
        report_dir,
        dataset_summary,
        column_summary,
        schema_comparison,
        schema_type_differences,
        label_distribution,
        label_summary,
        label_file_mapping,
    )
    print_results(
        raw_dir,
        dataset_summary,
        schema_comparison,
        schema_type_differences,
        label_distribution,
        label_summary,
    )
    print(f"\nReports written to: {project_relative(report_dir)}")
    print("Raw data was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
