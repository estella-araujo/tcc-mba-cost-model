"""Valida e prepara os datasets de preços e workloads para análise.

Exemplo:
    python run_data_pipeline.py \
        --prices prices.parquet \
        --workloads synthetic_workloads_30d.parquet \
        --output-dir processed_data

O pipeline gera dois Parquets limpos e um relatório JSON de qualidade.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PRICE_REQUIRED = {
    "provider", "region", "instance_type", "payment_model", "currency",
    "resource_category", "unit", "price", "source_api", "retrieved_at",
}

WORKLOAD_REQUIRED = {
    "scenario_id", "workload", "scenario", "timestamp_utc",
    "cpu_capacity_cores", "ram_capacity_gib", "cpu_utilization_pct",
    "cpu_demand_cores", "total_storage_gb", "network_traffic_gb",
    "sla_pct", "rpo_minutes", "rto_minutes",
}


def missing_values(frame: pd.DataFrame) -> dict[str, int]:
    return {column: int(value) for column, value in frame.isna().sum().items() if value}


def validate_schema(frame: pd.DataFrame, required: set[str]) -> list[str]:
    return sorted(required - set(frame.columns))


def clean_prices(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    original_rows = len(frame)
    report: dict[str, Any] = {
        "input_rows": original_rows,
        "missing_schema_columns": validate_schema(frame, PRICE_REQUIRED),
        "missing_values_before_cleaning": missing_values(frame),
    }
    if report["missing_schema_columns"]:
        raise ValueError(f"Schema de preços incompleto: {report['missing_schema_columns']}")

    frame = frame.copy()
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["retrieved_at"] = pd.to_datetime(frame["retrieved_at"], errors="coerce", utc=True)
    if "effective_start" in frame:
        frame["effective_start"] = pd.to_datetime(frame["effective_start"], errors="coerce", utc=True)

    invalid = frame["price"].isna() | (frame["price"] < 0) | frame["retrieved_at"].isna()
    report["invalid_rows_removed"] = int(invalid.sum())
    frame = frame.loc[~invalid].copy()
    before_duplicates = len(frame)
    frame = frame.drop_duplicates()
    report["duplicates_removed"] = before_duplicates - len(frame)
    report["output_rows"] = len(frame)
    report["missing_values_after_cleaning"] = missing_values(frame)
    return frame, report


def clean_workloads(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    original_rows = len(frame)
    report: dict[str, Any] = {
        "input_rows": original_rows,
        "missing_schema_columns": validate_schema(frame, WORKLOAD_REQUIRED),
        "missing_values_before_cleaning": missing_values(frame),
    }
    if report["missing_schema_columns"]:
        raise ValueError(f"Schema de workloads incompleto: {report['missing_schema_columns']}")

    frame = frame.copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], errors="coerce", utc=True)
    numeric_columns = [
        "cpu_capacity_cores", "ram_capacity_gib", "cpu_utilization_pct",
        "cpu_demand_cores", "total_storage_gb", "network_traffic_gb",
        "sla_pct", "rpo_minutes", "rto_minutes",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    invalid = (
        frame["timestamp_utc"].isna()
        | frame["scenario_id"].isna()
        | (frame["cpu_capacity_cores"] < 0)
        | (frame["ram_capacity_gib"] < 0)
        | (frame["cpu_utilization_pct"] < 0)
        | (frame["cpu_utilization_pct"] > 100)
        | (frame["cpu_demand_cores"] < 0)
        | (frame["total_storage_gb"] < 0)
        | (frame["network_traffic_gb"] < 0)
        | (frame["sla_pct"] < 0)
        | (frame["sla_pct"] > 100)
    )
    invalid = invalid.fillna(True)
    report["invalid_rows_removed"] = int(invalid.sum())
    frame = frame.loc[~invalid].copy()

    duplicate_keys = ["scenario_id", "timestamp_utc"]
    before_duplicates = len(frame)
    frame = frame.drop_duplicates(subset=duplicate_keys, keep="first")
    report["duplicates_removed"] = before_duplicates - len(frame)
    report["output_rows"] = len(frame)
    report["missing_values_after_cleaning"] = missing_values(frame)
    return frame, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("processed_data"))
    parser.add_argument("--coverage-report", type=Path, help="relatório gerado pelo collect_prices.py")
    parser.add_argument("--allow-incomplete", action="store_true", help="permite processar cobertura incompleta")
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    workloads = pd.read_parquet(args.workloads)
    coverage = None
    if args.coverage_report:
        coverage = json.loads(args.coverage_report.read_text(encoding="utf-8"))
        if not coverage.get("complete", False) and not args.allow_incomplete:
            raise SystemExit("Cobertura de preços incompleta. Corrija a coleta antes do pipeline.")
    clean_price_frame, price_report = clean_prices(prices)
    clean_workload_frame, workload_report = clean_workloads(workloads)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    price_output = args.output_dir / "prices_clean.parquet"
    workload_output = args.output_dir / "workloads_clean.parquet"
    report_output = args.output_dir / "data_quality_report.json"
    clean_price_frame.to_parquet(price_output, index=False)
    clean_workload_frame.to_parquet(workload_output, index=False)

    report = {
        "pipeline": "run_data_pipeline.py",
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {"prices": str(args.prices), "workloads": str(args.workloads)},
        "coverage_report": coverage,
        "outputs": {"prices": str(price_output), "workloads": str(workload_output)},
        "datasets": {"prices": price_report, "workloads": workload_report},
    }
    report_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Preços limpos: {price_output.resolve()}")
    print(f"Workloads limpos: {workload_output.resolve()}")
    print(f"Relatório de qualidade: {report_output.resolve()}")


if __name__ == "__main__":
    main()
