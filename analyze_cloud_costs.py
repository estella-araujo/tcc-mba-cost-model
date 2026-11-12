"""Calcula custos e métricas comparativas para os workloads sintéticos.

Entradas:
  - prices_clean.parquet: preços horários normalizados;
  - workloads_clean.parquet: observações horárias dos workloads;
  - cost_model_config.json: catálogo de instâncias e preços não computacionais.

Exemplo:
  python analyze_cloud_costs.py --prices processed_data/prices_clean.parquet \
      --workloads processed_data/workloads_clean.parquet \
      --config cost_model_config.json --output-dir analysis_results
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def canonical_payment(value: str) -> str:
    text = str(value).strip().lower()
    if text in {"on_demand", "on-demand", "ondemand", "consumption"}:
        return "on_demand"
    if text in {"reserved", "reservation", "reserved/committed", "committed", "savings"}:
        return "reserved"
    if text in {"spot", "spot/preemptible", "preemptible"}:
        return "spot"
    return text


def hourly_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Seleciona preços compatíveis com cobrança horária de computação."""
    frame = prices.copy()
    for col in ("provider", "region", "instance_type", "payment_model"):
        frame[col] = normalize_text(frame[col])
    frame["payment_model"] = frame["payment_model"].map(canonical_payment)
    frame["unit_text"] = frame["unit"].fillna("").astype(str).str.lower()
    frame["is_hourly"] = frame["unit_text"].str.contains("hr|hour|hora", regex=True, na=False)
    frame["analysis_price"] = pd.to_numeric(frame["price"], errors="coerce")

    # Algumas ofertas reservadas da Azure aparecem como valor total do prazo,
    # embora a unidade informada seja "1 Hour". Converte-se esse valor para
    # uma tarifa horária efetiva somente quando o prazo está explícito.
    term_hours = frame["term_length"].astype(str).str.lower().map({
        "1 yr": 8760.0, "1 year": 8760.0,
        "3 yr": 26280.0, "3 year": 26280.0,
    })
    reserved_total = (
        (frame["payment_model"] == "reserved")
        & frame["is_hourly"]
        & (frame["analysis_price"] > 1)
        & term_hours.notna()
    )
    frame.loc[reserved_total, "analysis_price"] = (
        frame.loc[reserved_total, "analysis_price"] / term_hours.loc[reserved_total]
    )
    reserved_uninterpretable = (
        (frame["payment_model"] == "reserved")
        & frame["is_hourly"]
        & (frame["analysis_price"] > 1)
        & term_hours.isna()
    )
    frame.loc[reserved_uninterpretable, "analysis_price"] = np.nan
    return frame


def find_compute_price(prices: pd.DataFrame, provider: str, region: str,
                       instance: str, payment_model: str) -> float | None:
    mask = (
        (prices["provider"] == provider.lower())
        & (prices["region"] == region.lower())
        & (prices["instance_type"] == instance.lower())
        & (prices["payment_model"] == canonical_payment(payment_model))
        & prices["is_hourly"]
    )
    values = pd.to_numeric(prices.loc[mask, "analysis_price"], errors="coerce")
    values = values[values > 0].dropna()
    return float(values.min()) if not values.empty else None


def find_resource_price(prices: pd.DataFrame, provider: str, region: str,
                        category: str) -> float | None:
    mask = (
        (prices["provider"] == provider.lower())
        & (prices["region"] == region.lower())
        & (prices["resource_category"] == category.lower())
    )
    values = pd.to_numeric(prices.loc[mask, "price"], errors="coerce")
    values = values[values > 0].dropna()
    return float(values.min()) if not values.empty else None


def calculate_candidate(workload: pd.DataFrame, provider_cfg: dict[str, Any],
                        region: str, instance_cfg: dict[str, Any],
                        payment_model: str, prices: pd.DataFrame,
                        config: dict[str, Any],
                        excluded_categories: set[tuple[str, str]]) -> dict[str, Any]:
    provider = provider_cfg["vendor"].lower()
    instance = instance_cfg["name"]
    compute_price = find_compute_price(prices, provider, region, instance, payment_model)
    output: dict[str, Any] = {
        "provider": provider, "region": region, "instance_type": instance,
        "payment_model": payment_model, "compute_price_per_hour": compute_price,
    }
    if compute_price is None:
        output["price_status"] = "missing_compute_price"
        return output

    hourly = []
    for _, row in workload.iterrows():
        cpu_demand = float(row["cpu_demand_cores"])
        ram_demand = float(row["ram_capacity_gib"])
        cpu_instances = math.ceil(cpu_demand / float(instance_cfg["vcpu"]))
        ram_instances = math.ceil(ram_demand / float(instance_cfg["ram_gib"]))
        count = max(1, cpu_instances, ram_instances)
        hourly.append({
            "timestamp_utc": row["timestamp_utc"],
            "instance_count": count,
            "compute_cost": count * compute_price,
            "cpu_demand_cores": cpu_demand,
            "ram_demand_gib": ram_demand,
            "network_traffic_gb": float(row["network_traffic_gb"]),
        })
    hourly_frame = pd.DataFrame(hourly)

    first = workload.iloc[0]
    storage_rates = {
        # Nenhuma tarifa é inventada ou substituída por valor de configuração.
        # Se a API não retornar a combinação, o resultado fica incompleto.
        "database_storage_cost": None if (provider, "database_storage") in excluded_categories else find_resource_price(prices, provider, region, "database_storage"),
        "object_storage_cost": None if (provider, "object_storage") in excluded_categories else find_resource_price(prices, provider, region, "object_storage"),
        "backup_storage_cost": None if (provider, "backup_storage") in excluded_categories else find_resource_price(prices, provider, region, "backup_storage"),
        "network_cost": None if (provider, "network_egress") in excluded_categories else find_resource_price(prices, provider, region, "network_egress"),
    }
    def amount_times_rate(amount: float, rate: float | None) -> float:
        return float("nan") if rate is None else amount * rate

    storage_costs = {
        "database_storage_cost": amount_times_rate(
            float(first.get("database_storage_gb", 0) or 0), storage_rates["database_storage_cost"]
        ),
        "object_storage_cost": amount_times_rate(
            float(first.get("object_storage_gb", 0) or 0), storage_rates["object_storage_cost"]
        ),
        "backup_storage_cost": amount_times_rate(
            float(first.get("backup_storage_gb", 0) or 0), storage_rates["backup_storage_cost"]
        ),
        "network_cost": amount_times_rate(
            hourly_frame["network_traffic_gb"].sum(), storage_rates["network_cost"]
        ),
    }
    excluded_rate_keys = [
        key for key, category in {
            "database_storage_cost": "database_storage",
            "object_storage_cost": "object_storage",
            "backup_storage_cost": "backup_storage",
            "network_cost": "network_egress",
        }.items() if (provider, category) in excluded_categories
    ]
    required_rates = [key for key, value in storage_rates.items() if value is None and key not in excluded_rate_keys]
    included_storage_cost = sum(value for value in storage_costs.values() if pd.notna(value))
    output.update({
        "compute_cost": hourly_frame["compute_cost"].sum(),
        **storage_costs,
        "total_cost": hourly_frame["compute_cost"].sum() + included_storage_cost if not required_rates else np.nan,
        "missing_resource_prices": ",".join(required_rates),
        "excluded_resource_prices": ",".join(excluded_rate_keys),
        "avg_cpu_demand_cores": hourly_frame["cpu_demand_cores"].mean(),
        "peak_cpu_demand_cores": hourly_frame["cpu_demand_cores"].max(),
        "avg_instance_count": hourly_frame["instance_count"].mean(),
        "peak_instance_count": hourly_frame["instance_count"].max(),
        "cost_per_core_hour": hourly_frame["compute_cost"].sum() / max(hourly_frame["cpu_demand_cores"].sum(), 1e-9),
        "cost_per_gib_hour": hourly_frame["compute_cost"].sum() / max(hourly_frame["ram_demand_gib"].sum(), 1e-9),
        "price_status": "complete_with_exclusions" if excluded_rate_keys and not required_rates else ("complete" if not required_rates else "missing_resource_prices"),
    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_results"))
    parser.add_argument("--coverage-report", type=Path, help="relatório gerado pelo collect_prices.py")
    parser.add_argument("--allow-incomplete", action="store_true", help="gera candidatos mesmo com cobertura incompleta")
    args = parser.parse_args()

    coverage = None
    excluded_categories: set[tuple[str, str]] = set()
    if args.coverage_report:
        coverage = json.loads(args.coverage_report.read_text(encoding="utf-8"))
        excluded_categories = {
            (item["provider"].lower(), item["resource_category"].lower())
            for item in coverage.get("excluded", [])
        }
        if not coverage.get("complete", False) and not args.allow_incomplete:
            raise SystemExit("Cobertura de preços incompleta. A modelagem foi interrompida.")

    prices = hourly_prices(pd.read_parquet(args.prices))
    workloads = pd.read_parquet(args.workloads)
    config = read_config(args.config)
    workloads["timestamp_utc"] = pd.to_datetime(workloads["timestamp_utc"], utc=True)
    rows: list[dict[str, Any]] = []

    for workload_id, workload_frame in workloads.groupby("scenario_id", sort=True):
        workload_name = workload_frame["workload"].iloc[0]
        scenario = workload_frame["scenario"].iloc[0]
        for provider_cfg in config["providers"]:
            for region in provider_cfg["regions"]:
                for instance_cfg in provider_cfg["instances"]:
                    for payment_model in provider_cfg["payment_models"]:
                        result = calculate_candidate(
                            workload_frame, provider_cfg, region,
                            instance_cfg, payment_model, prices, config,
                            excluded_categories,
                        )
                        result.update({
                            "scenario_id": workload_id,
                            "workload": workload_name,
                            "scenario": scenario,
                        })
                        rows.append(result)

    results = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_parquet(args.output_dir / "cost_candidates.parquet", index=False)

    complete = results[results["price_status"].isin(["complete", "complete_with_exclusions"])].copy()
    if not complete.empty:
        complete["rank"] = complete.groupby("scenario_id")["total_cost"].rank(method="min")
        complete["saving_vs_provisioned_pct"] = np.nan
        for workload in complete["workload"].unique():
            for provider in complete["provider"].unique():
                provisioned = complete[(complete["workload"] == workload) & (complete["provider"] == provider) & (complete["scenario"] == "provisionado")]["total_cost"].min()
                mask = (complete["workload"] == workload) & (complete["provider"] == provider) & (complete["scenario"] == "rightsized")
                if pd.notna(provisioned) and provisioned > 0:
                    complete.loc[mask, "saving_vs_provisioned_pct"] = (provisioned - complete.loc[mask, "total_cost"]) / provisioned * 100
        complete.to_parquet(args.output_dir / "cost_results_complete.parquet", index=False)
        best = complete.loc[complete.groupby("scenario_id")["total_cost"].idxmin()].copy()
        best.to_csv(args.output_dir / "best_alternatives.csv", index=False)
    else:
        best = pd.DataFrame()

    report = {
        "candidate_rows": len(results),
        "complete_rows": len(complete),
        "incomplete_rows": int((results["price_status"] != "complete").sum()),
        "best_alternatives_rows": len(best),
        "price_coverage_complete": None if coverage is None else bool(coverage.get("complete", False)),
        "excluded_price_categories": [] if coverage is None else coverage.get("excluded", []),
        "note": "Custos totais só são calculados quando os preços de computação, armazenamento e rede estão configurados.",
    }
    (args.output_dir / "analysis_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
