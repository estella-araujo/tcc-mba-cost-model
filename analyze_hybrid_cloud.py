"""Avalia cenários híbridos com computação e dados em provedores distintos."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


STORAGE_COMPONENTS = ["database_storage_cost", "object_storage_cost", "backup_storage_cost"]
PROVIDERS = ["aws", "azure", "gcp"]
SHARES = [0.0, 0.10, 0.30, 0.50, 0.80]


def read_dataset(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def best_row(group: pd.DataFrame, provider: str, column: str) -> pd.Series | None:
    subset = group[group["provider"].eq(provider)].copy()
    subset[column] = pd.to_numeric(subset[column], errors="coerce")
    subset = subset[subset[column].notna() & (subset[column] >= 0)]
    if subset.empty:
        return None
    return subset.loc[subset[column].idxmin()]


def scenario_traffic(workloads: pd.DataFrame) -> pd.DataFrame:
    return workloads.groupby("scenario_id", as_index=False)["network_traffic_gb"].sum().rename(columns={"network_traffic_gb": "network_gb_30d"})


def build_hybrid_candidates(candidates: pd.DataFrame, traffic: pd.DataFrame, share: float) -> list[dict]:
    rows: list[dict] = []
    for (workload, scenario), group in candidates.groupby(["workload", "scenario"]):
        scenario_id = group["scenario_id"].iloc[0]
        traffic_row = traffic[traffic["scenario_id"].eq(scenario_id)]
        network_gb = float(traffic_row["network_gb_30d"].iloc[0]) if not traffic_row.empty else 0.0
        single = group.loc[group["total_cost"].idxmin()]

        for compute_provider in PROVIDERS:
            compute = best_row(group, compute_provider, "compute_cost")
            network = best_row(group, compute_provider, "network_cost")
            if compute is None or network is None:
                continue
            compute_cost = float(compute["compute_cost"])
            internet_cost = float(network["network_cost"])
            network_rate = internet_cost / network_gb if network_gb > 0 else 0.0
            for storage_provider in PROVIDERS:
                if storage_provider == compute_provider:
                    continue
                storage_rows = [best_row(group, storage_provider, component) for component in STORAGE_COMPONENTS]
                if any(row is None for row in storage_rows):
                    continue
                storage_costs = {component: float(row[component]) for component, row in zip(STORAGE_COMPONENTS, storage_rows)}
                cross_cloud_cost = network_gb * share * network_rate
                total = compute_cost + sum(storage_costs.values()) + internet_cost + cross_cloud_cost
                rows.append({
                    "workload": workload,
                    "scenario": scenario,
                    "scenario_id": scenario_id,
                    "compute_provider": compute_provider,
                    "storage_provider": storage_provider,
                    "compute_region": compute["region"],
                    "storage_region": storage_rows[0]["region"],
                    "cross_cloud_traffic_share": share,
                    "network_gb_30d": network_gb,
                    "compute_cost": compute_cost,
                    **storage_costs,
                    "internet_egress_cost": internet_cost,
                    "cross_cloud_transfer_cost": cross_cloud_cost,
                    "total_cost": total,
                    "best_single_cloud_cost": float(single["total_cost"]),
                    "difference_vs_single_cloud_pct": (total / float(single["total_cost"]) - 1) * 100,
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = read_dataset(args.candidates)
    candidates = candidates[candidates["price_status"].eq("complete")].copy()
    candidates["total_cost"] = pd.to_numeric(candidates["total_cost"], errors="coerce")
    candidates = candidates[candidates["total_cost"] > 0]
    workloads = read_dataset(args.workloads)
    traffic = scenario_traffic(workloads)

    all_rows = []
    for share in SHARES:
        all_rows.extend(build_hybrid_candidates(candidates, traffic, share))
    hybrid = pd.DataFrame(all_rows)
    hybrid["rank"] = hybrid.groupby(["workload", "scenario", "cross_cloud_traffic_share"])["total_cost"].rank(method="dense", ascending=True).astype(int)
    hybrid.to_csv(args.output_dir / "hybrid_candidates.csv", index=False, encoding="utf-8-sig")

    best = hybrid[hybrid["rank"].eq(1)].copy()
    best.to_csv(args.output_dir / "hybrid_best_by_scenario.csv", index=False, encoding="utf-8-sig")
    sensitivity = best[["workload", "scenario", "cross_cloud_traffic_share", "compute_provider", "storage_provider", "total_cost", "best_single_cloud_cost", "difference_vs_single_cloud_pct"]]
    sensitivity.to_csv(args.output_dir / "hybrid_sensitivity.csv", index=False, encoding="utf-8-sig")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "assumption": "A fração definida de tráfego intercloud é adicional ao tráfego de saída e é tarifada pela taxa de saída do provedor de computação.",
        "traffic_shares": SHARES,
        "hybrid_rows": int(len(hybrid)),
        "best_rows": int(len(best)),
        "output_files": ["hybrid_candidates.csv", "hybrid_best_by_scenario.csv", "hybrid_sensitivity.csv"],
    }
    (args.output_dir / "hybrid_analysis_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Cenários híbridos calculados: {len(hybrid)}")
    print(f"Resultados salvos em: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
