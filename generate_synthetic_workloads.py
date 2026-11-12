"""Gera cenários sintéticos horários de workloads para 30 dias.

Saída padrão: synthetic_workloads_30d.parquet

As faixas foram transcritas das tabelas fornecidas no TCC. O cenário
Rightsized altera CPU e RAM; armazenamento, tráfego e requisitos de
confiabilidade permanecem iguais aos do cenário Provisionado.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


HOURS = 30 * 24
SEED = 20261018


WORKLOADS = {
    "ecommerce_medio": {
        "name": "E-commerce de Médio Porte",
        "scenarios": {
            "provisionado": {"cpu": (8, 16), "ram": (32, 64)},
            "rightsized": {"cpu": (5, 10), "ram": (19, 38)},
        },
        "database_storage_gb": (100, 500),
        "object_storage_gb": (1000, 4000),
        "telemetry_storage_gb": (0, 0),
        "data_storage_gb": (0, 0),
        "backup_storage_gb": (0, 0),
        "total_storage_gb": (2000, 5000),
        "network_monthly_gb": (3000, 8000),
        "baseline_pct": (10, 25),
        "peak_pct": (60, 80),
        "peak_factor": (5, 10),
        "sla_pct": 99.95,
        "rpo_minutes": 60,
        "rto_minutes": 15,
        "profile": "ecommerce",
    },
    "fintech_pequeno": {
        "name": "Fintech de Pequeno Porte",
        "scenarios": {
            "provisionado": {"cpu": (4, 8), "ram": (16, 32)},
            "rightsized": {"cpu": (2, 5), "ram": (10, 19)},
        },
        "database_storage_gb": (200, 800),
        "object_storage_gb": (300, 1500),
        "telemetry_storage_gb": (0, 0),
        "data_storage_gb": (0, 0),
        "backup_storage_gb": (0, 0),
        "total_storage_gb": (500, 2000),
        "network_monthly_gb": (500, 1000),
        "baseline_pct": (30, 50),
        "peak_pct": (70, 90),
        "peak_factor": (2, 3),
        "sla_pct": 99.99,
        "rpo_minutes": 60,
        "rto_minutes": 15,
        "profile": "fintech",
    },
    "manufatura_medio": {
        "name": "Manufatura/Indústria de Médio Porte",
        "scenarios": {
            "provisionado": {"cpu": (16, 32), "ram": (64, 128)},
            "rightsized": {"cpu": (10, 21), "ram": (42, 83)},
        },
        "database_storage_gb": (0, 0),
        "object_storage_gb": (0, 0),
        "telemetry_storage_gb": (5000, 10000),
        "data_storage_gb": (2000, 5000),
        "backup_storage_gb": (3000, 5000),
        "total_storage_gb": (10000, 20000),
        "network_monthly_gb": (1000, 3000),
        "baseline_pct": (50, 70),
        "peak_pct": (70, 85),
        "peak_factor": (1, 1),
        "sla_pct": 99.99,
        "rpo_minutes": 1440,
        "rto_minutes": 240,
        "profile": "manufacturing",
    },
}


def draw(rng: np.random.Generator, value_range: tuple[float, float]) -> float:
    """Sorteia um valor uniforme dentro de uma faixa inclusiva."""
    low, high = value_range
    if low == high:
        return float(low)
    return float(rng.uniform(low, high))


def hourly_profile(profile: str, timestamp: datetime) -> float:
    """Retorna intensidade relativa da demanda para uma hora."""
    hour = timestamp.hour
    weekend = timestamp.weekday() >= 5
    if profile == "ecommerce":
        weights = [0.65] * 8 + [0.95] * 4 + [1.25] * 3 + [1.0] * 4 + [1.45] * 4 + [0.85]
        value = weights[hour] * (0.78 if weekend else 1.0)
    elif profile == "fintech":
        value = 1.35 if 8 <= hour <= 17 else 0.72
        if weekend:
            value *= 0.55
    else:  # Produção diurna e pico de processamento noturno.
        if 7 <= hour <= 18:
            value = 1.0
        elif hour in (22, 23, 0, 1, 2):
            value = 1.25
        else:
            value = 0.65
    return value


def build_rows(rng: np.random.Generator, start: datetime) -> list[dict]:
    rows: list[dict] = []
    for workload_id, workload in WORKLOADS.items():
        # Os recursos de armazenamento e a rede são compartilhados pelos dois
        # cenários para isolar o efeito da redução de CPU/RAM.
        storage = {
            key: round(draw(rng, workload[key]), 2)
            for key in (
                "database_storage_gb", "object_storage_gb", "telemetry_storage_gb",
                "data_storage_gb", "backup_storage_gb", "total_storage_gb",
            )
        }
        monthly_network = draw(rng, workload["network_monthly_gb"])
        baseline = draw(rng, workload["baseline_pct"])
        peak = draw(rng, workload["peak_pct"])
        peak_factor = draw(rng, workload["peak_factor"])

        for scenario, limits in workload["scenarios"].items():
            cpu_capacity = round(draw(rng, limits["cpu"]), 2)
            ram_capacity = round(draw(rng, limits["ram"]), 2)
            timestamps = [start + timedelta(hours=i) for i in range(HOURS)]
            raw_profile = np.array([hourly_profile(workload["profile"], ts) for ts in timestamps])
            # Normalização para preservar a faixa baseline-pico da tabela.
            normalized = (raw_profile - raw_profile.min()) / (raw_profile.max() - raw_profile.min())
            noise = rng.normal(0, 2.0, HOURS)
            utilization = np.clip(baseline + normalized * (peak - baseline) + noise, 0, 100)
            traffic_weights = np.maximum(raw_profile + rng.normal(0, 0.03, HOURS), 0.05)
            hourly_network = traffic_weights / traffic_weights.sum() * monthly_network

            for i, timestamp in enumerate(timestamps):
                rows.append({
                    "scenario_id": f"{workload_id}_{scenario}",
                    "workload": workload["name"],
                    "scenario": scenario,
                    "timestamp_utc": timestamp.isoformat(),
                    "day": i // 24 + 1,
                    "hour": timestamp.hour,
                    "weekday": timestamp.strftime("%A"),
                    "cpu_capacity_cores": cpu_capacity,
                    "ram_capacity_gib": ram_capacity,
                    "cpu_utilization_pct": round(float(utilization[i]), 2),
                    "cpu_demand_cores": round(cpu_capacity * utilization[i] / 100, 2),
                    "database_storage_gb": storage["database_storage_gb"],
                    "object_storage_gb": storage["object_storage_gb"],
                    "telemetry_storage_gb": storage["telemetry_storage_gb"],
                    "data_storage_gb": storage["data_storage_gb"],
                    "backup_storage_gb": storage["backup_storage_gb"],
                    "total_storage_gb": storage["total_storage_gb"],
                    "network_traffic_gb": round(float(hourly_network[i]), 4),
                    "network_monthly_reference_gb": round(monthly_network, 2),
                    "baseline_cpu_pct": round(baseline, 2),
                    "peak_cpu_pct": round(peak, 2),
                    "peak_factor": round(peak_factor, 2),
                    "sla_pct": workload["sla_pct"],
                    "rpo_minutes": workload["rpo_minutes"],
                    "rto_minutes": workload["rto_minutes"],
                    "random_seed": SEED,
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("synthetic_workloads_30d.parquet"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    frame = pd.DataFrame(build_rows(rng, start))
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(f"{len(frame)} registros gravados em {args.output.resolve()}")
    print(frame.groupby(["workload", "scenario"]).size().to_string())


if __name__ == "__main__":
    main()
