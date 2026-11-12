"""Gera tabelas e gráficos da análise comparativa dos custos."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


COMPONENTS = [
    "compute_cost",
    "database_storage_cost",
    "object_storage_cost",
    "backup_storage_cost",
    "network_cost",
]


def money(value: float) -> str:
    return f"US$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def font(size: int = 18):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def new_chart(width: int = 1400, height: int = 700):
    image = Image.new("RGB", (width, height), "white")
    return image, ImageDraw.Draw(image)


def axis(draw, left, top, right, bottom, ymax, ylabel):
    draw.line((left, top, left, bottom), fill="#444444", width=2)
    draw.line((left, bottom, right, bottom), fill="#444444", width=2)
    for tick in range(6):
        value = ymax * tick / 5
        y = bottom - (bottom - top) * tick / 5
        draw.line((left, y, right, y), fill="#DDDDDD", width=1)
        draw.text((left - 75, y - 10), f"{value:.0f}", fill="#333333", font=font(14))
    draw.text((20, top), ylabel, fill="#333333", font=font(15))


def load_inputs(candidates_path: Path, workloads_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(candidates_path) if candidates_path.suffix.lower() == ".csv" else pd.read_parquet(candidates_path)
    workloads = pd.read_csv(workloads_path) if workloads_path.suffix.lower() == ".csv" else pd.read_parquet(workloads_path)
    candidates = candidates[candidates["price_status"].eq("complete")].copy()
    candidates["total_cost"] = pd.to_numeric(candidates["total_cost"], errors="coerce")
    candidates = candidates[candidates["total_cost"] > 0].copy()
    return candidates, workloads


def add_ranks(candidates: pd.DataFrame) -> pd.DataFrame:
    keys = ["workload", "scenario"]
    candidates["rank"] = candidates.groupby(keys)["total_cost"].rank(method="dense", ascending=True).astype(int)
    candidates["best_cost"] = candidates.groupby(keys)["total_cost"].transform("min")
    candidates["difference_from_best_pct"] = (candidates["total_cost"] / candidates["best_cost"] - 1) * 100
    return candidates


def create_tables(candidates: pd.DataFrame, workloads: pd.DataFrame, out: Path) -> None:
    best = candidates.loc[candidates["rank"].eq(1)].copy()
    best.to_csv(out / "comparative_best_alternatives.csv", index=False, encoding="utf-8-sig")

    summary = (
        candidates.groupby(["workload", "scenario", "provider"], as_index=False)
        .agg(
            alternatives=("total_cost", "size"),
            minimum_cost=("total_cost", "min"),
            mean_cost=("total_cost", "mean"),
            maximum_cost=("total_cost", "max"),
            median_cost=("total_cost", "median"),
        )
    )
    summary.to_csv(out / "comparative_summary_by_provider.csv", index=False, encoding="utf-8-sig")

    composition = best[["workload", "scenario", "provider", "region", "total_cost", *COMPONENTS]].copy()
    composition.to_csv(out / "comparative_cost_composition.csv", index=False, encoding="utf-8-sig")

    provisioned = best[best["scenario"].str.lower().eq("provisionado")][["workload", "total_cost"]].rename(columns={"total_cost": "provisioned_cost"})
    rightsized = best[best["scenario"].str.lower().eq("rightsized")][["workload", "total_cost"]].rename(columns={"total_cost": "rightsized_cost"})
    savings = provisioned.merge(rightsized, on="workload", how="inner")
    savings["saving_usd"] = savings["provisioned_cost"] - savings["rightsized_cost"]
    savings["saving_pct"] = savings["saving_usd"] / savings["provisioned_cost"] * 100
    savings.to_csv(out / "comparative_rightsizing_savings.csv", index=False, encoding="utf-8-sig")

    traffic = workloads.groupby("scenario_id", as_index=False)["network_traffic_gb"].sum().rename(columns={"network_traffic_gb": "network_gb_30d"})
    candidates = candidates.merge(traffic, on="scenario_id", how="left")
    candidates["fixed_cost"] = candidates["total_cost"] - candidates["network_cost"]
    candidates["network_rate"] = candidates["network_cost"] / candidates["network_gb_30d"].replace(0, pd.NA)
    break_even_rows = []
    for (workload, scenario), group in candidates.groupby(["workload", "scenario"]):
        group = group.sort_values("total_cost")
        if len(group) < 2:
            continue
        a, b = group.iloc[0], group.iloc[1]
        denominator = float(a["network_rate"] - b["network_rate"])
        if denominator == 0 or pd.isna(denominator):
            continue
        threshold = float((b["fixed_cost"] - a["fixed_cost"]) / denominator)
        if threshold > 0:
            break_even_rows.append({
                "workload": workload,
                "scenario": scenario,
                "alternative_a": f"{a['provider']} / {a['region']}",
                "alternative_b": f"{b['provider']} / {b['region']}",
                "break_even_network_gb_30d": threshold,
                "observed_network_gb_30d": float(a["network_gb_30d"]),
            })
    pd.DataFrame(break_even_rows).to_csv(out / "comparative_break_even.csv", index=False, encoding="utf-8-sig")


def chart_best_cost(best: pd.DataFrame, out: Path) -> None:
    plot = best.copy()
    image, draw = new_chart()
    left, top, right, bottom = 120, 100, 1340, 580
    ymax = max(plot["total_cost"].max() * 1.2, 1)
    axis(draw, left, top, right, bottom, ymax, "US$ / 30 dias")
    colors = {"aws": "#FF9900", "azure": "#0078D4", "gcp": "#34A853"}
    slot = (right - left) / len(plot)
    for i, (_, row) in enumerate(plot.reset_index(drop=True).iterrows()):
        x0 = left + i * slot + slot * 0.2
        x1 = left + i * slot + slot * 0.8
        y = bottom - (bottom - top) * row["total_cost"] / ymax
        draw.rectangle((x0, y, x1, bottom), fill=colors.get(row["provider"], "#666666"))
        draw.text((x0, y - 28), f"{row['total_cost']:.0f}", fill="#222222", font=font(14))
        label = f"{row['workload'].split(' de ')[0]}\n{row['scenario'].title()}"
        draw.multiline_text((x0, bottom + 12), label, fill="#333333", font=font(13), align="center")
    draw.text((470, 25), "Menor custo por workload e cenário", fill="#222222", font=font(24))
    image.save(out / "comparative_best_cost.png")


def chart_composition(best: pd.DataFrame, out: Path) -> None:
    plot = best.copy()
    image, draw = new_chart()
    left, top, right, bottom_y = 120, 100, 1340, 580
    ymax = max(plot["total_cost"].max() * 1.2, 1)
    axis(draw, left, top, right, bottom_y, ymax, "US$ / 30 dias")
    bottoms = [0.0] * len(plot)
    labels = {
        "compute_cost": "Computação",
        "database_storage_cost": "Banco de dados",
        "object_storage_cost": "Objetos",
        "backup_storage_cost": "Backup",
        "network_cost": "Rede",
    }
    colors = ["#2F5597", "#70AD47", "#A5A5A5", "#FFC000", "#C00000"]
    slot = (right - left) / len(plot)
    for component, color in zip(COMPONENTS, colors):
        for i, (_, row) in enumerate(plot.reset_index(drop=True).iterrows()):
            value = float(row[component] or 0)
            x0 = left + i * slot + slot * 0.2
            x1 = left + i * slot + slot * 0.8
            y1 = bottom_y - (bottom_y - top) * bottoms[i] / ymax
            y0 = bottom_y - (bottom_y - top) * (bottoms[i] + value) / ymax
            draw.rectangle((x0, y0, x1, y1), fill=color)
            bottoms[i] += value
    draw.text((430, 25), "Composição das melhores alternativas", fill="#222222", font=font(24))
    for i, (_, row) in enumerate(plot.reset_index(drop=True).iterrows()):
        x = left + i * slot + slot * 0.2
        draw.multiline_text((x, bottom_y + 12), f"{row['workload'].split(' de ')[0]}\n{row['scenario'].title()}", fill="#333333", font=font(13))
    legend_x = 120
    for component, color in zip(COMPONENTS, colors):
        draw.rectangle((legend_x, 625, legend_x + 18, 643), fill=color)
        draw.text((legend_x + 25, 623), labels[component], fill="#333333", font=font(13))
        legend_x += 235
    image.save(out / "comparative_cost_composition.png")


def chart_savings(savings: pd.DataFrame, out: Path) -> None:
    image, draw = new_chart(1100, 600)
    left, top, right, bottom = 140, 100, 1000, 480
    ymax = max(savings["saving_pct"].max() * 1.25, 1)
    axis(draw, left, top, right, bottom, ymax, "% de redução")
    slot = (right - left) / len(savings)
    for i, (_, row) in enumerate(savings.iterrows()):
        x0 = left + i * slot + slot * 0.2
        x1 = left + i * slot + slot * 0.8
        y = bottom - (bottom - top) * row["saving_pct"] / ymax
        draw.rectangle((x0, y, x1, bottom), fill="#4472C4")
        draw.text((x0, y - 28), f"{row['saving_pct']:.1f}%", fill="#222222", font=font(15))
        draw.multiline_text((x0, bottom + 12), row["workload"].split(" de ")[0], fill="#333333", font=font(14))
    draw.text((330, 25), "Efeito do cenário Rightsized sobre o custo", fill="#222222", font=font(24))
    image.save(out / "comparative_rightsizing_savings.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates, workloads = load_inputs(args.candidates, args.workloads)
    candidates = add_ranks(candidates)
    create_tables(candidates, workloads, args.output_dir)
    best = candidates[candidates["rank"].eq(1)].copy()
    chart_best_cost(best, args.output_dir)
    chart_composition(best, args.output_dir)
    savings = pd.read_csv(args.output_dir / "comparative_rightsizing_savings.csv")
    chart_savings(savings, args.output_dir)
    print(f"Alternativas analisadas: {len(candidates)}")
    print(f"Tabelas e gráficos salvos em: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
