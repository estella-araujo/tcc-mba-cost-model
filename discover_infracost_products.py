"""Descobre combinações válidas de serviço/categoria na Cloud Pricing API.

Uso:
    $env:INFRACOST_API_KEY = "SUA_CHAVE"
    python discover_infracost_products.py --output infracost_discovery.json

O arquivo gerado contém apenas metadados e amostras retornadas pela API. Ele
serve para corrigir os filtros de coleta; não cria preços nem substitui a
coleta oficial.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


API_URL = "https://pricing.api.infracost.io/graphql"

REGIONS = {
    "aws": ["us-east-1", "sa-east-1", "eu-west-1"],
    "azure": ["eastus", "brazilsouth", "westeurope"],
    "gcp": ["us-east1", "southamerica-east1", "europe-west1"],
}

# Combinações candidatas. A API será a fonte de verdade: somente as que
# retornarem produtos serão aproveitadas na configuração final.
CANDIDATES = {
    "aws": {
        "database_storage": [("AmazonRDS", x) for x in ["Database Storage", "Storage", "Provisioned Throughput"]],
        "object_storage": [("AmazonS3", x) for x in ["Storage", "Data Transfer", "Requests"]],
        "backup_storage": [(x, y) for x in ["AWSBackup", "AmazonS3", "AmazonEC2"] for y in ["Backup", "Storage", "Data Transfer"]],
        "network_egress": [(x, y) for x in ["AmazonEC2", "AmazonCloudFront", "AWSDataTransfer"] for y in ["Data Transfer", "Data Transfer Out", "Network"]],
    },
    "azure": {
        "database_storage": [(x, y) for x in ["SQL Database", "Azure SQL Database", "Storage"] for y in ["Storage", "Databases"]],
        "object_storage": [(x, y) for x in ["Storage", "Storage Accounts", "Azure Blob Storage"] for y in ["Storage", "Data Transfer"]],
        "backup_storage": [(x, y) for x in ["Backup", "Recovery Services", "Storage"] for y in ["Storage", "Backup"]],
        "network_egress": [(x, y) for x in ["Bandwidth", "Networking", "Virtual Network"] for y in ["Data Transfer", "Bandwidth", "Networking"]],
    },
    "gcp": {
        "database_storage": [(x, y) for x in ["Cloud SQL", "Cloud Databases"] for y in ["Storage", "Database"]],
        "object_storage": [(x, y) for x in ["Cloud Storage", "Storage"] for y in ["Storage", "Data Transfer", "Network"]],
        "backup_storage": [(x, y) for x in ["Cloud Storage", "Backup", "Compute Engine"] for y in ["Storage", "Backup"]],
        "network_egress": [(x, y) for x in ["Compute Engine", "Cloud Storage", "Virtual Private Cloud", "Network"] for y in ["Data Transfer", "Network", "Networking"]],
    },
}


def gql(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def request_graphql(query: str, api_key: str) -> dict[str, Any]:
    response = requests.post(
        API_URL,
        headers={"X-Api-Key": api_key, "Content-Type": "application/json"},
        json={"query": query},
        timeout=120,
    )
    if response.status_code == 403:
        raise RuntimeError("403 Forbidden: a chave não foi autorizada pela Cloud Pricing API.")
    response.raise_for_status()
    body = response.json()
    if body.get("errors"):
        raise RuntimeError(json.dumps(body["errors"], ensure_ascii=False))
    return body.get("data", {})


def discover(vendor: str, service: str, family: str, region: str, api_key: str) -> dict[str, Any]:
    query = f"""query {{ products(filter: {{
        vendorName: {gql(vendor)}
        service: {gql(service)}
        productFamily: {gql(family)}
        region: {gql(region)}
    }}) {{
        attributes {{ key value }}
        prices {{ USD unit description purchaseOption termPurchaseOption termLength }}
    }} }}"""
    products = request_graphql(query, api_key).get("products", [])
    samples = []
    for product in products[:5]:
        attrs = {item.get("key"): item.get("value") for item in product.get("attributes", [])}
        prices = [
            {
                "USD": p.get("USD"),
                "unit": p.get("unit"),
                "description": p.get("description"),
                "purchaseOption": p.get("purchaseOption"),
                "termPurchaseOption": p.get("termPurchaseOption"),
                "termLength": p.get("termLength"),
            }
            for p in product.get("prices", [])[:3]
        ]
        samples.append({"attributes": attrs, "prices": prices})
    return {"product_count": len(products), "samples": samples}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("infracost_discovery.json"))
    args = parser.parse_args()
    api_key = os.environ.get("INFRACOST_API_KEY")
    if not api_key:
        raise SystemExit("Defina INFRACOST_API_KEY antes de executar o diagnóstico.")

    result: dict[str, Any] = {
        "api_url": API_URL,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "results": [],
    }
    for vendor, categories in CANDIDATES.items():
        for category, combinations in categories.items():
            for service, family in combinations:
                for region in REGIONS[vendor]:
                    print(f"Testando {vendor} / {category} / {service} / {family} / {region}")
                    try:
                        found = discover(vendor, service, family, region, api_key)
                        result["results"].append({
                            "provider": vendor,
                            "resource_category": category,
                            "service": service,
                            "product_family": family,
                            "region": region,
                            **found,
                        })
                    except Exception as exc:
                        result["results"].append({
                            "provider": vendor,
                            "resource_category": category,
                            "service": service,
                            "product_family": family,
                            "region": region,
                            "error": str(exc),
                        })
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    hits = [row for row in result["results"] if row.get("product_count", 0) > 0]
    print(f"Combinações com produtos: {len(hits)}")
    print(f"Diagnóstico salvo em: {args.output.resolve()}")


if __name__ == "__main__":
    main()
