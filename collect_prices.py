"""Coleta preços via Infracost e AWS Price List.

Uso:
    $env:INFRACOST_API_KEY = "SUA_CHAVE"
    python collect_prices.py --config config_prices.json --output prices.parquet

O arquivo de configuração é orientado a recursos. Isso permite ajustar os
filtros GraphQL às nomenclaturas específicas de cada provedor sem alterar o
código principal.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


API_URL = "https://pricing.api.infracost.io/graphql"
AWS_BULK_BASE = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws"
AWS_REGION_LOCATIONS = {
    "us-east-1": "US East (N. Virginia)",
    "sa-east-1": "South America (Sao Paulo)",
    "eu-west-1": "Europe (Ireland)",
}
COLUMNS = [
    "provider", "resource_category", "service", "product_family", "region",
    "resource_key", "instance_type", "operating_system", "payment_model",
    "payment_model_raw", "term_length", "currency", "unit", "price",
    "price_description", "start_usage_amount", "end_usage_amount",
    "product_attributes", "source_api", "source_url", "retrieved_at",
]


DEFAULT_CONFIG = {
    "config_version": 2,
    "resources": [
        {
            "resource_category": "compute",
            "vendor": "aws", "service": "AmazonEC2",
            "product_family": "Compute Instance",
            "regions": ["us-east-1", "sa-east-1", "eu-west-1"],
            "resources": [
                {"key": "t3.medium", "attributes": {"instanceType": "t3.medium", "operatingSystem": "Linux", "tenancy": "Shared", "capacitystatus": "Used", "preInstalledSw": "NA"}},
                {"key": "m5.large", "attributes": {"instanceType": "m5.large", "operatingSystem": "Linux", "tenancy": "Shared", "capacitystatus": "Used", "preInstalledSw": "NA"}}
            ],
            "purchase_options": ["on_demand", "reserved", "spot"]
        },
        {
            "resource_category": "compute",
            "vendor": "azure", "service": "Virtual Machines",
            "product_family": "Compute",
            "regions": ["eastus", "brazilsouth", "westeurope"],
            "resources": [
                {"key": "Standard_B2s", "attributes": {"armSkuName": "Standard_B2s"}},
                {"key": "Standard_D2s_v5", "attributes": {"armSkuName": "Standard_D2s_v5"}}
            ],
            "purchase_options": ["Consumption", "Reservation"]
        },
        {
            "resource_category": "compute",
            "vendor": "gcp", "service": "Compute Engine",
            "product_family": "Compute Instance",
            "regions": ["us-east1", "southamerica-east1", "europe-west1"],
            "resources": [
                {"key": "e2-standard-2", "attributes": {"machineType": "e2-standard-2"}},
                {"key": "n2-standard-2", "attributes": {"machineType": "n2-standard-2"}}
            ],
            "purchase_options": ["on_demand", "preemptible"]
        },
        {
            "resource_category": "database_storage",
            "vendor": "aws", "service": "AmazonRDS",
            "product_family": "Database Storage",
            "regions": ["us-east-1", "sa-east-1", "eu-west-1"],
            "resources": [{"key": "gp3_single_az", "attributes": {"volumeType": "General Purpose-GP3", "databaseEngine": "Any", "deploymentOption": "Single-AZ"}}],
            "purchase_options": ["on_demand"]
        },
        {
            "resource_category": "object_storage",
            "vendor": "aws", "service": "AmazonS3",
            "product_family": "Storage",
            "regions": ["us-east-1", "sa-east-1", "eu-west-1"],
            "resources": [{"key": "standard", "attributes": {"storageClass": "General Purpose"}}],
            "purchase_options": ["on_demand"]
        },
        {
            "resource_category": "backup_storage",
            "vendor": "aws", "service": "AmazonEC2",
            "product_family": "Storage",
            "required": False,
            "regions": ["us-east-1", "sa-east-1", "eu-west-1"],
            "resources": [{"key": "snapshot", "attributes": {}, "attribute_contains_any": [
                {"usagetype": "Snapshot"}, {"volumeType": "Snapshot"}, {"description": "Snapshot"}
            ]}],
            "purchase_options": ["on_demand"]
        },
        {
            "resource_category": "network_egress",
            "vendor": "aws", "service": "AmazonEC2",
            "product_family": "Data Transfer",
            "required": False,
            "regions": ["us-east-1", "sa-east-1", "eu-west-1"],
            "resources": [{"key": "internet_egress", "attributes": {}, "attribute_contains_any": [
                {"usagetype": "DataTransfer"}, {"description": "Data Transfer"}
            ]}],
            "purchase_options": ["on_demand"]
        },
        {
            "resource_category": "database_storage",
            "vendor": "azure", "service": "SQL Database",
            "product_family": "Databases",
            "regions": ["eastus", "brazilsouth", "westeurope"],
            "resources": [{"key": "general_purpose", "attributes": {"meterName": "General Purpose Data Stored"}}],
            "purchase_options": ["Consumption"]
        },
        {
            "resource_category": "object_storage",
            "vendor": "azure", "service": "Storage",
            "product_family": "Storage",
            "regions": ["eastus", "brazilsouth", "westeurope"],
            "resources": [{"key": "hot_lrs", "attributes": {"productName": "Blob Storage", "skuName": "Hot LRS", "meterName": "Hot LRS Data Stored"}}],
            "purchase_options": ["Consumption"]
        },
        {
            "resource_category": "backup_storage",
            "vendor": "azure", "service": "Backup",
            "product_family": "Storage",
            "regions": ["eastus", "brazilsouth", "westeurope"],
            "resources": [{"key": "azure_vm_grs", "attributes": {"skuName": "Azure VM", "meterName": "GRS Data Stored"}}],
            "purchase_options": ["Consumption"]
        },
        {
            "resource_category": "network_egress",
            "vendor": "azure", "service": "Bandwidth",
            "product_family": "Networking",
            "regions": ["eastus", "brazilsouth", "westeurope"],
            "resources": [{"key": "internet_egress", "attributes": {"skuName": "Standard", "meterName": "Standard Data Transfer Out"}}],
            "purchase_options": ["Consumption"]
        },
        {
            "resource_category": "database_storage",
            "vendor": "gcp", "service": "Cloud SQL",
            "product_family": None,
            "regions": ["us-east1", "southamerica-east1", "europe-west1"],
            "resources": [{"key": "general", "attributes": {}, "attribute_contains": {"description": "storage"}}],
            "purchase_options": ["OnDemand"]
        },
        {
            "resource_category": "object_storage",
            "vendor": "gcp", "service": "Cloud Storage",
            "product_family": "Storage",
            "regions": ["us-east1", "southamerica-east1", "europe-west1"],
            "resources": [{"key": "standard", "attributes": {}, "attribute_contains": {"description": "Standard Storage"}}],
            "purchase_options": ["OnDemand"]
        },
        {
            "resource_category": "backup_storage",
            "vendor": "gcp", "service": "Compute Engine",
            "product_family": "Storage",
            "regions": ["us-east1", "southamerica-east1", "europe-west1"],
            "resources": [{"key": "backup", "attributes": {"resourceGroup": "PDSnapshot"}}],
            "purchase_options": ["OnDemand"]
        },
        {
            "resource_category": "network_egress",
            "vendor": "gcp", "service": "Compute Engine",
            "product_family": "Network",
            "regions": ["us-east1", "southamerica-east1", "europe-west1"],
            "resources": [{"key": "internet_egress", "attributes": {}, "attribute_contains": {"description": "Internet Data Transfer Out"}}],
            "purchase_options": ["OnDemand"]
        }
    ]
}


def gql(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def canonical_payment(raw: str | None) -> str:
    value = (raw or "").lower()
    if value in {"consumption", "on_demand", "on-demand", "ondemand"}:
        return "on_demand"
    if any(word in value for word in ("reserved", "reservation", "committed", "savings")):
        return "reserved"
    if any(word in value for word in ("spot", "preemptible")):
        return "spot"
    return value or "unknown"


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


def query_resource(resource: dict[str, Any], region: str,
                   resource_def: dict[str, Any], purchase_option: str,
                   api_key: str) -> list[dict[str, Any]]:
    filters = resource_def.get("attributes", {})
    attr_filters = ", ".join(
        f"{{key: {gql(key)}, value: {gql(value)}}}"
        for key, value in filters.items()
    )
    price_filter = f"filter: {{purchaseOption: {gql(purchase_option)}}}"
    family_filter = "" if not resource.get("product_family") else f"productFamily: {gql(resource['product_family'])}"
    query = f"""query {{ products(filter: {{
        vendorName: {gql(resource['vendor'])}
        service: {gql(resource['service'])}
        {family_filter}
        region: {gql(region)}
        attributeFilters: [{attr_filters}]
    }}) {{
        attributes {{ key value }}
        prices({price_filter}) {{ USD unit description purchaseOption
            termPurchaseOption termLength termOfferingClass
            startUsageAmount endUsageAmount }}
    }} }}"""
    products = request_graphql(query, api_key).get("products", [])
    if not products:
        print(
            f"Aviso: nenhum produto retornado para {resource['vendor']} / "
            f"{resource['resource_category']} / {region} / {resource_def['key']} / "
            f"{purchase_option}. Revise os filtros no arquivo de configuração."
        )
    rows = []
    for product in products:
        attrs = {item.get("key"): item.get("value") for item in product.get("attributes", [])}
        contains_filters = resource_def.get("attribute_contains", {})
        if any(str(value).lower() not in str(attrs.get(key, "")).lower()
               for key, value in contains_filters.items()):
            continue
        any_filters = resource_def.get("attribute_contains_any", [])
        if any_filters and not any(
            str(value).lower() in str(attrs.get(key, "")).lower()
            for candidate in any_filters
            for key, value in candidate.items()
        ):
            continue
        for price in product.get("prices", []):
            if price.get("USD") is None:
                continue
            raw_payment = price.get("purchaseOption") or price.get("termPurchaseOption") or purchase_option
            rows.append({
                "provider": resource["vendor"],
                "resource_category": resource["resource_category"],
                "service": resource["service"],
                "product_family": resource["product_family"],
                "region": region,
                "resource_key": resource_def["key"],
                "instance_type": attrs.get("instanceType") or attrs.get("armSkuName") or attrs.get("machineType"),
                "operating_system": attrs.get("operatingSystem"),
                "payment_model": canonical_payment(raw_payment),
                "payment_model_raw": raw_payment,
                "term_length": price.get("termLength"),
                "currency": "USD", "unit": price.get("unit"),
                "price": float(price["USD"]),
                "price_description": price.get("description"),
                "start_usage_amount": price.get("startUsageAmount"),
                "end_usage_amount": price.get("endUsageAmount"),
                "product_attributes": json.dumps(attrs, ensure_ascii=False, sort_keys=True),
                "source_api": "Infracost Cloud Pricing API",
                "source_url": API_URL,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            })
    return rows


def aws_bulk_url(service_code: str) -> str:
    """Monta a URL pública do arquivo Bulk API de um serviço AWS."""
    return f"{AWS_BULK_BASE}/{service_code}/current/index.json"


def _aws_matches_region(attrs: dict[str, Any], region: str) -> bool:
    location = str(attrs.get("location", ""))
    region_code = str(attrs.get("regionCode", ""))
    from_location = str(attrs.get("fromLocation", ""))
    from_region_code = str(attrs.get("fromRegionCode", ""))
    expected = AWS_REGION_LOCATIONS[region]
    return (
        region_code == region
        or from_region_code == region
        or location.lower() == expected.lower()
        or from_location.lower() == expected.lower()
    )


def collect_aws_bulk_resource(resource: dict[str, Any], region: str,
                              cache: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Coleta as categorias AWS ausentes no Infracost usando o Bulk API oficial.

    O arquivo Bulk API é público e não exige uma segunda chave de API. A função
    filtra por região e por família de produto, preservando a unidade e os
    limites de consumo publicados pela AWS.
    """
    category = resource["resource_category"]
    service_code = {
        "backup_storage": "AWSBackup",
        "network_egress": "AWSDataTransfer",
    }.get(category)
    if not service_code:
        return []

    url = aws_bulk_url(service_code)
    if service_code not in cache:
        response = requests.get(url, timeout=240)
        response.raise_for_status()
        cache[service_code] = response.json()
    offer = cache[service_code]
    rows: list[dict[str, Any]] = []
    expected_location = AWS_REGION_LOCATIONS[region]

    for sku, product in offer.get("products", {}).items():
        attrs = product.get("attributes", {})
        if not _aws_matches_region(attrs, region):
            continue
        family = str(attrs.get("productFamily", ""))
        usage_type = str(attrs.get("usagetype", ""))
        description = str(attrs.get("description", ""))
        searchable = " ".join((family, usage_type, description, str(attrs))).lower()

        if category == "backup_storage":
            if attrs.get("operation") != "Storage" or "bytehrs" not in usage_type.lower():
                continue
        else:
            transfer_type = str(attrs.get("transferType", ""))
            if transfer_type != "AWS Outbound":
                continue

        for term_name, term_group in offer.get("terms", {}).items():
            for term_key, term in term_group.get(sku, {}).items():
                for dimension_key, dimension in term.get("priceDimensions", {}).items():
                    price_per_unit = dimension.get("pricePerUnit", {}).get("USD")
                    if price_per_unit in (None, ""):
                        continue
                    try:
                        price_value = float(price_per_unit)
                    except (TypeError, ValueError):
                        continue
                    rows.append({
                        "provider": "aws",
                        "resource_category": category,
                        "service": "AWS Backup" if category == "backup_storage" else "AWS Data Transfer",
                        "product_family": family or product.get("productFamily"),
                        "region": region,
                        "resource_key": "snapshot" if category == "backup_storage" else "internet_egress",
                        "instance_type": None,
                        "operating_system": None,
                        "payment_model": "on_demand",
                        "payment_model_raw": term_name,
                        "term_length": None,
                        "currency": "USD",
                        "unit": dimension.get("unit"),
                        "price": price_value,
                        "price_description": dimension.get("description") or description,
                        "start_usage_amount": dimension.get("beginRange"),
                        "end_usage_amount": dimension.get("endRange"),
                        "product_attributes": json.dumps({
                            "sku": sku,
                            "offer_term": term_key,
                            "dimension": dimension_key,
                            **attrs,
                        }, ensure_ascii=False, sort_keys=True),
                        "source_api": "AWS Price List Bulk API",
                        "source_url": url,
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    })
    return rows


def collect_aws_complement(frame_rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    """Adiciona preços AWS oficiais somente para categorias não cobertas pelo Infracost."""
    cache: dict[str, dict[str, Any]] = {}
    for resource in config.get("resources", []):
        if resource.get("vendor") != "aws" or resource.get("resource_category") not in {
            "backup_storage", "network_egress"
        }:
            continue
        print(f"Consultando AWS Price List / {resource['resource_category']}...")
        for region in resource.get("regions", []):
            try:
                complement = collect_aws_bulk_resource(resource, region, cache)
                if not complement:
                    print(f"Aviso: AWS Price List não retornou produto para {resource['resource_category']} / {region}.")
                frame_rows.extend(complement)
            except (requests.RequestException, ValueError) as exc:
                print(f"Aviso: falha no complemento AWS / {resource['resource_category']} / {region}: {exc}")


def verify_coverage(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Verifica a matriz provedor x região x recurso x modalidade."""
    checks = []
    excluded = []
    for resource in config.get("resources", []):
        if resource.get("required", True) is False:
            has_complement = not frame.loc[
                (frame["provider"] == resource["vendor"])
                & (frame["resource_category"] == resource["resource_category"])
            ].empty
            if not has_complement:
                excluded.append({
                    "provider": resource["vendor"],
                    "resource_category": resource["resource_category"],
                    "reason": "categoria não retornada pelo Infracost nem pela fonte complementar AWS",
                })
                continue
        for region in resource.get("regions", []):
            for resource_def in resource.get("resources", []):
                for purchase_option in resource.get("purchase_options", []):
                    mask = (
                        (frame["provider"] == resource["vendor"])
                        & (frame["resource_category"] == resource["resource_category"])
                        & (frame["region"] == region)
                        & (frame["resource_key"] == resource_def["key"])
                        & (frame["payment_model"] == canonical_payment(purchase_option))
                    )
                    subset = frame.loc[mask]
                    valid = bool(not subset.empty and subset["price"].notna().any())
                    checks.append({
                        "provider": resource["vendor"],
                        "resource_category": resource["resource_category"],
                        "region": region,
                        "resource_key": resource_def["key"],
                        "requested_payment_model": purchase_option,
                        "normalized_payment_model": canonical_payment(purchase_option),
                        "records": int(len(subset)),
                        "valid": valid,
                    })
    missing = [check for check in checks if not check["valid"]]
    return {
        "complete": not missing,
        "checks": checks,
        "total_checks": len(checks),
        "valid_checks": len(checks) - len(missing),
        "missing_checks": len(missing),
        "missing": missing,
        "excluded": excluded,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config_prices.json"))
    parser.add_argument("--output", type=Path, default=Path("prices_final.parquet"))
    parser.add_argument("--coverage-report", type=Path, default=Path("price_coverage_report.json"))
    parser.add_argument("--allow-incomplete", action="store_true", help="grava e retorna sucesso mesmo com combinações ausentes")
    parser.add_argument("--skip-aws-complement", action="store_true", help="não consulta a AWS Price List Bulk API")
    parser.add_argument("--reset-config", action="store_true", help="substitui a configuração pelo perfil-base desta versão")
    args = parser.parse_args()
    if args.reset_config or not args.config.exists():
        args.config.write_text(json.dumps(DEFAULT_CONFIG, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Configuração criada: {args.config}")
    api_key = os.environ.get("INFRACOST_API_KEY")
    if not api_key:
        raise SystemExit("Defina INFRACOST_API_KEY e execute novamente para consultar a API.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for resource in config["resources"]:
        print(f"Consultando {resource['vendor']} / {resource['resource_category']}...")
        for region in resource["regions"]:
            for resource_def in resource["resources"]:
                for purchase_option in resource["purchase_options"]:
                    try:
                        rows.extend(query_resource(resource, region, resource_def, purchase_option, api_key))
                    except RuntimeError as exc:
                        print(
                            f"Aviso: consulta rejeitada para {resource['vendor']} / "
                            f"{resource['resource_category']} / {region}: {exc}"
                        )
    if not args.skip_aws_complement:
        collect_aws_complement(rows, config)
    frame = pd.DataFrame(rows)
    for column in COLUMNS:
        if column not in frame:
            frame[column] = None
    frame = frame[COLUMNS].drop_duplicates()
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["retrieved_at"] = pd.to_datetime(frame["retrieved_at"], utc=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    coverage = verify_coverage(frame, config)
    coverage["prices_file"] = str(args.output.resolve())
    coverage["executed_at"] = datetime.now(timezone.utc).isoformat()
    args.coverage_report.write_text(json.dumps(coverage, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(frame)} preços gravados em {args.output.resolve()}")
    print(frame.groupby(["provider", "resource_category"]).size().to_string())
    print(f"Cobertura: {coverage['valid_checks']}/{coverage['total_checks']} combinações válidas")
    print(f"Relatório de cobertura: {args.coverage_report.resolve()}")
    if not coverage["complete"] and not args.allow_incomplete:
        raise SystemExit("Coleta incompleta. Corrija os filtros antes de seguir para o pipeline.")


if __name__ == "__main__":
    main()
