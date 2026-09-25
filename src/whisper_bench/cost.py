"""Self-computed cost from a documented, versioned rate catalog.

We deliberately do **not** derive cost from ``system.billing.usage``: those rows bill non-production
setup (model download, cluster/endpoint spin-up, warmup) and land ~12h late. Instead we time the
inference work ourselves (see :mod:`whisper_bench.metrics`) and multiply by a documented $/hr from
``conf/compute_costs.yml``.

Each rate is expressed **itemized** where possible — ``dbus_per_hour × usd_per_dbu`` (+ cloud VM
cost for classic compute) — so every dollar in the results table is auditable back to a named SKU
and a DBU rate. ``usd_per_dbu`` is a snapshot of ``system.billing.list_prices`` and can be refreshed
in-workspace via :func:`refresh_usd_per_dbu`. A rate may instead give a flat ``usd_per_hour`` when
that is how the product is priced.

No Spark import at module load (the refresh helper takes a spark session as an argument), so this
is unit-testable off-cluster.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Rate:
    """The $/hr for one named compute, itemized for auditability."""

    key: str
    description: str = ""
    usd_per_hour: Optional[float] = None  # flat all-in rate (used if itemized fields absent)
    dbus_per_hour: Optional[float] = None
    databricks_sku: Optional[str] = None  # SKU name in system.billing.list_prices
    usd_per_dbu: Optional[float] = None  # snapshot of list_prices.pricing.effective_list
    cloud_instance: Optional[str] = None  # e.g. "g5.xlarge" (classic compute only)
    cloud_usd_per_hour: float = 0.0  # cloud VM on-demand list (0 for serverless/serving)

    def effective_usd_per_hour(self) -> float:
        if self.dbus_per_hour is not None and self.usd_per_dbu is not None:
            return self.dbus_per_hour * self.usd_per_dbu + (self.cloud_usd_per_hour or 0.0)
        if self.usd_per_hour is not None:
            return self.usd_per_hour
        raise ValueError(
            f"rate {self.key!r} needs either 'usd_per_hour' or ('dbus_per_hour' and 'usd_per_dbu')"
        )


@dataclass
class RateCatalog:
    version: str
    source: str
    rates: dict[str, Rate]

    @classmethod
    def load(cls, path: str | Path) -> "RateCatalog":
        raw = yaml.safe_load(Path(path).read_text())
        rates = {key: Rate(key=key, **spec) for key, spec in (raw.get("rates") or {}).items()}
        return cls(version=str(raw.get("version", "unset")), source=str(raw.get("source", "")), rates=rates)

    def rate(self, key: str) -> Rate:
        if key not in self.rates:
            raise KeyError(f"rate key {key!r} not in catalog (have: {sorted(self.rates)})")
        return self.rates[key]

    def usd_per_hour(self, key: str) -> float:
        return self.rate(key).effective_usd_per_hour()


def compute_cost(inference_wall_sec: float, usd_per_hour: float, replicas: int = 1) -> float:
    """Cost of running one compute for the measured inference window."""
    return (inference_wall_sec / 3600.0) * usd_per_hour * max(replicas, 1)


def cost_per_audio_hour(total_cost_usd: float, total_audio_sec: float) -> float:
    """The headline price/performance metric: dollars to transcribe one hour of audio."""
    audio_hours = total_audio_sec / 3600.0
    if audio_hours <= 0:
        return float("nan")
    return total_cost_usd / audio_hours


def refresh_usd_per_dbu(spark, catalog: RateCatalog) -> RateCatalog:
    """Update each rate's ``usd_per_dbu`` from ``system.billing.list_prices`` (in-workspace).

    Uses the currently-effective list price (``price_end_time IS NULL``) for each rate's
    ``databricks_sku``. Rates without a SKU (or SKUs not found) are left untouched. Mutates and
    returns ``catalog``.
    """
    skus = sorted({r.databricks_sku for r in catalog.rates.values() if r.databricks_sku})
    if not skus:
        return catalog
    in_list = ", ".join(f"'{s}'" for s in skus)
    rows = spark.sql(
        f"""
        SELECT sku_name, pricing.effective_list AS usd_per_dbu
        FROM system.billing.list_prices
        WHERE sku_name IN ({in_list}) AND price_end_time IS NULL
        """
    ).collect()
    latest = {row["sku_name"]: float(row["usd_per_dbu"]) for row in rows}
    for r in catalog.rates.values():
        if r.databricks_sku in latest:
            r.usd_per_dbu = latest[r.databricks_sku]
    return catalog
