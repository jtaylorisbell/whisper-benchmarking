"""AI Runtime (serverless GPU) entrypoint — a plain Python script, run by an ``ai_runtime_task``.

The AI Runtime environment runs this script (via command.sh) on a single GPU accelerator, with no
Spark session and no Databricks job parameters. So:
  * config + rate catalog travel INSIDE the code tarball (conf/), selected via CLI args / env vars;
  * we filter to this deployment's GPU so each accelerator only runs its matching configs;
  * results are emitted as JSON to the UC Volume, to be ingested into Delta later (30_report).

All heavy logic lives in the installed ``whisper_bench`` package.
"""

from __future__ import annotations

import argparse
import os

from whisper_bench import ARM_AI_RUNTIME
from whisper_bench.config import load_suite
from whisper_bench.cost import RateCatalog
from whisper_bench.sweep import run_suite


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.environ.get("WHISPER_CONFIG", "conf/smoke.yml"))
    ap.add_argument("--rate-catalog", default=os.environ.get("WHISPER_RATE_CATALOG", "conf/compute_costs.yml"))
    ap.add_argument("--gpu", default=os.environ.get("WHISPER_GPU") or None,
                    help="Restrict to configs for this GPU (e.g. A10, H100). Matches this deployment's accelerator.")
    args = ap.parse_args()

    suite = load_suite(args.config)
    rate_catalog = RateCatalog.load(args.rate_catalog)
    print(f"AI Runtime arm | config={args.config} | gpu={args.gpu or 'ALL'} | model={suite.model}")

    results = run_suite(suite, rate_catalog, arm=ARM_AI_RUNTIME, gpu=args.gpu)
    print(f"Done: emitted {len(results)} result(s) to {suite.volume_root}/results")


if __name__ == "__main__":
    main()
