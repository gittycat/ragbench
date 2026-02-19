"""Command-line interface for evals.

Usage:
    # Run evaluation with defaults
    python -m evals.cli eval

    # Run with specific datasets
    python -m evals.cli eval --datasets ragbench,squad_v2

    # Run with limited samples
    python -m evals.cli eval --samples 10

    # Show dataset stats
    python -m evals.cli stats

    # List available datasets
    python -m evals.cli datasets

    # Export results for manual review
    python -m evals.cli export --run-id abc123

    # Compare multiple runs
    python -m evals.cli compare run1 run2 run3
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from evals.config import EvalConfig, DatasetName, EvalTier, DATASET_TIER_SUPPORT, JudgeConfig, MetricConfig
from evals.datasets.registry import list_datasets, get_dataset
from evals.runner import EvaluationRunner, run_evaluation, compute_pareto_frontier
from evals.schemas import EvalRun
from infrastructure.config.display import print_config_banner
from infrastructure.settings import init_settings


def main():
    """Main CLI entry point."""
    init_settings()
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # eval command
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument(
        "--datasets",
        type=str,
        help="Comma-separated list of datasets (ragbench,squad_v2,qasper,hotpotqa,msmarco)",
        default="ragbench",
    )
    eval_parser.add_argument(
        "--samples",
        type=int,
        help="Number of samples per dataset (default: 100)",
        default=100,
    )
    eval_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
        default=42,
    )
    eval_parser.add_argument(
        "--name",
        type=str,
        help="Name for this evaluation run",
    )
    eval_parser.add_argument(
        "--rag-url",
        type=str,
        help="RAG server URL (default: RAG_SERVER_URL env or http://localhost:8001)",
        default=os.environ.get("RAG_SERVER_URL", "http://localhost:8001"),
    )
    eval_parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Disable LLM-as-judge metrics (faster but less comprehensive)",
    )
    eval_parser.add_argument(
        "--output",
        type=str,
        help="Output directory for results",
        default="data/eval_runs",
    )
    eval_parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file",
    )
    eval_parser.add_argument(
        "--tier",
        type=str,
        choices=["generation", "end_to_end"],
        default="end_to_end",
        help="Evaluation tier: generation (inject context, no ingestion) or end_to_end (full pipeline)",
    )

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")
    stats_parser.add_argument(
        "--dataset",
        type=str,
        help="Specific dataset to show stats for",
    )

    # datasets command
    subparsers.add_parser("datasets", help="List available datasets")

    # export command
    export_parser = subparsers.add_parser("export", help="Export results for manual review")
    export_parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID to export",
    )
    export_parser.add_argument(
        "--format",
        type=str,
        choices=["json", "csv"],
        default="json",
        help="Output format",
    )
    export_parser.add_argument(
        "--output",
        type=str,
        help="Output file path",
    )

    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple evaluation runs")
    compare_parser.add_argument(
        "run_ids",
        nargs="+",
        help="Run IDs to compare",
    )
    compare_parser.add_argument(
        "--pareto",
        action="store_true",
        help="Show Pareto frontier analysis",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "eval":
        cmd_eval(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "datasets":
        cmd_datasets(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()
        sys.exit(1)


def cmd_eval(args):
    """Run evaluation."""
    # Print config banner
    print_config_banner(compact=True)
    print()

    # Load config from file or build from args
    tier = EvalTier(args.tier)

    if args.config:
        config = EvalConfig.from_yaml(args.config)
        # CLI --tier overrides YAML tier
        config.tier = tier
        print(f"Loaded config from: {args.config}")
    else:
        # Parse datasets
        dataset_names = [
            DatasetName(ds.strip())
            for ds in args.datasets.split(",")
        ]

        # Validate dataset+tier combinations before building config
        incompatible = []
        for ds in dataset_names:
            supported = DATASET_TIER_SUPPORT.get(ds, list(EvalTier))
            if tier not in supported:
                incompatible.append((ds.value, [t.value for t in supported]))

        if incompatible:
            print(f"\nERROR: Incompatible dataset/tier combinations for tier '{tier.value}':")
            for ds_name, supported_tiers in incompatible:
                print(f"  - {ds_name}: supports {supported_tiers}")
            sys.exit(1)

        # Build config
        try:
            config = EvalConfig(
                datasets=dataset_names,
                samples_per_dataset=args.samples,
                seed=args.seed,
                rag_server_url=args.rag_url,
                runs_dir=Path(args.output),
                judge=JudgeConfig(enabled=not args.no_judge),
                tier=tier,
            )
        except ValueError as e:
            print(f"\nERROR: {e}")
            sys.exit(1)

    # For END_TO_END, verify the RAG server's upload endpoint is reachable
    if config.tier == EvalTier.END_TO_END:
        import httpx
        try:
            resp = httpx.get(f"{config.rag_server_url}/health", timeout=5.0)
            if resp.status_code != 200:
                print(f"\nERROR: RAG server at {config.rag_server_url} returned status {resp.status_code}")
                print("For END_TO_END tier, the full RAG stack (rag-server + task-worker) must be running.")
                sys.exit(1)
        except Exception as e:
            print(f"\nERROR: Cannot reach RAG server at {config.rag_server_url}: {e}")
            print("For END_TO_END tier, the full RAG stack (rag-server + task-worker) must be running.")
            sys.exit(1)

    print(f"Tier: {config.tier.value}")
    print(f"Datasets: {[ds.value for ds in config.datasets]}")
    print(f"Samples per dataset: {config.samples_per_dataset}")
    print(f"RAG server: {config.rag_server_url}")
    print(f"Judge enabled: {config.judge.enabled}")
    print("-" * 60)

    try:
        result = run_evaluation(config)
        print_run_summary(result)
    except ConnectionError as e:
        print(f"\nERROR: {e}")
        print("Make sure the RAG server is running.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_stats(args):
    """Show dataset statistics."""
    # Print config banner
    print_config_banner(compact=True)
    print()

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = [ds["name"] for ds in list_datasets()]

    print("=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    for ds_name in datasets:
        try:
            # Load full dataset to get stats
            dataset = get_dataset(ds_name, max_samples=None)
            print(f"\n{ds_name.upper()}")
            print("-" * 40)
            print(f"  Total questions: {len(dataset)}")
            print(f"  Version: {dataset.version}")
            print(f"  Description: {dataset.description[:80]}...")

            # Count by query type
            query_types = {}
            domains = {}
            for q in dataset.questions:
                qt = q.query_type.value
                query_types[qt] = query_types.get(qt, 0) + 1
                domains[q.domain] = domains.get(q.domain, 0) + 1

            if query_types:
                print(f"  Query types:")
                for qt, count in sorted(query_types.items()):
                    print(f"    - {qt}: {count}")

            if len(domains) <= 10:
                print(f"  Domains:")
                for domain, count in sorted(domains.items()):
                    print(f"    - {domain}: {count}")
            else:
                print(f"  Domains: {len(domains)} unique")

        except Exception as e:
            print(f"\n{ds_name}: ERROR - {e}")


def cmd_datasets(args):
    """List available datasets."""
    # Print config banner
    print_config_banner(compact=True)
    print()

    print("=" * 60)
    print("Available Datasets")
    print("=" * 60)

    for ds in list_datasets():
        print(f"\n{ds['name']}")
        print(f"  {ds.get('description', 'No description')[:70]}...")
        print(f"  URL: {ds.get('source_url', 'N/A')}")


def cmd_export(args):
    """Export results for manual review."""
    runs_dir = Path("data/eval_runs")

    # Find the run file
    run_file = None
    for f in runs_dir.glob(f"{args.run_id}*.json"):
        run_file = f
        break

    if not run_file:
        print(f"ERROR: Run {args.run_id} not found in {runs_dir}")
        sys.exit(1)

    with open(run_file) as f:
        run_data = json.load(f)

    if args.format == "json":
        output_path = args.output or f"export_{args.run_id}.json"
        with open(output_path, "w") as f:
            json.dump(run_data, f, indent=2)
        print(f"Exported to: {output_path}")

    elif args.format == "csv":
        import csv
        output_path = args.output or f"export_{args.run_id}.csv"

        # Flatten metrics for CSV
        rows = []
        if run_data.get("scorecard"):
            for metric in run_data["scorecard"]["metrics"]:
                rows.append({
                    "run_id": run_data["id"],
                    "run_name": run_data["name"],
                    "metric_name": metric["name"],
                    "metric_group": metric["group"],
                    "value": metric["value"],
                    "sample_size": metric.get("sample_size", ""),
                })

        with open(output_path, "w", newline="") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        print(f"Exported to: {output_path}")


def cmd_compare(args):
    """Compare multiple evaluation runs."""
    runs_dir = Path("data/eval_runs")
    runs = []

    for run_id in args.run_ids:
        for f in runs_dir.glob(f"{run_id}*.json"):
            with open(f) as fh:
                run_data = json.load(fh)
                runs.append(run_data)
            break
        else:
            print(f"WARNING: Run {run_id} not found")

    if not runs:
        print("ERROR: No runs found")
        sys.exit(1)

    print("=" * 80)
    print("Run Comparison")
    print("=" * 80)

    # Header
    header = ["Metric"] + [r["name"][:15] for r in runs]
    col_width = 15
    print(" | ".join(h.ljust(col_width) for h in header))
    print("-" * (col_width * len(header) + 3 * (len(header) - 1)))

    # Collect all metric names
    all_metrics = set()
    for run in runs:
        if run.get("scorecard"):
            for m in run["scorecard"]["metrics"]:
                all_metrics.add(m["name"])

    # Print each metric
    for metric_name in sorted(all_metrics):
        row = [metric_name[:col_width]]
        for run in runs:
            value = "-"
            if run.get("scorecard"):
                for m in run["scorecard"]["metrics"]:
                    if m["name"] == metric_name:
                        value = f"{m['value']:.3f}"
                        break
            row.append(value)
        print(" | ".join(str(v).ljust(col_width) for v in row))

    # Weighted scores
    print("-" * (col_width * len(header) + 3 * (len(header) - 1)))
    row = ["WEIGHTED SCORE"]
    for run in runs:
        ws = run.get("weighted_score", {})
        score = ws.get("score", 0)
        row.append(f"{score:.3f}")
    print(" | ".join(str(v).ljust(col_width) for v in row))

    # Pareto analysis
    if args.pareto and len(runs) > 1:
        print("\n" + "=" * 80)
        print("Pareto Analysis")
        print("=" * 80)
        pareto_points = _compute_pareto_from_dicts(runs)
        _print_pareto_analysis(pareto_points)


def _compute_pareto_from_dicts(runs: list[dict]) -> list[dict]:
    """Compute Pareto frontier from run dictionaries.

    A run is Pareto-optimal if no other run dominates it
    (better in at least one objective without being worse in any).
    """
    points = []

    for run in runs:
        ws = run.get("weighted_score", {})
        objectives = ws.get("objectives", {})
        if not objectives:
            continue

        point = {
            "run_id": run["id"],
            "config_name": run["name"],
            "objectives": objectives.copy(),
            "is_dominated": False,
            "dominates": [],
        }
        points.append(point)

    # Determine dominance
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i == j:
                continue

            # Check if p2 dominates p1
            better_in_one = False
            worse_in_one = False

            for obj in p1["objectives"]:
                v1 = p1["objectives"].get(obj, 0)
                v2 = p2["objectives"].get(obj, 0)

                if v2 > v1:
                    better_in_one = True
                elif v2 < v1:
                    worse_in_one = True

            if better_in_one and not worse_in_one:
                p1["is_dominated"] = True
                p2["dominates"].append(p1["run_id"])

    return points


def _print_pareto_analysis(points: list[dict]) -> None:
    """Print Pareto frontier analysis results."""
    if not points:
        print("No runs with objective data found.")
        return

    # Separate frontier from dominated points
    frontier = [p for p in points if not p["is_dominated"]]
    dominated = [p for p in points if p["is_dominated"]]

    print(f"\nPareto Frontier ({len(frontier)} runs):")
    print("-" * 40)

    for p in frontier:
        print(f"\n  {p['config_name']} [{p['run_id']}]")
        for obj, val in sorted(p["objectives"].items()):
            print(f"    {obj}: {val:.3f}")
        if p["dominates"]:
            print(f"    Dominates: {', '.join(p['dominates'])}")

    if dominated:
        print(f"\nDominated Runs ({len(dominated)}):")
        print("-" * 40)
        for p in dominated:
            print(f"  {p['config_name']} [{p['run_id']}] - dominated")

    # Recommendations
    if frontier:
        print("\nRecommendations:")
        print("-" * 40)

        # Find best for each objective
        all_objectives = set()
        for p in points:
            all_objectives.update(p["objectives"].keys())

        for obj in sorted(all_objectives):
            best_point = max(
                [p for p in points if obj in p["objectives"]],
                key=lambda p: p["objectives"].get(obj, 0),
                default=None,
            )
            if best_point:
                print(f"  Best {obj}: {best_point['config_name']} ({best_point['objectives'][obj]:.3f})")


def print_run_summary(run: EvalRun):
    """Print a summary of an evaluation run."""
    print("\n" + "=" * 60)
    print(f"Evaluation Complete: {run.name}")
    print("=" * 60)
    print(f"Run ID: {run.id}")
    print(f"Duration: {run.duration_seconds:.1f}s" if run.duration_seconds else "")
    print(f"Questions: {run.question_count} ({run.error_count} errors)")
    print(f"Success rate: {run.success_rate:.1%}")

    if run.scorecard:
        print("\n" + "-" * 40)
        print("Metrics by Group:")
        for group, metrics in run.scorecard.by_group.items():
            print(f"\n  {group.value.upper()}:")
            for metric in metrics:
                print(f"    {metric.name}: {metric.value:.3f}")

    if run.weighted_score:
        print("\n" + "-" * 40)
        print(f"WEIGHTED SCORE: {run.weighted_score.score:.3f}")
        print("\nObjective contributions:")
        for obj, contrib in sorted(
            run.weighted_score.contributions.items(),
            key=lambda x: -x[1]
        ):
            weight = run.weighted_score.weights.get(obj, 0)
            value = run.weighted_score.objectives.get(obj, 0)
            print(f"  {obj}: {value:.3f} * {weight:.2f} = {contrib:.3f}")


if __name__ == "__main__":
    main()
