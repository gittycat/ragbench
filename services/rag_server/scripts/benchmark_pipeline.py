"""
Benchmark the ingestion and query pipeline against a running rag-server instance.

Times:
  (a) ingestion of sample_documents/ end-to-end via the upload API
  (b) 10 sequential + 10 concurrent queries

Usage:
    uv run python scripts/benchmark_pipeline.py [--base-url http://localhost:8001] [--sample-docs ../../sample_documents]

Requires the rag-server stack to be running (docker compose up) and reachable.
"""

import argparse
import asyncio
import statistics
import time
from pathlib import Path

import httpx

DEFAULT_QUESTIONS = [
    "What is machine learning?",
    "What are the main types of machine learning?",
    "How does Docker differ from a virtual machine?",
    "What is a Docker container?",
    "What are Python data types?",
    "How do you define a function in Python?",
    "What is supervised learning?",
    "What is a Docker image?",
    "What are Python list comprehensions?",
    "What is overfitting in machine learning?",
]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    idx = int(round((pct / 100) * (len(values_sorted) - 1)))
    return values_sorted[idx]


def _summarize(label: str, latencies_ms: list[float]) -> dict:
    return {
        "label": label,
        "n": len(latencies_ms),
        "mean_ms": statistics.mean(latencies_ms) if latencies_ms else 0.0,
        "p95_ms": _percentile(latencies_ms, 95),
        "min_ms": min(latencies_ms) if latencies_ms else 0.0,
        "max_ms": max(latencies_ms) if latencies_ms else 0.0,
    }


async def benchmark_ingestion(client: httpx.AsyncClient, sample_docs_dir: Path) -> dict:
    files = sorted(p for p in sample_docs_dir.iterdir() if p.is_file())
    if not files:
        raise SystemExit(f"No sample documents found in {sample_docs_dir}")

    print(f"[INGEST] Uploading {len(files)} document(s) from {sample_docs_dir}...")

    upload_start = time.perf_counter()
    file_handles = [("files", (f.name, f.open("rb"))) for f in files]
    try:
        resp = await client.post("/upload", files=file_handles)
        resp.raise_for_status()
        batch = resp.json()
    finally:
        for _, (_, fh) in file_handles:
            fh.close()

    batch_id = batch["batch_id"]
    print(f"[INGEST] Batch {batch_id} created, polling for completion...")

    max_wait_s = 300
    poll_start = time.perf_counter()
    while True:
        status_resp = await client.get(f"/tasks/{batch_id}/status")
        status_resp.raise_for_status()
        status = status_resp.json()
        if status["completed"] >= status["total"]:
            break
        if time.perf_counter() - poll_start > max_wait_s:
            raise TimeoutError(f"Batch {batch_id} did not complete within {max_wait_s}s: {status}")
        await asyncio.sleep(1.0)

    total_duration_ms = (time.perf_counter() - upload_start) * 1000
    print(f"[INGEST] Batch complete in {total_duration_ms / 1000:.1f}s")

    return {
        "label": "ingestion (end-to-end via upload API)",
        "n": len(files),
        "total_ms": total_duration_ms,
        "mean_ms_per_doc": total_duration_ms / len(files),
    }


async def _run_query(client: httpx.AsyncClient, question: str) -> float:
    start = time.perf_counter()
    resp = await client.post("/query", json={"query": question, "is_temporary": True})
    resp.raise_for_status()
    return (time.perf_counter() - start) * 1000


async def benchmark_queries_sequential(client: httpx.AsyncClient, questions: list[str]) -> dict:
    print(f"[QUERY] Running {len(questions)} sequential queries...")
    latencies = []
    for q in questions:
        latencies.append(await _run_query(client, q))
    return _summarize("queries (sequential)", latencies)


async def benchmark_queries_concurrent(client: httpx.AsyncClient, questions: list[str]) -> dict:
    print(f"[QUERY] Running {len(questions)} concurrent queries...")
    latencies = await asyncio.gather(*(_run_query(client, q) for q in questions))
    return _summarize("queries (concurrent)", list(latencies))


async def check_health_responsive(client: httpx.AsyncClient) -> float:
    start = time.perf_counter()
    resp = await client.get("/health")
    resp.raise_for_status()
    return (time.perf_counter() - start) * 1000


def print_table(rows: list[dict]) -> None:
    print()
    print(f"{'Stage':<35}{'n':>5}{'mean (ms)':>12}{'p95 (ms)':>12}{'min (ms)':>12}{'max (ms)':>12}")
    print("-" * 88)
    for row in rows:
        if "total_ms" in row:
            print(f"{row['label']:<35}{row['n']:>5}{row['mean_ms_per_doc']:>12.1f}{'':>12}{'':>12}{row['total_ms']:>12.1f}")
        else:
            print(f"{row['label']:<35}{row['n']:>5}{row['mean_ms']:>12.1f}{row['p95_ms']:>12.1f}{row['min_ms']:>12.1f}{row['max_ms']:>12.1f}")
    print()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument(
        "--sample-docs",
        default=str(Path(__file__).resolve().parents[3] / "sample_documents"),
    )
    parser.add_argument("--skip-ingestion", action="store_true", help="Skip the ingestion benchmark (useful for re-running query benchmarks only)")
    args = parser.parse_args()

    sample_docs_dir = Path(args.sample_docs)

    async with httpx.AsyncClient(base_url=args.base_url, timeout=300.0) as client:
        health_ms = await check_health_responsive(client)
        print(f"[HEALTH] rag-server responsive ({health_ms:.1f}ms)")

        rows = []

        if not args.skip_ingestion:
            rows.append(await benchmark_ingestion(client, sample_docs_dir))

        rows.append(await benchmark_queries_sequential(client, DEFAULT_QUESTIONS))
        rows.append(await benchmark_queries_concurrent(client, DEFAULT_QUESTIONS))

        # Confirm the event loop stayed responsive during concurrent queries
        health_ms_after = await check_health_responsive(client)
        print(f"[HEALTH] rag-server responsive after concurrent load ({health_ms_after:.1f}ms)")

        print_table(rows)


if __name__ == "__main__":
    asyncio.run(main())
