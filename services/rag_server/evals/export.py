"""Manual review export functionality.

Exports evaluation results for human review, particularly useful for:
- Citation verification
- Answer quality spot-checking
- High-stakes domain validation
- Disagreement resolution between judge and humans

Output formats:
- JSON: Full structured data
- CSV: Flattened for spreadsheet review
- Markdown: Human-readable reports
"""

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from evals.schemas import (
    EvalQuestion,
    EvalResponse,
    MetricResult,
    Scorecard,
    EvalRun,
)


def export_for_review(
    questions: list[EvalQuestion],
    responses: list[EvalResponse],
    output_path: Path | str,
    format: str = "json",
    include_context: bool = True,
    include_gold: bool = True,
) -> Path:
    """Export question/response pairs for manual review.

    Args:
        questions: List of evaluation questions
        responses: List of RAG responses
        output_path: Where to save the export
        format: Output format (json, csv, markdown)
        include_context: Include retrieved context in export
        include_gold: Include gold passages in export

    Returns:
        Path to the exported file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        return _export_json(questions, responses, output_path, include_context, include_gold)
    elif format == "csv":
        return _export_csv(questions, responses, output_path, include_context, include_gold)
    elif format == "markdown":
        return _export_markdown(questions, responses, output_path, include_context, include_gold)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _export_json(
    questions: list[EvalQuestion],
    responses: list[EvalResponse],
    output_path: Path,
    include_context: bool,
    include_gold: bool,
) -> Path:
    """Export to JSON format."""
    records = []

    for q, r in zip(questions, responses):
        record = {
            "id": q.id,
            "question": q.question,
            "expected_answer": q.expected_answer,
            "is_unanswerable": q.is_unanswerable,
            "query_type": q.query_type.value,
            "difficulty": q.difficulty.value,
            "domain": q.domain,
            "generated_answer": r.answer,
            "citations": [
                {
                    "source_index": c.source_index,
                    "doc_id": c.doc_id,
                    "chunk_id": c.chunk_id,
                    "text_span": c.text_span,
                }
                for c in r.citations
            ],
        }

        if include_context:
            record["retrieved_chunks"] = [
                {
                    "rank": c.rank,
                    "doc_id": c.doc_id,
                    "chunk_id": c.chunk_id,
                    "text": c.text[:500] + "..." if len(c.text) > 500 else c.text,
                    "score": c.score,
                }
                for c in r.retrieved_chunks
            ]

        if include_gold:
            record["gold_passages"] = [
                {
                    "doc_id": p.doc_id,
                    "chunk_id": p.chunk_id,
                    "text": p.text[:500] + "..." if len(p.text) > 500 else p.text,
                    "relevance_score": p.relevance_score,
                }
                for p in q.gold_passages
            ]

        records.append(record)

    export_data = {
        "exported_at": datetime.now().isoformat(),
        "count": len(records),
        "records": records,
    }

    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)

    return output_path


def _export_csv(
    questions: list[EvalQuestion],
    responses: list[EvalResponse],
    output_path: Path,
    include_context: bool,
    include_gold: bool,
) -> Path:
    """Export to CSV format for spreadsheet review."""
    rows = []

    for q, r in zip(questions, responses):
        # Get cited doc/chunk IDs
        cited_docs = ", ".join(c.doc_id for c in r.citations if c.doc_id)
        cited_chunks = ", ".join(c.chunk_id for c in r.citations if c.chunk_id)

        # Get gold doc/chunk IDs
        gold_docs = ", ".join(p.doc_id for p in q.gold_passages)
        gold_chunks = ", ".join(p.chunk_id for p in q.gold_passages)

        row = {
            "id": q.id,
            "domain": q.domain,
            "query_type": q.query_type.value,
            "difficulty": q.difficulty.value,
            "is_unanswerable": q.is_unanswerable,
            "question": q.question[:200],
            "expected_answer": (q.expected_answer or "")[:300],
            "generated_answer": r.answer[:500],
            "citation_count": len(r.citations),
            "cited_doc_ids": cited_docs,
            "cited_chunk_ids": cited_chunks,
            "retrieved_count": len(r.retrieved_chunks),
            # Review columns (empty for human to fill)
            "answer_correct": "",
            "citations_correct": "",
            "faithfulness": "",
            "notes": "",
        }

        if include_gold:
            row["gold_doc_ids"] = gold_docs
            row["gold_chunk_ids"] = gold_chunks
            row["gold_passage_count"] = len(q.gold_passages)

        rows.append(row)

    if rows:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return output_path


def _export_markdown(
    questions: list[EvalQuestion],
    responses: list[EvalResponse],
    output_path: Path,
    include_context: bool,
    include_gold: bool,
) -> Path:
    """Export to Markdown format for human-readable review."""
    lines = [
        "# RAG Evaluation Manual Review",
        f"\nExported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\nTotal Questions: {len(questions)}",
        "\n---\n",
    ]

    for i, (q, r) in enumerate(zip(questions, responses), 1):
        lines.append(f"## Question {i}: {q.id}\n")
        lines.append(f"**Domain:** {q.domain} | **Type:** {q.query_type.value} | **Difficulty:** {q.difficulty.value}")
        if q.is_unanswerable:
            lines.append("**[UNANSWERABLE]**")
        lines.append("")

        lines.append("### Question")
        lines.append(f"> {q.question}\n")

        lines.append("### Expected Answer")
        if q.expected_answer:
            lines.append(f"> {q.expected_answer}\n")
        else:
            lines.append("> *No expected answer*\n")

        lines.append("### Generated Answer")
        lines.append(f"```\n{r.answer}\n```\n")

        if r.citations:
            lines.append("### Citations")
            for c in r.citations:
                lines.append(f"- **[{c.source_index}]** doc: `{c.doc_id}`, chunk: `{c.chunk_id}`")
                if c.text_span:
                    lines.append(f"  - *\"{c.text_span[:100]}...\"*")
            lines.append("")

        if include_gold and q.gold_passages:
            lines.append("### Gold Passages")
            for p in q.gold_passages:
                lines.append(f"- **doc:** `{p.doc_id}`, **chunk:** `{p.chunk_id}`")
                lines.append(f"  > {p.text[:200]}...")
            lines.append("")

        if include_context and r.retrieved_chunks:
            lines.append("### Retrieved Chunks (Top 3)")
            for c in r.retrieved_chunks[:3]:
                lines.append(f"- **[{c.rank}]** doc: `{c.doc_id}`, chunk: `{c.chunk_id}` (score: {c.score:.3f})" if c.score else f"- **[{c.rank}]** doc: `{c.doc_id}`, chunk: `{c.chunk_id}`")
                lines.append(f"  > {c.text[:150]}...")
            lines.append("")

        # Review section
        lines.append("### Manual Review")
        lines.append("- [ ] Answer correct")
        lines.append("- [ ] Citations accurate")
        lines.append("- [ ] Faithful to context")
        lines.append("- **Notes:**")
        lines.append("")

        lines.append("---\n")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    return output_path


def export_scorecard(
    scorecard: Scorecard,
    output_path: Path | str,
    format: str = "json",
) -> Path:
    """Export a scorecard to a file.

    Args:
        scorecard: The scorecard to export
        output_path: Where to save the export
        format: Output format (json, csv, markdown)

    Returns:
        Path to the exported file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        return _export_scorecard_json(scorecard, output_path)
    elif format == "csv":
        return _export_scorecard_csv(scorecard, output_path)
    elif format == "markdown":
        return _export_scorecard_markdown(scorecard, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _export_scorecard_json(scorecard: Scorecard, output_path: Path) -> Path:
    """Export scorecard to JSON."""
    data = {
        "exported_at": datetime.now().isoformat(),
        "metrics": [
            {
                "name": m.name,
                "value": m.value,
                "group": m.group.value,
                "sample_size": m.sample_size,
                "details": m.details,
            }
            for m in scorecard.metrics
        ],
        "by_group": {
            group.value: [
                {"name": m.name, "value": m.value}
                for m in metrics
            ]
            for group, metrics in scorecard.by_group.items()
        },
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return output_path


def _export_scorecard_csv(scorecard: Scorecard, output_path: Path) -> Path:
    """Export scorecard to CSV."""
    rows = [
        {
            "metric_name": m.name,
            "value": m.value,
            "group": m.group.value,
            "sample_size": m.sample_size,
        }
        for m in scorecard.metrics
    ]

    if rows:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return output_path


def _export_scorecard_markdown(scorecard: Scorecard, output_path: Path) -> Path:
    """Export scorecard to Markdown."""
    lines = [
        "# Evaluation Scorecard",
        f"\nExported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Metrics by Group\n",
    ]

    for group, metrics in scorecard.by_group.items():
        lines.append(f"### {group.value.title()}\n")
        lines.append("| Metric | Value | Sample Size |")
        lines.append("|--------|-------|-------------|")
        for m in metrics:
            lines.append(f"| {m.name} | {m.value:.4f} | {m.sample_size} |")
        lines.append("")

    # Summary statistics
    lines.append("## Summary\n")
    for group in scorecard.by_group:
        avg = scorecard.get_group_average(group)
        if avg is not None:
            lines.append(f"- **{group.value.title()} Average:** {avg:.4f}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    return output_path


def export_run_report(
    run: EvalRun,
    output_path: Path | str,
) -> Path:
    """Export a full evaluation run report in Markdown.

    Args:
        run: The evaluation run to export
        output_path: Where to save the report

    Returns:
        Path to the exported file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Evaluation Report: {run.name}",
        f"\n**Run ID:** {run.id}",
        f"**Created:** {run.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Completed:** {run.completed_at.strftime('%Y-%m-%d %H:%M:%S') if run.completed_at else 'In Progress'}",
        f"**Duration:** {run.duration_seconds:.1f}s" if run.duration_seconds else "",
        "",
        "## Configuration\n",
        f"- **LLM:** {run.config.llm_provider}/{run.config.llm_model}",
        f"- **Embedding:** {run.config.embedding_model}",
        f"- **Reranker:** {run.config.reranker_model or 'Disabled'}",
        f"- **Hybrid Search:** {'Enabled' if run.config.hybrid_search_enabled else 'Disabled'}",
        f"- **Contextual Retrieval:** {'Enabled' if run.config.contextual_retrieval_enabled else 'Disabled'}",
        f"- **Top-K:** {run.config.retrieval_top_k}",
        "",
        "## Datasets\n",
        ", ".join(run.datasets),
        "",
        "## Results Summary\n",
        f"- **Questions Evaluated:** {run.question_count}",
        f"- **Errors:** {run.error_count}",
        f"- **Success Rate:** {run.success_rate:.1%}",
        "",
    ]

    if run.weighted_score:
        lines.extend([
            "## Weighted Score\n",
            f"**Overall Score: {run.weighted_score.score:.3f}**\n",
            "### Objective Breakdown\n",
            "| Objective | Value | Weight | Contribution |",
            "|-----------|-------|--------|--------------|",
        ])
        for obj in sorted(run.weighted_score.objectives.keys()):
            value = run.weighted_score.objectives[obj]
            weight = run.weighted_score.weights.get(obj, 0)
            contrib = run.weighted_score.contributions.get(obj, 0)
            lines.append(f"| {obj} | {value:.3f} | {weight:.2f} | {contrib:.3f} |")
        lines.append("")

    if run.scorecard:
        lines.append("## Detailed Metrics\n")
        for group, metrics in run.scorecard.by_group.items():
            lines.append(f"### {group.value.title()}\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for m in metrics:
                lines.append(f"| {m.name} | {m.value:.4f} |")
            lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    return output_path
