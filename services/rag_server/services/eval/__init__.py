"""Evaluation services module.

Consolidated eval-related services:
- history: Evaluation run storage and retrieval
- baseline: Golden baseline management
- comparison: Run comparison
- recommendation: Configuration recommendations
"""

from services.eval.history import (
    save_evaluation_run,
    load_evaluation_history,
    get_evaluation_run_by_id,
    delete_evaluation_run,
    get_evaluation_summary,
    get_metric_definitions,
    EVAL_RESULTS_DIR,
)

from services.eval.baseline import (
    get_baseline,
    set_baseline,
    clear_baseline,
    check_against_baseline,
)

from services.eval.comparison import compare_runs

from services.eval.recommendation import get_recommendation

__all__ = [
    # History
    "save_evaluation_run",
    "load_evaluation_history",
    "get_evaluation_run_by_id",
    "delete_evaluation_run",
    "get_evaluation_summary",
    "get_metric_definitions",
    "EVAL_RESULTS_DIR",
    # Baseline
    "get_baseline",
    "set_baseline",
    "clear_baseline",
    "check_against_baseline",
    # Comparison
    "compare_runs",
    # Recommendation
    "get_recommendation",
]
