import os
import json
from collections import Counter

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "failure_logs.json")


def load_logs():
    """
    Load failure logs safely.
    """
    if not os.path.exists(LOG_FILE):
        return []

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)

        if isinstance(logs, list):
            return logs

        return []

    except Exception:
        return []


def summarize_failures():
    """
    Return aggregate failure statistics.
    """
    logs = load_logs()

    if not logs:
        return {
            "total_failures": 0,
            "strategy_counts": {},
            "reason_counts": {},
            "query_counts": {}
        }

    strategy_counter = Counter()
    reason_counter = Counter()
    query_counter = Counter()

    for log in logs:
        strategy = log.get("strategy", "UNKNOWN")
        reason = log.get("reason", "UNKNOWN")
        query = log.get("query", "UNKNOWN")

        strategy_counter[strategy] += 1
        reason_counter[reason] += 1
        query_counter[query] += 1

    return {
        "total_failures": len(logs),
        "strategy_counts": dict(strategy_counter),
        "reason_counts": dict(reason_counter),
        "query_counts": dict(query_counter)
    }


def top_problem_queries(top_k=5):
    """
    Return most frequently failing queries.
    """
    summary = summarize_failures()

    query_counts = summary["query_counts"]

    sorted_queries = sorted(
        query_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_queries[:top_k]


def pretty_failure_report():
    """
    Human-readable report.
    """
    summary = summarize_failures()

    if summary["total_failures"] == 0:
        return "No failures logged yet."

    lines = []

    lines.append("=== FAILURE ANALYTICS REPORT ===")
    lines.append(f"Total Failures: {summary['total_failures']}")
    lines.append("")

    lines.append("Strategy Counts:")
    for strategy, count in summary["strategy_counts"].items():
        lines.append(f"  {strategy}: {count}")

    lines.append("")

    lines.append("Reason Counts:")
    for reason, count in summary["reason_counts"].items():
        lines.append(f"  {reason}: {count}")

    lines.append("")

    lines.append("Top Problem Queries:")
    for query, count in top_problem_queries():
        lines.append(f"  {query} ({count})")

    return "\n".join(lines)