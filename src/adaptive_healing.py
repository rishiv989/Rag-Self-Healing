from src.failure_analytics import summarize_failures

REWRITE_FAILURE_THRESHOLD = 10
MMR_FAILURE_THRESHOLD = 10
REFUSE_THRESHOLD = 20


def get_strategy_failures():
    """
    Return failure counts by strategy.
    """
    summary = summarize_failures()

    return summary.get("strategy_counts", {})


def choose_adaptive_strategy(base_strategy):
    """
    Adapt healing strategy based on historical failures.
    """
    strategy_counts = get_strategy_failures()

    rewrite_failures = strategy_counts.get("REWRITE", 0)
    mmr_failures = strategy_counts.get("MMR", 0)
    refuse_failures = strategy_counts.get("REFUSE", 0)
    max_retry_failures = strategy_counts.get("MAX_RETRIES", 0)

    total_failures = (
        rewrite_failures
        + mmr_failures
        + refuse_failures
        + max_retry_failures
    )

    # No meaningful history yet
    if total_failures < 5:
        return base_strategy

    # If rewrite keeps failing badly, avoid it
    if (
        base_strategy == "REWRITE"
        and rewrite_failures >= REWRITE_FAILURE_THRESHOLD
    ):
        if mmr_failures < MMR_FAILURE_THRESHOLD:
            return "MMR"

        return "REFUSE"

    # If MMR keeps failing badly, avoid it
    if (
        base_strategy == "MMR"
        and mmr_failures >= MMR_FAILURE_THRESHOLD
    ):
        if rewrite_failures < REWRITE_FAILURE_THRESHOLD:
            return "REWRITE"

        return "REFUSE"

    # If system is broadly failing too much, fail fast
    if refuse_failures >= REFUSE_THRESHOLD:
        return "REFUSE"

    return base_strategy


def adaptive_decision(base_strategy):
    """
    Public interface.
    """
    return choose_adaptive_strategy(base_strategy)