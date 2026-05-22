import os
import json
import re
from datetime import datetime

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "failure_logs.json")


def ensure_log_file():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)


def is_valid_query(query):
    """
    Reject garbage / malformed / debug pollution queries.
    """
    if not query:
        return False

    if not isinstance(query, str):
        return False

    query = query.strip()

    if len(query) < 3:
        return False

    # punctuation-only junk
    if re.fullmatch(r"[\W_]+", query):
        return False

    # obvious code/debug fragments
    bad_prefixes = [
        "#",
        "import ",
        "from ",
        "def ",
        "class ",
        "return ",
        "if ",
        "for ",
        "while ",
        "try:",
        "except",
    ]

    query_lower = query.lower()

    for prefix in bad_prefixes:
        if query_lower.startswith(prefix):
            return False

    # too many symbols compared to letters
    letters = len(re.findall(r"[a-zA-Z]", query))
    symbols = len(re.findall(r"[^\w\s]", query))

    if letters == 0:
        return False

    if symbols > letters:
        return False

    return True


def sanitize_scores(scores):
    """
    Ensure scores are valid JSON-safe floats.
    """
    if not scores:
        return []

    clean_scores = []

    for score in scores:
        try:
            clean_scores.append(float(score))
        except Exception:
            continue

    return clean_scores


def log_failure(
    query,
    search_query,
    strategy,
    scores,
    reason
):
    """
    Safe persistent failure logger.
    """
    if not is_valid_query(query):
        return

    ensure_log_file()

    record = {
        "timestamp": datetime.now().isoformat(),
        "query": query.strip(),
        "search_query": str(search_query).strip(),
        "strategy": str(strategy),
        "top_scores": sanitize_scores(scores),
        "reason": str(reason)
    }

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)

        if not isinstance(logs, list):
            logs = []

        logs.append(record)

        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)

    except Exception:
        pass