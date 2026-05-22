import re
import src.state as state

REFERENCE_MAP = {
    "his": True,
    "he": True,
    "him": True,
    "she": True,
    "her": True,
    "it": True,
    "its": True,
}

STOP_WORDS = {
    "what", "who", "when", "where", "why", "how", "which", "whose", "whom",
    "tell", "give", "show", "explain", "describe", "summarize", "about", "means", "person",
    "did", "does", "do", "is", "are", "was", "were", "am", "be", "been", "being",
    "compare", "between",
    "he", "she", "it", "they", "his", "her", "its", "their", "him", "them", "this", "that", "these", "those",
    "father", "mother", "name", "age", "height", "weight", "wife", "husband", "son", "daughter",
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there",
    "document", "topics", "included", "include", "contains", "contain", "text", "context", "information", "info", "details", "detail"
}

BROAD_QUERY_PREFIXES = [
    "tell me",
    "explain",
    "describe",
    "give details",
    "summarize",
    "tell me about"
]

AMBIGUOUS_WORDS = {
    "better",
    "more",
    "less",
    "best",
    "compare",
    "difference",
    "versus",
    "vs"
}


def extract_entities(query):
    # The simplest fix is to title-case the query first before regex if we assume basic names.
    title_query = query.title()
    pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
    matches = re.findall(pattern, title_query)

    valid = []

    for match in matches:
        words = match.lower().split()
        filtered_words = [w for w in words if w not in STOP_WORDS]
        if filtered_words:
            valid.append(" ".join(w.title() for w in filtered_words))

    return valid


def has_pronoun(query):
    tokens = re.findall(r"\b\w+\b", query.lower())

    for token in tokens:
        if token in REFERENCE_MAP:
            return True

    return False


def resolve_query(query, latest_entity):
    if not has_pronoun(query):
        return query

    if not latest_entity:
        return None

    tokens = re.findall(r"\b\w+\b|[^\w\s]", query)
    resolved = []

    for token in tokens:
        lower = token.lower()

        if lower in REFERENCE_MAP:
            if lower in ["his", "her", "its"]:
                resolved.append(f"{latest_entity}'s")
            else:
                resolved.append(latest_entity)
        else:
            resolved.append(token)

    final = " ".join(resolved)
    final = re.sub(r"\s+([?.!,])", r"\1", final)

    return final


def rewrite_query(query, latest_entity):
    words = query.split()
    keywords = []

    for word in words:
        clean = word.lower().strip("?.!,")

        if clean not in STOP_WORDS:
            keywords.append(clean)

    if latest_entity:
        return f"{latest_entity} {' '.join(keywords)} mental toughness resilience"

    return " ".join(keywords)


def is_ambiguous_multi_entity(query, recent_mentions):
    if len(recent_mentions) < 2:
        return False

    if has_pronoun(query):
        return False

    tokens = query.lower().split()

    for token in tokens:
        if token in AMBIGUOUS_WORDS:
            return True

    return False


def is_broad_query(query):
    query_lower = query.lower().strip()

    for prefix in BROAD_QUERY_PREFIXES:
        if query_lower.startswith(prefix):
            return True

    return False
