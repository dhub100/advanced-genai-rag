"""
Query classifier: labels an incoming query as factoid, semantic, or balanced.

Used by the confidence and adaptive orchestrators to set agent weights.
"""


def classify_query(query: str) -> dict:
    """Classify a query into one of three retrieval-strategy categories.

    Used by :class:`ConfidenceOrchestrator` and
    :class:`AdaptiveOrchestrator` to select appropriate RRF weight presets.

    Args:
        query: Natural-language question string.

    Returns:
        Dict with keys:

        - ``label`` (str): ``"factoid"``, ``"semantic"``, or ``"balanced"``.
        - ``length`` (int): Token count.
        - ``has_digits`` (bool): True if any digit character is present.
        - ``has_quotes`` (bool): True if the query contains quote characters.
        - ``has_names`` (bool): True if any token starts with an uppercase letter.
        - ``q_words`` (int): Count of interrogative words (who/what/when/…).
        - ``is_factoid`` (bool): Raw factoid signal.
        - ``is_semantic`` (bool): Raw semantic signal.
    """
    tokens    = query.split()
    q_lower   = query.lower()

    has_digits = any(ch.isdigit() for ch in query)
    has_quotes = ('"' in query) or ("'" in query)
    has_names  = any(t[0].isupper() for t in tokens if len(t) > 1)
    q_words    = sum(1 for w in ("who", "what", "when", "where", "why", "how") if w in q_lower)

    is_factoid  = has_digits or has_quotes or len(tokens) <= 6
    is_semantic = len(tokens) > 10 and not has_digits

    if is_factoid:
        label = "factoid"
    elif is_semantic:
        label = "semantic"
    else:
        label = "balanced"

    return {
        "label":       label,
        "length":      len(tokens),
        "has_digits":  has_digits,
        "has_quotes":  has_quotes,
        "has_names":   has_names,
        "q_words":     q_words,
        "is_factoid":  is_factoid,
        "is_semantic": is_semantic,
    }
