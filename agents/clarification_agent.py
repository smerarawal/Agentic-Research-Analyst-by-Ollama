def needs_clarification(query):

    vague_words = [
        "analyze",
        "research",
        "study"
    ]

    query_lower = query.lower()

    short_query = len(query.split()) < 4

    if short_query:
        return True

    only_vague = any(
        query_lower.startswith(word)
        for word in vague_words
    ) and len(query.split()) < 6

    return only_vague