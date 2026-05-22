class HealingPolicy:
    def __init__(self, strong_threshold=5.0):
        self.strong_threshold = strong_threshold

    def decide(
        self,
        ranked_docs,
        original_query,
        resolved_query,
        current_entity
    ):
        if not ranked_docs:
            return "REFUSE"

        best_score = ranked_docs[0][1]

        original_lower = original_query.lower()

        pronouns = [
            " he ",
            " she ",
            " him ",
            " her ",
            " his ",
            " its ",
            " it "
        ]

        has_pronoun = any(
            p in f" {original_lower} "
            for p in pronouns
        )

        # pronoun but no entity context
        if has_pronoun and not current_entity:
            return "CLARIFY"

        # pronoun with known entity
        if has_pronoun and current_entity:
            return "REWRITE"

        if best_score < 0:
            return "REFUSE"

        if best_score < self.strong_threshold:
            return "MMR"

        return "ANSWER"