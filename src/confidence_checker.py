class ConfidenceChecker:
    def __init__(self):
        self.primary_threshold = 5.0
        self.retry_threshold = 3.5

    def is_confident(self, ranked_docs, retry=False):
        if not ranked_docs:
            return False

        best_score = ranked_docs[0][1]

        threshold = (
            self.retry_threshold
            if retry
            else self.primary_threshold
        )

        return best_score >= threshold