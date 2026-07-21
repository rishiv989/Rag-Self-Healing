import os


class ConfidenceChecker:
    def __init__(self):
        # Cloud mode uses normalized 0.0-1.0 scores; local mode uses CrossEncoder logit scores (-10 to +10)
        is_cloud = bool(os.environ.get("GROQ_API_KEY"))
        self.primary_threshold = 0.5 if is_cloud else 5.0
        self.retry_threshold = 0.3 if is_cloud else 3.5

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