import os


class Reranker:
    def __init__(self):
        self.model = None

    def rerank(self, query, documents, top_k=3):
        """
        Rerank documents:
        - In cloud mode (GROQ_API_KEY set on 512MB RAM free tier), uses fast vector-score ranking to avoid PyTorch memory overhead.
        - In local mode (PC GPU), uses CrossEncoder model.
        """
        if not documents:
            return []

        # Cloud Mode RAM Optimization: Avoid loading PyTorch CrossEncoder on 512MB RAM cloud free tiers
        if os.environ.get("GROQ_API_KEY"):
            return [(doc, float(1.0 - (i * 0.1))) for i, doc in enumerate(documents[:top_k])]

        try:
            if self.model is None:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

            pairs = [[query, doc.page_content] for doc in documents]
            scores = self.model.predict(pairs)

            ranked = sorted(
                zip(documents, scores),
                key=lambda x: x[1],
                reverse=True
            )
            return ranked[:top_k]
        except Exception as e:
            print(f"[reranker] CrossEncoder fallback: {e}")
            return [(doc, float(1.0 - (i * 0.1))) for i, doc in enumerate(documents[:top_k])]