class Reranker:
    def __init__(self):
        self.model = None

    def rerank(self, query, documents, top_k=3):
        if not documents:
            return []
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
            # Safe fallback if RAM constraint prevents loading PyTorch model
            return [(doc, float(1.0 - (i * 0.1))) for i, doc in enumerate(documents[:top_k])]