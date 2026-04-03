from sentence_transformers import CrossEncoder
import numpy as np


class Reranker:

    def __init__(self):

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )


    def rerank(self, query, documents, top_k=3):

        if len(documents) == 0:
            return documents, []

        pairs = [
            (query, doc.page_content)
            for doc in documents
        ]

        scores = self.model.predict(pairs)

        scores = np.array(scores)

        ranked_indices = np.argsort(scores)[::-1]

        top_docs = [
            documents[i]
            for i in ranked_indices[:top_k]
        ]

        top_scores = [
            scores[i]
            for i in ranked_indices[:top_k]
        ]

        return top_docs, top_scores