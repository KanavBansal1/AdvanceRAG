from rank_bm25 import BM25Okapi
import numpy as np


class HybridRetriever:

    def __init__(self, vector_retriever, documents):

        self.vector_retriever = vector_retriever
        self.documents = documents

        self.texts = [doc.page_content for doc in documents]
        self.tokenized = [text.split() for text in self.texts]

        self.bm25 = BM25Okapi(self.tokenized)


    def retrieve(self, query, k=5):

        # Semantic retrieval
        semantic_docs = self.vector_retriever.invoke(query)

        # Keyword retrieval
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        top_k = np.argsort(bm25_scores)[-k:]
        keyword_docs = [self.documents[i] for i in top_k]

        # Combine results
        combined = semantic_docs + keyword_docs

        return combined