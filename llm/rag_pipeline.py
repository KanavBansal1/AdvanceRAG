from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieval.reranker import Reranker


def create_rag(llm, retriever):

    reranker = Reranker()

    map_prompt = ChatPromptTemplate.from_template(
        """Summarize the following document chunk:

        {context}

        Summary:
        """
    )

    reduce_prompt = ChatPromptTemplate.from_template(
        """Answer using the summaries below:

        Summaries:
        {context}

        Question:
        {question}

        Final Answer:
        """
    )


    def retrieve_and_rerank(query):

        docs = retriever.retrieve(query)

        reranked_docs, scores = reranker.rerank(query, docs)

        return reranked_docs, scores


    def map_step(docs):

        summaries = []

        for doc in docs:

            summary = (
                map_prompt
                | llm
                | StrOutputParser()
            ).invoke({
                "context": doc.page_content
            })

            summaries.append(summary)

        return summaries


    def multi_doc_reasoning(query):

        docs, scores = retrieve_and_rerank(query)

        summaries = map_step(docs)

        combined = "\n\n".join(summaries)

        final_answer = (
            reduce_prompt
            | llm
            | StrOutputParser()
        ).invoke({
            "context": combined,
            "question": query
        })

        sources = [
            doc.metadata
            for doc in docs
        ]

        contexts = [
            doc.page_content
            for doc in docs
        ]

        confidence = max(scores) if len(scores) > 0 else 0

        return {
            "answer": final_answer,
            "sources": sources,
            "contexts": contexts,
            "confidence": float(confidence)
        }


    return multi_doc_reasoning