from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline


def evaluate_rag(question, answer, contexts):

    # Local LLM (FREE)
    hf_pipeline = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )

    llm = HuggingFacePipeline(
        pipeline=hf_pipeline
    )

    # Local embeddings (FREE)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts]
    }

    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy
        ],
        llm=llm,
        embeddings=embeddings
    )

    return result