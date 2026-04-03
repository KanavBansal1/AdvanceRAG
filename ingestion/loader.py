import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_documents(data_path):

    if not os.path.exists(data_path):
        return []

    documents = []

    for file in os.listdir(data_path):

        path = os.path.join(data_path, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            documents.extend(loader.load())

        elif file.endswith(".txt"):
            loader = TextLoader(path)
            documents.extend(loader.load())

    return documents