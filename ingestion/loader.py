import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_documents(data_path):

    os.makedirs(data_path, exist_ok=True)

    documents = []

    files = os.listdir(data_path)

    # If no files, return empty list
    if len(files) == 0:
        return documents

    for file in files:

        path = os.path.join(data_path, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            documents.extend(loader.load())

        elif file.endswith(".txt"):
            loader = TextLoader(path)
            documents.extend(loader.load())

    return documents