from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os

def load_documents(data_path):
    
    documents = []

    for file in os.listdir(data_path):
        path = os.path.join(data_path, file)
        if(file.endswith(".pdf")):
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
        elif(file.endswith(".txt")):
            loader = TextLoader(path)
            documents.extend(loader.load())
    return documents