from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

def get_llm():

    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.1-8b-instant"
    )

    return llm