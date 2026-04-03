import streamlit as st
from langchain_groq import ChatGroq


def get_llm():

    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.1-8b-instant",
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )

    return llm