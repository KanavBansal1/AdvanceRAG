def get_retriever(vectorstore):

    retriever = vectorstore.as_retriever(
        search_kwargs={"k":5}
    )

    return retriever