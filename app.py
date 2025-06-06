import tempfile
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

uploaded_file = st.file_uploader("Choose a .pdf file", "pdf")
question = st.text_input("Enter your query:")

if st.button("Submit"):
    if uploaded_file is not None and question != "":
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(file_path=tmp_file_path)
        data = loader.load()
        text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=30, separator="\n"
        )
        data = text_splitter.split_documents(documents=data)
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
        vector_store = DocArrayInMemorySearch.from_documents(data, embeddings)
        vector_store

        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(model = "gpt-4o-mini", max_tokens=800),
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=True,
        )

        result = qa({"query": question})

        st.write(result["result"])
    else:
        st.error("Please upload a document and enter a query!")
