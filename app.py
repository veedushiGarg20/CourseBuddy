import tempfile
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

st.title("Welcome to CourseBuddy!")
    
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
        
        for d in data:
            d.metadata['source'] = uploaded_file.name
        
        
        
        vector_store = FAISS.from_documents(data, embeddings)
        vector_store.save_local("faiss_index_2")
        
        new_vector_store = FAISS.load_local("faiss_index_2", embeddings)

        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(model = "gpt-4o-mini", max_tokens=800),
            chain_type="stuff",
            retriever=new_vector_store.as_retriever(),
            return_source_documents=True,
        )

        result = qa({"query": question})

        response = result["result"]
        sources = result["source_documents"]

        source_text = f"{sources[0].metadata['source']}&nbsp; - &nbsp;Page {sources[0].metadata['page']}"
        
        print("\n\n### SOURCE ###\n\n")
        print(sources)
        
        st.write(response)
        with st.expander("Sources"):
            st.badge(source_text)
        
    else:
        st.error("Please upload a document and enter a query!")
