import tempfile
import os
import hashlib
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.prompt import PROMPT

load_dotenv()

def get_file_hash(file_path : str) -> str:
    ''' Generating a MD5 hash of file content for indexing '''
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def process_pdf(pdf_path: str) -> FAISS:
    ''' Dynamically handle user uploaded pdf with caching '''
    file_hash = get_file_hash(pdf_path)
    index_dir = "faiss_indexes"
    index_path = os.path.join(index_dir, f"{file_hash}")
    
    os.makedirs(index_dir, exist_ok=True)
    embeddings = OpenAIEmbeddings(openai_api_key = os.environ.get("OPENAI_API_KEY"))
    
    # Returning existing index if found
    if os.path.exists(index_path):
        print(f"Loading cached index file for {pdf_path}")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    # Creating new index file
    print(f"Creating new index file for {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    # print(f"First page text length: {len(documents[0].page_content)} chars")
    # print("Sample text:", documents[0].page_content[:200] + "...")
    
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=[],
        length_function=len,
    )
    
    docs = text_splitter.split_documents(documents=documents)
    
    # for i, doc in enumerate(docs):
    #     print(f"\nChunk {i+1} ({len(doc.page_content)} chars):")
    #     print(doc.page_content[:150] + "...")
    
    # exit()
    
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    vectorstore.save_local(index_path)
    return vectorstore

if __name__ == "__main__":
    st.title("Welcome to CourseBuddy!")
        
    uploaded_file = st.file_uploader("Choose a .pdf file", "pdf")
    question = st.text_input("Enter your query:")


    if st.button("Submit"):
        if uploaded_file is not None and question != "":
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            new_vector_store = process_pdf(pdf_path=tmp_file_path)

            qa = RetrievalQA.from_chain_type(
                llm=OpenAI(model = "gpt-4o-mini", max_tokens=1000),
                chain_type="stuff",
                chain_type_kwargs= {"prompt": PROMPT},
                retriever=new_vector_store.as_retriever(),
                return_source_documents=True,
            )

            result = qa({"query": question})

            response = result["result"]
            sources = result["source_documents"]

            source_text = f"{sources[0].metadata['source']}&nbsp; - &nbsp;Page {sources[0].metadata['page_label']}"
            
            print("\n\n### SOURCE ###\n\n")
            print(sources)
            
            st.write(response)
            with st.expander("Sources"):
                st.badge(source_text)
            
        else:
            st.error("Please upload a document and enter a query!")
