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
from langchain.chains.retrieval_qa.prompt import PROMPT_INTERNAL

load_dotenv()

# configurations
UPLOAD_DIR = "pdf_uploads"
INDEX_DIR = "faiss_indexes"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

def hash_password(pswd: str) -> str:
    return hashlib.sha256(pswd.encode()).hexdigest()

USER_DB = {
    os.environ.get("ADMIN_USR"): {
        "password" : os.environ.get("ADMIN_PSWD"),
        "role" : "admin"
    },
    os.environ.get("USER_USR"): {
        "password" : os.environ.get("USER_PSWD"),
        "role" : "user"
    }
}

# ''' Initialise session state '''

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'role' not in st.session_state:
    st.session_state.role = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

def get_file_hash(file_path : str) -> str:
    ''' Generating a MD5 hash of file content for indexing '''
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def process_pdf(pdf_path: str) -> FAISS:
    ''' Dynamically handle user uploaded pdf with caching '''
    file_hash = get_file_hash(pdf_path)
    index_path = os.path.join(INDEX_DIR, f"{file_hash}")
    
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

def handle_pdf_upload(uploaded_file):
    try:
        filename = uploaded_file.name
        dest_path = os.path.join(UPLOAD_DIR, filename)
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
        
        file_hash = get_file_hash(tmp_file_path)
        
        if file_hash in st.session_state.processed_files:
            st.warning("This file has been processed.")
            return None
            
        with st.spinner("Processing PDF..."):
            new_vectorstore = process_pdf(tmp_file_path)
            st.session_state.processed_files.add(file_hash)
            return new_vectorstore
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        
    finally:
        if 'temp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def login():
    st.subheader("Login")
    username = st.text_input("Username: ")
    password = st.text_input("Password: ", type="password")
    
    if st.button("Login"):
        user = USER_DB.get(username)
        if user and user['password'] == hash_password(password):
            st.session_state.authenticated = True
            st.session_state.role = user['role']
            st.session_state.username = username
            st.success(f"Welcome {username}!")
            st.rerun()
        else:
            st.error("Invalid credentials.")
            
def admin_dashboard():
    st.title("CourseBuddy!")
    st.header("Admin Dashboard")
    st.subheader(f"Welcome Admin: {st.session_state.username}")
    
    uploaded_file = st.file_uploader("Upload PDF: ", type="pdf")
    if uploaded_file is not None:
        new_vector_store = handle_pdf_upload(uploaded_file)
        if new_vector_store:
            if st.session_state.vectorstore:
                st.session_state.vectorstore.merge_from(new_vector_store)
                st.success("PDF merged with existing knowledge base.")
            else:
                st.session_state.vectorstore = new_vector_store
                st.success("New knowledge base has been created.")

                    
def user_dashboard():
    st.title("CourseBuddy!")
    st.header("User Dashboard")
    st.subheader(f"Welcome {st.session_state.username}")
    
    uploaded_file = st.file_uploader("Upload PDF: ", type="pdf")
    if uploaded_file is not None:
        new_vector_store = handle_pdf_upload(uploaded_file)
        if new_vector_store:
            if st.session_state.vectorstore:
                st.session_state.vectorstore.merge_from(new_vector_store)
                st.success("PDF merged with existing knowledge base.")
            else:
                st.session_state.vectorstore = new_vector_store
                st.success("New knowledge base has been created.")
    
    question = st.text_input("Enter your query: ")
    
    if st.button("Submit") and question:
        if st.session_state.vectorstore:
            try:
                qa = RetrievalQA.from_chain_type(
                    llm=OpenAI(model = "gpt-4o-mini", max_tokens=1000),
                    chain_type="stuff",
                    chain_type_kwargs= {"prompt": PROMPT_INTERNAL},
                    retriever=st.session_state.vectorstore.as_retriever(),
                    return_source_documents=True,
                )

                result = qa({"query": question})

                response = result["result"]
                sources = result["source_documents"]

                if sources and len(sources) > 0:
                    source_text = f"{sources[0].metadata['source']}&nbsp; - &nbsp;Page {sources[0].metadata['page_label']}"
                else:
                    source_text = "No specific source found"
                
                
                print("\n\n### SOURCE ###\n\n")
                print(sources)
                
                st.write(response)
                with st.expander("Sources"):
                    st.badge(source_text)
                    
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                
        else:
            st.warning("No documents loaded - please upload a PDF first")
            
            
def main():
    if not st.session_state.authenticated:
        login()
    else:
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.role = None
            st.session_state.vectorstore = None 
            st.rerun()
        if st.session_state.role == "admin":
            admin_dashboard()
        else:
            user_dashboard()




if __name__ == "__main__":
    main()
    # st.title("CourseBuddy!")
        
    # uploaded_file = st.file_uploader("Choose a .pdf file", "pdf")
    # question = st.text_input("Enter your query:")


    # if st.button("Submit"):
    #     if uploaded_file is not None and question != "":
    #         with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
    #             tmp_file.write(uploaded_file.getvalue())
    #             tmp_file_path = tmp_file.name

    #         new_vector_store = process_pdf(pdf_path=tmp_file_path)

    #         qa = RetrievalQA.from_chain_type(
    #             llm=OpenAI(model = "gpt-4o-mini", max_tokens=1000),
    #             chain_type="stuff",
    #             chain_type_kwargs= {"prompt": PROMPT},
    #             retriever=new_vector_store.as_retriever(),
    #             return_source_documents=True,
    #         )

    #         result = qa({"query": question})

    #         response = result["result"]
    #         sources = result["source_documents"]

    #         source_text = f"{sources[0].metadata['source']}&nbsp; - &nbsp;Page {sources[0].metadata['page_label']}"
            
    #         print("\n\n### SOURCE ###\n\n")
    #         print(sources)
            
    #         st.write(response)
    #         with st.expander("Sources"):
    #             st.badge(source_text)
            
    #     else:
    #         st.error("Please upload a document and enter a query!")
