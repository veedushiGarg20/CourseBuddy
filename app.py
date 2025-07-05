import os
import hashlib
import requests
import tempfile
from typing import List, Optional
from urllib.parse import quote, urlparse
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from dotenv import load_dotenv
from langchain.chains import RetrievalQA, LLMChain
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains.retrieval_qa.prompt import PROMPT_INTERNAL, PROMPT_NO_SEARCH, PROMPT_WEB_QUERY

st.set_page_config(layout="wide")

load_dotenv()

# Configurations
UPLOAD_DIR = "pdf_uploads"
INDEX_DIR = "faiss_indexes"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Authentication
def hash_password(pswd: str) -> str:
    return hashlib.sha256(pswd.encode()).hexdigest()

USER_DB = {
    os.environ.get("ADMIN_USR"): {
        "password": os.environ.get("ADMIN_PSWD"),
        "role": "admin"
    },
    os.environ.get("USER_USR"): {
        "password": os.environ.get("USER_PSWD"),
        "role": "user"
    }
}

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'role' not in st.session_state:
    st.session_state.role = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()
if 'mode' not in st.session_state:
    st.session_state.mode = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

class WebContentProcessor:
    @staticmethod
    def download_pdf(url: str, temp_dir: str) -> Optional[str]:
        ''' Download pdf from url to temporary dictionary '''
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            
            if response.headers.get('content-type') == 'application/pdf':
                # Extract filename from PDF or generate one
                parsed_url = urlparse(url)
                filename = os.path.basename(parsed_url.path) or "downloaded_file.pdf"
                safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_')).rstrip()
                filepath = os.path.join(temp_dir, safe_filename)
                
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return filepath
        except Exception as e:
            st.error(f"Failed to download PDF from {url}: {str(e)}")
            return None
    
    @staticmethod
    def load_pdf_content(pdf_path: str) -> List[Document]:
        ''' Load and process PDF content '''
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            for doc in documents:
                doc.metadata["source_type"] = "pdf"
                doc.metadata["original_url"] = pdf_path
                print(pdf_path)
            return documents
        except Exception as e:
            st.error(f"Failed to process PDF {pdf_path}: {str(e)}")
            return []
        
    @staticmethod
    def load_web_content(url: str) -> List[Document]:
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            for doc in documents:
                doc.metadata["source_type"] = "web"
                doc.metadata["original_url"] = url
            return documents
        except Exception as e:
            st.error(f"Failed to process web-page {url}: {str(e)}")
            return []
        
    @staticmethod
    def process_url(url: str, temp_dir: str) -> List[Document]:
        if url.lower().endswith("pdf"):
            pdf_path = WebContentProcessor.download_pdf(url, temp_dir)
            if pdf_path:
                return WebContentProcessor.load_pdf_content(pdf_path)
        else:
            return WebContentProcessor.load_web_content(url)
        return []
    
    @staticmethod
    def google_search(query: str, num: int = 2):
        api_key = os.environ.get("GOOGLE_SEARCH_KEY")
        cse_id = os.environ.get("CSE_ID")
        
        if not api_key or not cse_id:
            raise ValueError("Missing API credentials")
        
        params = {
            'q': query,
            'key': api_key,
            'cx': cse_id,
            'num': num,
            'excludeTerms': 'youtube, video, watch',
        }
        
        url = f"https://www.googleapis.com/customsearch/v1"
        response = requests.get(url, params)
        if response.status_code == 200:
            return response.json()
        raise Exception(f"API Error {response.status_code} : {response.text}")
    
    @staticmethod
    def filter_search_results(results: dict) -> list:
        filtered_items = []
        for item in results.get('items', []):
            url = item.get('link', '')
            if any(skip in url.lower() for skip in ['youtube.com', 'video', 'watch']):
                continue
            filtered_items.append(item)
        return filtered_items
    
    @staticmethod
    def process_search_results(queries: list):
        with tempfile.TemporaryDirectory() as temp_dir:
            all_docs = []
            for query in queries:
                try:
                    results = WebContentProcessor.google_search(query)
                    filtered_results = WebContentProcessor.filter_search_results(results)
                    for item in filtered_results[:3]:
                        url = item.get('link')
                        if url:
                            docs = WebContentProcessor.process_url(url, temp_dir)
                            if docs:
                                docs[0].metadata.update({
                                    'source_url': url,
                                    'title': item.get('title', 'No title')
                                })
                                all_docs.extend(docs)
                except Exception as e:
                    st.error(f"Error processing query '{query}': {str(e)}")
            return all_docs

class LLMOperations:
    @staticmethod
    def no_context_llm(question: str, prompt_template) -> str:
        qa = LLMChain(
            llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=1000),
            prompt=prompt_template,
            verbose=True
        )
        return qa.run(question)
    
    @staticmethod
    def retrieval_qa(question: str, vectorstore: FAISS, prompt_template):
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4o-mini", max_tokens=1000), 
            chain_type="stuff", 
            chain_type_kwargs={"prompt": prompt_template},
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,
        )
        result = qa({"query": question})
        return result["result"], result["source_documents"]

def get_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def process_pdf(pdf_path: str) -> FAISS:
    file_hash = get_file_hash(pdf_path)
    index_path = os.path.join(INDEX_DIR, f"{file_hash}")
    
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=[],
        length_function=len,
    )
    
    docs = text_splitter.split_documents(documents=documents)
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
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
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

def no_search():
    st.subheader("General Knowledge Search")
    question = st.text_input("Enter your question:")
    
    if st.button("Submit") and question:
        with st.spinner("Generating response..."):
            response = LLMOperations.no_context_llm(question, PROMPT_NO_SEARCH)
            st.write(response)

def internal_search():
    st.subheader("Internal Knowledge Search")
    col1, col2 = st.columns([2, 1])
    pdf_path = None
    with col1:
        uploaded_file = st.file_uploader("Upload PDF: ", type="pdf", key="internal_pdf")
        if uploaded_file is not None:
            new_vector_store = handle_pdf_upload(uploaded_file)
            if new_vector_store:
                if st.session_state.vectorstore:
                    st.session_state.vectorstore.merge_from(new_vector_store)
                    st.success("PDF merged with existing knowledge base.")
                else:
                    st.session_state.vectorstore = new_vector_store
                    st.success("New knowledge base has been created.")
        
        question = st.text_input("Enter your query:")
    
        if st.button("Submit") and question:
            if st.session_state.vectorstore:
                with st.spinner("Searching knowledge base..."):
                    response, sources = LLMOperations.retrieval_qa(
                        question, 
                        st.session_state.vectorstore, 
                        PROMPT_INTERNAL
                    )
                    
                    st.write(response)
                    
                    if sources and len(sources) > 0:
                        source_text = f"{sources[0].metadata['source']} - Page {sources[0].metadata['page']+1}"
                        with st.expander("Sources"):
                            st.write(source_text)
                        
                        # pdf_path = None
                        for filename in os.listdir(UPLOAD_DIR):
                            if filename in sources[0].metadata['source']:
                                pdf_path = os.path.join(UPLOAD_DIR, filename)
                                break
                        
        if pdf_path:
            with st.container():
                with col2:
                    st.subheader("PDF Viewer")
                    pdf_viewer(
                        pdf_path,
                        height=800,
                        width=700,
                    )
        else:
            st.warning("No documents loaded - please upload a PDF first")

def external_search():
    st.subheader("Web Search")
    question = st.text_input("Enter your question:")
    
    if st.button("Search") and question:
        with st.spinner("Generating search queries..."):
            search_queries = LLMOperations.no_context_llm(question, PROMPT_WEB_QUERY)
            queries = [q.strip().strip('"') for q in search_queries.split("\n") if q.strip()]
            
            st.session_state.search_results = WebContentProcessor.process_search_results(queries)
            
            if st.session_state.search_results:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=[],
                    length_function=len,
                )
                docs = text_splitter.split_documents(st.session_state.search_results)
                vectorstore = FAISS.from_documents(docs, embeddings)
                
                response, sources = LLMOperations.retrieval_qa(
                    question, 
                    vectorstore, 
                    PROMPT_INTERNAL
                )
                
                st.write(response)
                
                with st.expander("Sources"):
                    for i, source in enumerate(sources, 1):
                        st.write(f"{i}. {source.metadata.get('source_url', 'Unknown source')}")
            else:
                st.warning("No relevant content found from web search.")

def user_dashboard():
    st.title("CourseBuddy!")
    st.header("User Dashboard")
    st.subheader(f"Welcome {st.session_state.username}!")
    
    st.write("""
        Choose how you'd like to study:
        1. **General Knowledge** - Ask questions without context
        2. **Internal Search** - Use uploaded PDFs for querying
        3. **Web Search** - Search the web for answers
    """)
    
    if st.button("General Knowledge"):
        st.session_state.mode = "no_search"
        st.rerun()
        
    if st.button("Internal Search"):
        st.session_state.mode = "internal"
        st.rerun()
        
    if st.button("Web Search"):
        st.session_state.mode = "external"
        st.rerun()
    
    if st.session_state.mode == "no_search":
        no_search()
    elif st.session_state.mode == "internal":
        internal_search()
    elif st.session_state.mode == "external":
        external_search()

def main():
    if not st.session_state.authenticated:
        login()
    else:
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.role = None
            st.session_state.vectorstore = None
            st.session_state.mode = None
            st.rerun()
            
        if st.session_state.role == "admin":
            admin_dashboard()
        else:
            user_dashboard()

if __name__ == "__main__":
    main()