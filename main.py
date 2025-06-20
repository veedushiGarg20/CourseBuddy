import os
import hashlib
import requests
import tempfile
from typing import List, Optional
from urllib.parse import quote, urlparse
from dotenv import load_dotenv
from langchain.chains import RetrievalQA, LLMChain
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains.retrieval_qa.prompt import PROMPT_INTERNAL, PROMPT_NO_SEARCH, PROMPT_WEB_QUERY

load_dotenv()

# configurations
UPLOAD_DIR = "pdf_uploads"
INDEX_DIR = "faiss_indexes"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)


# Authentication
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

# Global state
active_vectorstore = None
embeddings = OpenAIEmbeddings(openai_api_key = os.environ.get("OPENAI_API_KEY"))

def get_file_hash(file_path : str) -> str:
    ''' Generating a MD5 hash of file content for indexing '''
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
    
    
def process_pdf(pdf_path: str) -> FAISS:
    ''' Dynamically handle user uploaded pdf with caching '''
    file_hash = get_file_hash(pdf_path)
    index_path = os.path.join(INDEX_DIR, f"{file_hash}")
    
    # Returning existing index if found
    if os.path.exists(index_path):
        print(f"Loading cached index file for {pdf_path}")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    # Creating new index file
    print(f"Creating new index file for {pdf_path}")
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


def add_pdf(pdf_path : str):
    ''' Add a pdf to knowledge base '''
    global active_vectorstore
    
    if not pdf_path.lower().endswith('.pdf'):
        raise ValueError("Only PDF files are supported.")
    
    if not os.path.exists(pdf_path):
        raise ValueError("PDF file not found.")
    
    new_store = process_pdf(pdf_path=pdf_path)
    
    if active_vectorstore:
        active_vectorstore.merge(new_store)
        print(f"Merged {os.path.basename(pdf_path)} with existing knowledge base")
    else:
        active_vectorstore = new_store
        print(f"Created new knowledge base from {os.path.basename(pdf_path)}")
        
class WebContentProcessor:
    @staticmethod
    def download_pdf(url: str, temp_dir: str) -> Optional[str]:
        ''' Download PDF from URL to temporary directory '''
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
            print(f"Failed to download PDF from {url}: {str(e)}")
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
            return documents
        
        except Exception as e:
            print(f"Failed to process PDF {pdf_path}: {str(e)}")
            return []
        
    @staticmethod
    def load_web_content(url: str) -> List[Document]:
        ''' Load and process web page content '''
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            for doc in documents:
                doc.metadata["source_type"] = "web"
                doc.metadata["original_url"] = url
            return documents
        
        except Exception as e:
            print(f"Failed to process web-page {url}: {str(e)}")
            return []
        
    @staticmethod
    def process_url(url: str, temp_dir: str) -> List[Document]:
        ''' Automatically process either PDF or web URL '''
        if url.lower().endswith("pdf"):
            pdf_path = WebContentProcessor.download_pdf(url, temp_dir)
            if pdf_path:
                return WebContentProcessor.load_pdf_content(pdf_path)
        else:
            return WebContentProcessor.load_web_content(url)
        return []
    
    @staticmethod
    def google_search(query: str, num: int = 2):
        ''' Perform google search using Custom Search API '''
        api_key = os.environ.get("GOOGLE_SEARCH_KEY")
        cse_id = os.environ.get("CSE_ID")
        
        if not api_key or not cse_id:
            raise ValueError("Missing API credentials")
        
        params = {
            'q': query,
            'key': api_key,
            'cx': cse_id,
            'num': num,
            'fileType': 'pdf',
            'excludeTerms': 'youtube, video, watch',
            'siteSearch': '-youtube.com',
        }
        
        url = f"https://www.googleapis.com/customsearch/v1"
        response = requests.get(url, params)
        
        if response.status_code == 200:
            return response.json()
        raise Exception(f"API Error {response.status_code} : {response.text}")
    
    @staticmethod
    def filter_search_results(results: dict) -> list:
        ''' Filter out unwanted urls '''
        filtered_items = []
        for item in results.get('items', []):
            url = item.get('link', '')
            
            if any(skip in url.lower() for skip in ['youtube.com', 'video', 'watch']):
                continue
            
            if url.lower().endswith('.pdf') or any(
                domain in url.lower() for domain in ['blog', 'article', 'docs']
            ):
                filtered_items.append(item)
                
        return filtered_items
    
    @staticmethod
    def process_search_results(queries: list):
        ''' Process multiple queries and return documents '''
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
                    print(f"Error processing query '{query}': {str(e)}")
            
            return all_docs
            

class LLMOperations:
    @staticmethod
    def no_context_llm(question: str, prompt_template) -> str:
        ''' LLM without context '''
        qa = LLMChain(
                    llm = OpenAI(model = "gpt-4o-mini",temperature=0.7, max_tokens=1000),
                    prompt= prompt_template,
                    verbose= True
                )
        response = qa.run(question)
        return response
    
    @staticmethod
    def retrieval_qa(question: str, vectorstore: FAISS, prompt_template):
        ''' RetrievalQA with sources '''
        qa = RetrievalQA.from_chain_type(
                llm = OpenAI(model = "gpt-4o-mini", max_tokens=1000), 
                chain_type = "stuff", 
                chain_type_kwargs = {"prompt": prompt_template},
                retriever = vectorstore.as_retriever(),
                return_source_documents = True,
            )
        result = qa({"query": question})
        return result["result"], result["source_documents"]
     
         
class UserInterface:
    @staticmethod
    def authenticate():
        ''' Authentication function '''
        print("\n------ LOGIN ------")
        username = input("Username: ").strip()
        password = input("Password: ").strip()
        
        user = USER_DB.get(username)
        
        if ((user) and (user['password'] == hash_password(password))):
            print(f"Welcome {username}!")
            return user["role"]
        print("Invalid credentials!")
        return None   
   
    @staticmethod
    def handle_pdf_upload():
        ''' Handle pdf upload process '''
        print(f"Upload folder : {os.path.abspath(UPLOAD_DIR)}")
        pdf_path = input("Enter path to pdf file: ").strip()
        
        try:
            filename = os.path.basename(pdf_path)
            dest_path = os.path.join(UPLOAD_DIR, filename)
            
            if os.path.exists(dest_path):
                print("File already exists in Upload folder.")
                if input("Process it anyway? (y/n): ").lower() == 'y':
                    return
        
            add_pdf(pdf_path)
            
            with open(pdf_path, "rb") as src, open(dest_path, "wb") as dest:
                dest.write(src.read())
            print(f"Successfully uploaded and processsed {filename}")
            
        except Exception as e:
            print(f"Error : {str(e)}")
        
    @staticmethod      
    def admin_console():
        ''' Admin interface '''
        if UserInterface.authenticate() != "admin":
            print("Admin access required!")
            return
        
        while True:
            response = input("Upload pdf? (y/n): ")
            if (response.lower() == 'y'):
                UserInterface.handle_pdf_upload()
            else:
                return
    
    @staticmethod       
    def user_console():
        ''' User interface '''
        if not UserInterface.authenticate():
            return
        while True:
            print("\n------ MENU ------")
            print("1. No Search Option")
            print("2. Internal Search Option")
            print("3. External Search Option")
            print("4. Exit")
            
            choice = input("Enter choice (1-4): ")
            
            if (choice == "1"):
                question = input("Query: ")
                response = LLMOperations.no_context_llm(question, PROMPT_NO_SEARCH)
                print("Response: \n")
                print(response)
                
            elif (choice == "2"):
                pdf_path = os.path.join("pdf_to_read", "Prob_Stats_Module_4.pdf")
                new_vectorstore = process_pdf(pdf_path=pdf_path)
                question = input("Query: ")
                
                response, sources = LLMOperations.retrieval_qa(question, new_vectorstore, PROMPT_INTERNAL)

                print("\nResponse: ")
                print(response)
                
                print("\n\n### SOURCES ###\n\n")
                print(sources)
                
                for i in range(len(sources)):
                    print(len(sources[i].page_content))
            
            elif (choice == "3"):
                question = input("Query: ")
                search_queries = LLMOperations.no_context_llm(question, PROMPT_WEB_QUERY)
                queries = [q.strip().strip('"') for q in search_queries.split("\n") if q.strip()]
                
                print("Generated Search Queries: \n")
                for i, query in enumerate (queries, 1):
                    print(f"{i}. {query}")
                    
                documents = WebContentProcessor.process_search_results(queries)
                
                if documents:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        separators=[],
                        length_function=len,
                    )
                    docs = text_splitter.split_documents(documents)
                    vectorstore = FAISS.from_documents(docs, embeddings)
                    
                    response, sources = LLMOperations.retrieval_qa(question, new_vectorstore, PROMPT_INTERNAL)
                    
                    print("\nResponse: ")
                    print(response)
                    print("\nSources:")
                    for i, source in enumerate(sources, 1):
                        print(f"{i}. {source.metadata.get('source_url', 'Unknown source')}")
                else:
                    print("No relevant content found from web search.")
                
            elif (choice == "4"):
                exit()
            else:
                print("Invalid choice!")


if __name__ == "__main__":
    while True:
        print("\n------ MAIN MENU ------")
        print("1. Admin login")
        print("2. User login")
        print("3. Exit")
        
        choice = input("Enter choice (1-3): ")
        
        if (choice == "1"):
            UserInterface.admin_console()
        elif (choice == "2"):
            UserInterface.user_console()
        elif (choice == "3"):
            exit()
        else:
            print("Invalid choice.")

    
