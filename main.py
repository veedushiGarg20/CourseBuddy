import os
import hashlib
import requests
from urllib.parse import quote
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.retrieval_qa.prompt import PROMPT_INTERNAL
from langchain.chains.retrieval_qa.prompt import PROMPT_NO_SEARCH
from langchain.chains.retrieval_qa.prompt import PROMPT_WEB_QUERY

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
        
        
def no_context_llm(question, my_prompt) -> str:
    qa = LLMChain(
                llm = OpenAI(model = "gpt-4o-mini",temperature=0.7, max_tokens=1000),
                prompt= my_prompt,
                verbose= True
            )
    response = qa.run(question)
    return response


def google_search(query, num=3):
    encoded_query = quote(query)
    url = f"https://www.googleapis.com/customsearch/v1?q={encoded_query}&key={os.environ.get('GOOGLE_SEARCH_KEY')}&cx={os.environ.get('CSE_ID')}&num={num}"
    
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return None
    
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")
        return None
    


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


        
def admin_console():
    ''' Admin interface '''
    if authenticate() != "admin":
        print("Admin access required!")
        return
    
    while True:
        response = input("Upload pdf? (y/n): ")
        if (response.lower() == 'y'):
            handle_pdf_upload()
        else:
            return
 
       
def user_console():
    ''' User interface '''
    if not authenticate():
        return
    while True:
        print("------ MENU ------")
        print("1. No Search Option")
        print("2. Internal Search Option")
        print("3. External Search Option")
        print("4. Exit")
        
        choice = input("Enter choice: ")
        
        if (choice == "1"):
            question = input("Query: ")
            response = no_context_llm(question, PROMPT_NO_SEARCH)
            print("Response: \n")
            print(response)
            
        elif (choice == "2"):
            pdf_path = os.path.join("pdf_to_read", "Prob_Stats_Module_4.pdf")
            
            new_vectorstore = process_pdf(pdf_path=pdf_path)
            
            qa = RetrievalQA.from_chain_type(
                llm=OpenAI(model = "gpt-4o-mini", max_tokens=1000), 
                chain_type="stuff", 
                chain_type_kwargs= {"prompt": PROMPT_INTERNAL},
                retriever=new_vectorstore.as_retriever(),
                return_source_documents = True,
            )

            question = "What is poisson distribution? Explain its pmf formula."
            result = qa({"query": question})
            
            response = result["result"]
            sources = result["source_documents"]

            print(response)
            
            print("\n\n### SOURCES ###\n\n")
            print(sources)
            
            for i in range(len(sources)):
                print(len(sources[i].page_content))
        elif (choice == "3"):
            question = input("Query: ")
            response = no_context_llm(question, PROMPT_WEB_QUERY)
            print("Response: \n")
            response_list = response.split("\n")
            for i, query in enumerate (response_list, 1):
                clean_query = query.strip().strip('"')
                if not clean_query:
                    continue
                
                print(f"\n{i}. Searching for: '{clean_query}'")
                print("-" * 50)
                
                try:
                    result = google_search(clean_query)
                    print(result)

                    if result and 'items' in result:
                        for j, item in enumerate(result['items'][:3], 1):
                            print(f"\nResult{j}: ")
                            print(f"Title: {item.get('title', 'No title')}")
                            print(f"URL: {item.get('link', 'No URL')}")
                            print(f"Description: {item.get('snippet', 'No description')}")
                            
                    else:
                        print("No results found or error occurred.")
                        
                except Exception as e:
                    print(f"Error processing query '{clean_query}': {str(e)}")
            
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
        
        choice = input("Enter choice (1-2): ")
        
        if (choice == "1"):
            admin_console()
        elif (choice == "2"):
            user_console()
        elif (choice == "3"):
            exit()
        else:
            print("Invalid choice.")

    
