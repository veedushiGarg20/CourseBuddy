import os
import hashlib
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
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
    pdf_path = os.path.join("pdf_to_read", "Prob_Stats_Module_4.pdf")
    
    new_vectorstore = process_pdf(pdf_path=pdf_path)
    
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(model = "gpt-4o-mini", max_tokens=1000), 
        chain_type="stuff", 
        chain_type_kwargs= {"prompt": PROMPT},
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
