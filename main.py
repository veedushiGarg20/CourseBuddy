import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

load_dotenv()

if __name__ == "__main__":
    pdf_path = os.path.join("pdf_to_read", "Prob_Stats_Module_4.pdf")
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    

    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    documents = text_splitter.split_documents(documents=documents)
    # print(documents[5].page_content)
    # exit()

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("faiss_index_pdf")

    new_vectorstore = FAISS.load_local("faiss_index_pdf", embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(model = "gpt-4o-mini", max_tokens=800), chain_type="stuff", retriever=new_vectorstore.as_retriever()
    )

    response = qa.run("explain binomial distribution and it's formula. also explain the solution of problem 1 of binomial distribution")

    print(response)
