# flake8: noqa
from langchain_core.prompts import PromptTemplate


# Internal Search option
prompt_template_internal = """Use the following pieces of context to answer the question at the end. Provide accurate inline citations which include only the pdf name and page number from the context for every line. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
PROMPT_INTERNAL = PromptTemplate(
    template=prompt_template_internal, input_variables=["context", "question"]
)



# No Search option
prompt_template_no_search = """You are an intelligent AI LLM model. Read the following question, comprehend its meaning accurately and answer it to best of your abilities. If you don't know the answer, just say you don't know. Don't try to make up an answer.

Question: {question}
Helpful Answer:"""
PROMPT_NO_SEARCH = PromptTemplate(
    template=prompt_template_no_search, input_variables=["question"]
)


# External Search Option
prompt_template_google_query = """You are a search query generator. Your only task is to generate EXACTLY 3 search queries in the following format - nothing more, nothing less. Do NOT include any explanations, headers or additional text.

STRICT FORMAT:
"Query 1"
"Query 2"
"Query 3"

Question: {question}
Search Query:"""
PROMPT_WEB_QUERY = PromptTemplate(
    template=prompt_template_google_query, input_variables=["question"]
)