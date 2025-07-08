# CourseBuddy

CourseBuddy is a Streamlit web app for uploading, indexing and querying PDF documents using OpenAI embeddings and LangChain. It also supports web search integration.

## Features

- Upload and process PDFs
- Vector-based semantic search on uploaded documents
- Web search integration
- User/Admin login with password hashing


## Setup
1. Clone this repo.
2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Create a .env file in the root directory with your keys:
```bash
OPENAI_API_KEY=your_openai_key
ADMIN_USR=admin
ADMIN_PSWD=hashed_password
USER_USR=user
USER_PSWD=hashed_password
GOOGLE_SEARCH_KEY=your_google_key
CSE_ID=your_cse_id
```
5. To run the app locally:
```bash
streamlit run app.py
```
