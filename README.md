# RagifyText
**Mini RAG System with FastAPI and Open Source Embeddings**

A minimal Retrieval-Augmented Generation (RAG) service built with FastAPI, designed to answer questions from a small set of local documents (e.g. .txt or .pdf files).
It leverages open-source embedding models and a lightweight vector store (FAISS) for efficient retrieval and generation.

---

# Features

- **Data Ingestion Endpoint** – Upload text or PDF files for embedding and indexing.
- **Query Endpoint** – Ask natural language questions and receive answers with sources.
- **Extensible Design** – Pluggable embedding and text generation models.
- **Optional Enhancements** – Health check, caching, authentication

---

## Components:
- FastAPI for serving REST endpoints
- Sentence Transformers / BGE / Ada embeddings for vector representation
- FAISS for efficient similarity search
- Hugging Face pipelines or OpenAI models for text generation

---

Project structure
```
REPO
├──main.py
├──.env (optional)
├──storage/
├──data/
├── requirements.txt
├── Dockerfile
├── sample_data/
└──README.md
```

---

Steps to execute this app without Docker

1. Create a virtual env
```
python -m venv venv
<!-- Linux -->
source venv/bin/activate
<!-- Windows -->
venv\Scipts\activate
```

2. Install dependencies
```pip install -r requirements.txt```

3. Set environment variables in .env file. Add `HF_TOKEN`, `OPENAI_API_KEY`.
```env
HF_TOKEN=<your_huggingface_token>
OPENAI_API_KEY=<your_openai_key>
RAG_AUTH_TOKEN="dnfhs8392kkijfy3"
USE_OPEN_AI="0"  # set to 1 if using OpenAI Embeddings
```
> `RAG_AUTH_TOKEN` is needed to authenticate the request

4. Execute 
- using uvicorn
```
uvicorn app.main:app --reload
```
- Using Fastapi dev cli
```
fastapi dev main.py
```
# Using Docker

> Build the Image
```
docker build -t mini-rag-fastapi .
```

> Run the Container
```
docker run -p 8000:8000 \
  -e RAG_AUTH_TOKEN=... \
  -e OPENAI_API_KEY=... \
  -e HF_TOKEN=... \
  -e USE_OPEN_AI=true \
  -v $(pwd)/app/data:/app/data \
  -v $(pwd)/app/storage:/app/storage \
  mini-rag-fastapi:latest
```

# API Requests
1. Ingest API
```
curl -X 'POST' \
  'http://localhost:8000/ingest' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer <your_rag_auth_token>' \
  -H 'Content-Type: multipart/form-data' \
  -F 'files=@GDPR Compliance Policy.pdf;type=application/pdf' \
  -F 'files=@holiday_policy.txt;type=text/plain' \
  -F 'files=@leave_policy.txt;type=text/plain' \
  -F 'files=@Remote_work_policy.txt;type=text/plain' \
  -F 'files=@workplace_policy.txt;type=text/plain'
  
  #	Example Response body

    {
    "message": "18 chunks indexed"
    }
  ```

2. Query API
```
curl -X 'POST' \
  'http://localhost:8000/query' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer <your_rag_auth_token>' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "How many sick leaves are there?",
  "k": 2
}'
# Example Response
{
  "answer": "There are 10 working days of sick leave per year.",
  "sources": [
    "leave_policy.txt: line 36 - 67",
    "leave_policy.txt: line 1 - 44"
  ]
}
```
```
<!-- Another request -->
curl -X 'POST' \
  'http://localhost:8000/query' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer <your_rag_auth_token>' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "How long is the training for GDPR Compliance for new hires?",
  "k": 1
}'
# Example Response
{
  "answer": "New hires must complete data protection onboarding within 2 weeks of joining.",
  "sources": [
    "GDPR Compliance Policy.pdf: page 8"
  ]
}
```


3. Health API
```
curl -X 'GET' \
  'http://localhost:8000/health' \
  -H 'accept: application/json'

# Example Response Body
{
  "status": "ok"
}
  ```