"""A simple RAG (Retrieval-Augmented Generation) service with FastAPI.

Accepts file uploads, processes them into a vector store, and allows querying.
"""

import os
import dotenv
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from typing import List
from pydantic import BaseModel, Field

from llama_index.llms.openai import OpenAI

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from sentence_transformers import SentenceTransformer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
    Settings,
)

from llama_index.vector_stores.faiss import FaissVectorStore
import faiss


# Loads environment variables
dotenv.load_dotenv()

# Logging configuration
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mini RAG Service")

# Ensure storage and data directory exists.
DATA_DIR = os.getenv("DATA_DIR", "data")
STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)
# For token based authentication
bearer = HTTPBearer(auto_error=True)


def auth_dependency(creds: HTTPAuthorizationCredentials = Depends(bearer)):
    """Authenticate using a bearer token."""
    expected = os.getenv("RAG_AUTH_TOKEN")
    token = creds.credentials
    if expected and token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True


def add_line_numbers_to_nodes(documents, nodes):
    """Add line number metadata to each node based on the document text."""
    # Map file name to its text (for fast lookup)
    doc_texts = {d.metadata.get("file_name", d.id_): d.text for d in documents}
    for node in nodes:
        metadata = node.metadata
        if metadata.get("file_type") == "application/pdf":
            metadata["line_start"], metadata["line_end"] = (None, None)
            continue
        file_name = metadata.get("file_name", node.id_)
        full_text = doc_texts[file_name]
        if not full_text:
            continue
        full_text = full_text.replace("\r\n", "\n")
        node_text = node.text.replace("\r\n", "\n")

        # Find where this chunk occurs in the full text
        start_char = full_text.find(node_text)
        if start_char == -1:
            continue  # chunk not found

        # Count lines before this chunk
        start_line = full_text[:start_char].count("\n") + 1
        end_line = start_line + node.text.count("\n")

        # Save in metadata
        metadata["file_name"] = file_name
        metadata["line_start"] = start_line
        metadata["line_end"] = end_line

    return nodes


# Models
class QueryIn(BaseModel):
    """Input model for query endpoint."""

    question: str = Field(..., min_length=3, description="User's question")
    k: int = Field(5, ge=1, le=20, description="Top-k chunks to retrieve")


class QueryOut(BaseModel):
    """Output model for query endpoint."""

    answer: str
    sources: List[str]


USE_OPEN_AI = os.getenv("USE_OPEN_AI", "false").lower() in ("1", "true")
api_key = os.getenv("OPENAI_API_KEY")

# Utility functions
def get_embedding_model():
    """Return a tuple containing embedding model and its dimensions."""
    global USE_OPEN_AI
    if USE_OPEN_AI:
        logger.info("Using OpenAI embeddings")
        return (OpenAIEmbedding(model="text-embedding-3-small"), 1536)
    else:
        embed_model_namae = "BAAI/bge-small-en-v1.5"
        logger.info(f"Using HuggingFace embeddings {embed_model_namae}")
        st_model = SentenceTransformer(embed_model_namae)
        dim = st_model.get_sentence_embedding_dimension()
        logger.info("bge-small-en embedding")
        return (
            HuggingFaceEmbedding(model_name=embed_model_namae, normalize=True),
            dim)


SYSTEM_PROMPT = """Respond to the user query based on the provided context.
     Output format:
       - A short answer in 1 to 5 lines.
"""

model_to_use = "gpt-5-mini"
LLM = OpenAI(model=model_to_use, system_prompt=SYSTEM_PROMPT, api_key=api_key)

Settings.llm = LLM
embedding_used, EMBED_DIM = get_embedding_model()
Settings.embed_model = embedding_used
splitter = SentenceSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separator="\n",
    paragraph_separator="\n",
    include_prev_next_rel=True
)


def get_faiss_vector_store():
    """Return a FAISS vector store instance based on embedding model."""
    global EMBED_DIM, USE_OPEN_AI
    if USE_OPEN_AI:
        return FaissVectorStore(faiss_index=faiss.IndexFlatL2(EMBED_DIM))
    else:
        return FaissVectorStore(faiss_index=faiss.IndexFlatIP(EMBED_DIM))


@app.post("/ingest", dependencies=[Depends(auth_dependency)])
async def ingest(
    files: List[UploadFile] = File(None,  description="TXT/PDF files")
):
    """Ingest files and return a success message."""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        logger.info(f"Received {len(files)} files for ingestion.")
        global DATA_DIR, STORAGE_DIR
        input_files = []
        for file in files:
            logger.info(f"checking file: {file.filename}")
            if file.content_type not in ["application/pdf", "text/plain"]:
                logger.error(f"Unsupported file type: {file.content_type}")
                raise HTTPException(
                    status_code=400,
                    detail=f"{file.filename} has unsupported file extension."
                )
            logger.info(f"Processing file: {file.filename}, \
                content_type: {file.content_type}")
            with open(os.path.join(DATA_DIR, file.filename), "wb") as f:
                content = await file.read()
                f.write(content)
                logger.info(f"Wrote {len(content)} bytes to {file.filename}")
            input_files.append(os.path.join(DATA_DIR, file.filename))

        documents = SimpleDirectoryReader(
            input_files=input_files, filename_as_id=True
        ).load_data()
        global Settings, splitter

        storage_context = StorageContext.from_defaults(
            vector_store=get_faiss_vector_store()
        )
        nodes = splitter.get_nodes_from_documents(documents)
        nodes = add_line_numbers_to_nodes(documents, nodes)
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            persist_dir=STORAGE_DIR,
            show_progress=True
        )
        # save index to disk
        index.storage_context.persist(persist_dir=STORAGE_DIR)
        return {"message": f"{len(nodes)} chunks indexed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/query",
    response_model=QueryOut,
    summary="Ask a question",
    dependencies=[Depends(auth_dependency)]
    )
async def search(body: QueryIn):
    """Query the vector store and return an answer.

    Accepts a JSON payload with a "question" field.
    """
    if not body.question:
        raise HTTPException(
            status_code=400,
            detail="Question is required"
            )
    if not body.k:
        raise HTTPException(
            status_code=400,
            detail="k is required"
            )
    if body.k < 1 or body.k > 20:
        raise HTTPException(
            status_code=400,
            detail="k must be between 1 and 20"
        )
    
    try:
        global embedding_used, LLM
        Settings.embed_model = embedding_used
        print("Loading index from storage...")
        # load index from disk
        storage_context = StorageContext.from_defaults(
            vector_store=FaissVectorStore.from_persist_dir(STORAGE_DIR),
            persist_dir=STORAGE_DIR
        )
        index = load_index_from_storage(storage_context=storage_context)

        query_engine = index.as_query_engine(
            llm=LLM,
            similarity_top_k=body.k
        )
        logger.info(f"Querying index for...{body.question}")
        response = query_engine.query(body.question)
        logger.info(response.response)
        sources = []
        for node in response.source_nodes:
            source = f"{node.metadata['file_name']}: "
            is_pdf = node.metadata['file_type'] == 'application/pdf'
            if is_pdf:
                source += f"page {node.metadata.get('page_label', 'unknown')}"
            else:
                source += f"line {node.metadata.get('line_start', 'unknown')}"
                source += f" - {node.metadata.get('line_end','unknown')}"
            sources.append(source)
        return QueryOut(
            answer=response.response,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}
