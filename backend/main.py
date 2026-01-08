import os
import shutil
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI(title="RAG Chatbot API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = "./data"

# Global configuration with defaults
config = {
    "llm_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small",
    "chunk_size": 500,
    "chunk_overlap": 100,
    "separator": "\n\n",
    "temperature": 0.7,
    "retrieval_k": 4,
    "search_mode": "hybrid",  # Options: "semantic", "keyword", "hybrid"
    "prompt_template": """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, make something up.
Keep the answer concise.

Context: {context}

Question: {question}

Helpful Answer:"""
}

# Global state
vectorstore = None
bm25_retriever = None
document_splits = None
rag_chain = None
api_key = None


class QueryRequest(BaseModel):
    question: str


class RetrievedDocument(BaseModel):
    content: str
    source: str
    score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    retrieved_docs: List[RetrievedDocument] = []
    search_mode: str


class ConfigUpdate(BaseModel):
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    separator: Optional[str] = None
    temperature: Optional[float] = None
    retrieval_k: Optional[int] = None
    search_mode: Optional[str] = None
    prompt_template: Optional[str] = None


def get_openai_api_key():
    """Get the OpenAI API key from environment."""
    if "OPENAI_API_KEY" not in os.environ:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not found in environment variables")
    return os.environ["OPENAI_API_KEY"]


def setup_vector_store():
    """Loads documents, splits them, creates embeddings, and builds both FAISS and BM25 retrievers."""
    global vectorstore, bm25_retriever, document_splits, api_key

    if not os.path.isdir(DATA_PATH):
        raise HTTPException(status_code=404, detail=f"Data directory not found at {DATA_PATH}")

    loader = DirectoryLoader(DATA_PATH, glob=["**/*.txt", "**/*.pdf", "**/*.md"])
    docs = loader.load()

    if not docs:
        raise HTTPException(status_code=404, detail="No documents found in data directory")

    text_splitter = CharacterTextSplitter(
        separator=config["separator"],
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"]
    )
    splits = text_splitter.split_documents(docs)
    document_splits = splits

    embeddings = OpenAIEmbeddings(api_key=api_key, model=config["embedding_model"])

    try:
        # Create FAISS vector store for semantic search
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

        # Create BM25 retriever for keyword search
        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = config["retrieval_k"]

        return vectorstore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating vector store: {str(e)}")


def create_rag_chain():
    """Creates the RAG chain with the selected retriever based on search mode."""
    global rag_chain, vectorstore, bm25_retriever

    if vectorstore is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized")

    llm = ChatOpenAI(
        model=config["llm_model"],
        temperature=config["temperature"]
    )

    # Select retriever based on search mode
    search_mode = config.get("search_mode", "hybrid")

    if search_mode == "semantic":
        retriever = vectorstore.as_retriever(search_kwargs={"k": config["retrieval_k"]})
    elif search_mode == "keyword":
        if bm25_retriever is None:
            raise HTTPException(status_code=500, detail="BM25 retriever not initialized")
        bm25_retriever.k = config["retrieval_k"]
        retriever = bm25_retriever
    else:  # hybrid
        if bm25_retriever is None:
            raise HTTPException(status_code=500, detail="BM25 retriever not initialized")
        semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": config["retrieval_k"]})
        bm25_retriever.k = config["retrieval_k"]
        # Ensemble retriever combines both with equal weights (0.5, 0.5)
        retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )

    prompt = ChatPromptTemplate.from_template(config["prompt_template"])

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def get_retriever():
    """Get the current retriever based on search mode."""
    global vectorstore, bm25_retriever

    search_mode = config.get("search_mode", "hybrid")

    if search_mode == "semantic":
        return vectorstore.as_retriever(search_kwargs={"k": config["retrieval_k"]})
    elif search_mode == "keyword":
        bm25_retriever.k = config["retrieval_k"]
        return bm25_retriever
    else:  # hybrid
        semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": config["retrieval_k"]})
        bm25_retriever.k = config["retrieval_k"]
        return EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    global api_key
    try:
        api_key = get_openai_api_key()
        if os.path.isdir(DATA_PATH) and os.listdir(DATA_PATH):
            setup_vector_store()
            create_rag_chain()
    except Exception as e:
        print(f"Warning: Could not initialize on startup: {e}")


@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running"}


@app.get("/config")
async def get_config():
    """Get current RAG configuration."""
    return config


@app.post("/config")
async def update_config(config_update: ConfigUpdate):
    """Update RAG configuration and rebuild the chain."""
    global config, rag_chain

    # Update config
    update_dict = config_update.dict(exclude_unset=True)
    config.update(update_dict)

    # If embedding or chunking params changed, need to rebuild vectorstore
    rebuild_vectorstore = any(key in update_dict for key in
                             ["embedding_model", "chunk_size", "chunk_overlap", "separator"])

    if rebuild_vectorstore and vectorstore is not None:
        setup_vector_store()

    # Rebuild RAG chain with new config
    if vectorstore is not None:
        create_rag_chain()

    return {"message": "Configuration updated successfully", "config": config}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a query using the RAG chain and return retrieved documents."""
    global rag_chain

    if rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG chain not initialized. Please upload documents first.")

    try:
        # Get the retriever and fetch relevant documents
        retriever = get_retriever()
        retrieved_docs = retriever.get_relevant_documents(request.question)

        # Invoke the chain to get the answer
        answer = rag_chain.invoke(request.question)

        # Format retrieved documents for response
        retrieved_docs_formatted = []
        for i, doc in enumerate(retrieved_docs[:config["retrieval_k"]]):
            source = doc.metadata.get("source", "Unknown")
            # Extract just the filename from the path
            if source != "Unknown":
                source = os.path.basename(source)

            retrieved_docs_formatted.append(RetrievedDocument(
                content=doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                source=source,
                score=round(1.0 / (i + 1), 3)  # Simple scoring based on rank
            ))

        return QueryResponse(
            answer=answer,
            retrieved_docs=retrieved_docs_formatted,
            search_mode=config.get("search_mode", "hybrid")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a document to the data directory."""
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    file_path = os.path.join(DATA_PATH, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"message": f"File {file.filename} uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@app.post("/rebuild-vectorstore")
async def rebuild_vectorstore():
    """Rebuild the vector store with current documents and configuration."""
    try:
        setup_vector_store()
        create_rag_chain()
        return {"message": "Vector store rebuilt successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    """List all documents in the data directory."""
    if not os.path.exists(DATA_PATH):
        return {"documents": []}

    files = []
    for file in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, file)
        if os.path.isfile(file_path):
            files.append({
                "name": file,
                "size": os.path.getsize(file_path)
            })

    return {"documents": files}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)