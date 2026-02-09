import os
import shutil
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = "./data"

config = {
    "llm_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small",
    "chunk_size": 500,
    "chunk_overlap": 100,
    "separator": "\n\n",
    "temperature": 0.7,
    "retrieval_k": 4,
    "search_mode": "hybrid",  # hybrid, semantic, or keyword
    "prompt_template": """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, make something up.
Keep the answer concise.

Context: {context}

Question: {question}

Helpful Answer:"""
}

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
    if "OPENAI_API_KEY" not in os.environ:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not found")
    return os.environ["OPENAI_API_KEY"]


def setup_vector_store():
    global vectorstore, bm25_retriever, document_splits, api_key

    if not os.path.isdir(DATA_PATH):
        raise HTTPException(status_code=404, detail=f"Data directory not found at {DATA_PATH}")

    loader = DirectoryLoader(DATA_PATH, glob=["**/*.txt", "**/*.pdf", "**/*.md"])
    docs = loader.load()

    if not docs:
        raise HTTPException(status_code=404, detail="No documents found")

    text_splitter = CharacterTextSplitter(
        separator=config["separator"],
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"]
    )
    splits = text_splitter.split_documents(docs)
    document_splits = splits

    embeddings = OpenAIEmbeddings(api_key=api_key, model=config["embedding_model"])

    vectorstore = FAISS.from_documents(splits, embeddings)
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = config["retrieval_k"]


def hybrid_search(query: str, k: int):
    """
    Custom hybrid search combining semantic and keyword search.
    Uses 60% semantic weight and 40% keyword weight.
    """
    global vectorstore, bm25_retriever

    # Get results from both retrievers
    bm25_retriever.k = k * 2  # Get more results to ensure good coverage
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": k * 2})

    semantic_docs = semantic_retriever.invoke(query)
    keyword_docs = bm25_retriever.invoke(query)

    # Score documents using Reciprocal Rank Fusion with weights
    doc_scores = {}

    # Semantic results with 60% weight
    for rank, doc in enumerate(semantic_docs):
        doc_id = doc.page_content  # Use content as unique identifier
        score = 0.6 * (1.0 / (rank + 1))  # RRF score with semantic weight
        doc_scores[doc_id] = {"score": score, "doc": doc}

    # Keyword results with 40% weight
    for rank, doc in enumerate(keyword_docs):
        doc_id = doc.page_content
        score = 0.4 * (1.0 / (rank + 1))  # RRF score with keyword weight
        if doc_id in doc_scores:
            doc_scores[doc_id]["score"] += score  # Add to existing score
        else:
            doc_scores[doc_id] = {"score": score, "doc": doc}

    # Sort by combined score and return top k
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_docs[:k]]


def get_retriever():
    global vectorstore, bm25_retriever

    if config["search_mode"] == "keyword":
        if bm25_retriever is None:
            raise HTTPException(status_code=500, detail="BM25 not initialized")
        bm25_retriever.k = config["retrieval_k"]
        return bm25_retriever
    elif config["search_mode"] == "hybrid":
        # Return None for hybrid, we'll handle it separately in the query endpoint
        return None
    else:  # semantic
        return vectorstore.as_retriever(search_kwargs={"k": config["retrieval_k"]})


def create_rag_chain():
    global rag_chain

    llm = ChatOpenAI(
        model=config["llm_model"],
        temperature=config["temperature"]
    )

    retriever = get_retriever()
    prompt = ChatPromptTemplate.from_template(config["prompt_template"])

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


@app.on_event("startup")
async def startup_event():
    global api_key
    try:
        api_key = get_openai_api_key()
        if os.path.isdir(DATA_PATH) and os.listdir(DATA_PATH):
            setup_vector_store()
            create_rag_chain()
    except Exception as e:
        print("Startup warning:", e)


@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running"}


@app.get("/config")
async def get_config():
    return config


@app.post("/config")
async def update_config(config_update: ConfigUpdate):
    global config

    update_dict = config_update.dict(exclude_unset=True)
    config.update(update_dict)

    if any(k in update_dict for k in ["embedding_model", "chunk_size", "chunk_overlap", "separator"]):
        if vectorstore is not None:
            setup_vector_store()

    if vectorstore is not None:
        create_rag_chain()

    return {"message": "Config updated", "config": config}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    # Handle hybrid search separately
    if config["search_mode"] == "hybrid":
        if vectorstore is None or bm25_retriever is None:
            raise HTTPException(status_code=500, detail="Retrievers not initialized")

        # Get hybrid search results
        retrieved_docs = hybrid_search(request.question, config["retrieval_k"])

        # Create context from retrieved docs
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Generate answer using LLM with context
        llm = ChatOpenAI(
            model=config["llm_model"],
            temperature=config["temperature"]
        )
        prompt = ChatPromptTemplate.from_template(config["prompt_template"])
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": request.question})
    else:
        if rag_chain is None:
            raise HTTPException(status_code=500, detail="RAG not initialized")
        retriever = get_retriever()
        retrieved_docs = retriever.invoke(request.question)
        answer = rag_chain.invoke(request.question)

    formatted = []
    for i, doc in enumerate(retrieved_docs[:config["retrieval_k"]]):
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        formatted.append(RetrievedDocument(
            content=doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
            source=source,
            score=round(1.0 / (i + 1), 3)
        ))

    return QueryResponse(
        answer=answer,
        retrieved_docs=formatted,
        search_mode=config["search_mode"]
    )


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    os.makedirs(DATA_PATH, exist_ok=True)
    path = os.path.join(DATA_PATH, file.filename)

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"message": f"{file.filename} uploaded"}


@app.post("/rebuild-vectorstore")
async def rebuild_vectorstore():
    setup_vector_store()
    create_rag_chain()
    return {"message": "Vector store rebuilt"}


@app.get("/documents")
async def list_documents():
    if not os.path.exists(DATA_PATH):
        return {"documents": []}

    return {
        "documents": [
            {"name": f, "size": os.path.getsize(os.path.join(DATA_PATH, f))}
            for f in os.listdir(DATA_PATH)
            if os.path.isfile(os.path.join(DATA_PATH, f))
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
