# FinanSearch - RAG Chatbot System

A full-stack Retrieval-Augmented Generation (RAG) chatbot application. This system allows you to upload documents and ask questions about them using advanced language models.

## Features

- **Document Upload**: Support for PDF, TXT, and Markdown files
- **RAG-based Q&A**: Ask questions about your uploaded documents
- **Hybrid Search System** ⭐ (Unique Feature):
  - **Semantic Search**: AI-powered understanding using embeddings
  - **Keyword Search**: BM25 algorithm for exact term matching
  - **Hybrid Mode**: Combines both approaches for optimal results
  - **Visual Feedback**: See retrieved document chunks with relevance scores
  - **Search Mode Toggle**: Switch between search modes in real-time
- **Configurable Parameters**: Adjust all RAG settings via the UI
  - LLM model selection (GPT-4o, GPT-4o-mini, GPT-3.5-turbo)
  - Embedding model selection
  - Chunk size and overlap
  - Temperature control
  - Number of retrieved documents
  - Custom prompt templates
- **Real-time Chat Interface**: Clean, modern UI for chatting with your documents
- **Retrieved Context Display**: See exactly which document chunks were used to generate each answer
- **Vector Store Management**: Rebuild embeddings on demand

## Tech Stack

**Backend:**
- FastAPI
- LangChain
- OpenAI API (GPT models & embeddings)
- FAISS (vector storage)

**Frontend:**
- React 18
- Vite
- Axios

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 16+
- OpenAI API key

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd FinanSearch/backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv

   # On Windows:
   venv\Scripts\activate

   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the backend directory:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. Create a `data` folder and add your documents:
   ```bash
   mkdir data
   # Add your .txt, .pdf, or .md files to the data folder
   ```

6. Run the FastAPI server:
   ```bash
   python main.py
   ```

   The backend will be available at `http://localhost:8000`

   API documentation: `http://localhost:8000/docs`

### Frontend Setup

1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd FinanSearch/frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

   The frontend will be available at `http://localhost:5173`

## Usage

1. **Upload Documents**: Click "Upload Docs" and select files to upload
2. **Rebuild Vector Store**: After uploading, click "Rebuild Vector Store" to index your documents
3. **Configure RAG**: Click "Show Config" to adjust RAG parameters
   - Select search mode: Hybrid, Semantic, or Keyword
   - Adjust model settings, chunking parameters, and retrieval count
4. **Ask Questions**: Type your questions in the chat interface
5. **View Retrieved Context**: See which document chunks were used, their sources, and relevance scores

## API Endpoints

- `GET /` - Health check
- `GET /config` - Get current configuration
- `POST /config` - Update configuration
- `POST /query` - Ask a question
- `POST /upload` - Upload a document
- `POST /rebuild-vectorstore` - Rebuild the vector store
- `GET /documents` - List uploaded documents

## Configuration Parameters

- **Search Mode**: Choose between Hybrid, Semantic, or Keyword search
  - **Hybrid**: Combines BM25 keyword matching with semantic embeddings (recommended)
  - **Semantic**: Uses AI embeddings to understand meaning and context
  - **Keyword**: Uses BM25 algorithm for exact term matching
- **LLM Model**: The language model used for answering questions
- **Embedding Model**: The model used to create document embeddings
- **Chunk Size**: Size of text chunks (in characters)
- **Chunk Overlap**: Overlap between consecutive chunks
- **Temperature**: Controls response randomness (0-2)
- **Retrieval K**: Number of relevant chunks to retrieve
- **Prompt Template**: The system prompt for the LLM

## Project Structure

```
FinanSearch/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   ├── .env                 # Environment variables
│   └── data/                # Document storage
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React component
│   │   ├── App.css          # Styling
│   │   ├── main.jsx         # Entry point
│   │   └── index.css        # Global styles
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## AI/ML Lab Project

This project demonstrates:
- **Hybrid Retrieval Systems**: Combining lexical (BM25) and semantic (embeddings) search
- **Ensemble Methods**: Weighted combination of multiple retrieval strategies
- Document preprocessing and chunking
- Vector embeddings and similarity search
- Retrieval-Augmented Generation (RAG)
- Information retrieval evaluation and visualization
- API design and development
- Full-stack application development

### Why Hybrid Search?

Traditional keyword search (BM25) excels at finding exact term matches but struggles with synonyms and context. Semantic search using embeddings understands meaning but can miss specific terminology. Our hybrid approach:

1. **BM25 Retriever**: Finds exact keyword matches using term frequency-inverse document frequency
2. **FAISS Semantic Retriever**: Uses OpenAI embeddings to understand contextual meaning
3. **Ensemble Retriever**: Combines both with configurable weights (default 50/50)

This results in more robust retrieval that works well for both specific queries ("find documents mentioning 'revenue'") and conceptual questions ("what is the company's financial performance?").

