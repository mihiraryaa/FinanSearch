import os
import sys
import getpass
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


DATA_PATH = "./data"


#    To load Markdown, change to "**/*.md" a
FILE_PATTERN = ["**/*.txt", "**/*.pdf"]
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"




def get_openai_api_key():
    """Securely get the OpenAI API key from environment or user input."""
    if "OPENAI_API_KEY" in os.environ:
        return os.environ["OPENAI_API_KEY"]
    else:
        print("OpenAI API key not found in environment variables.")
        return getpass.getpass("Please enter your OpenAI API Key: ")


def setup_vector_store(api_key):
    """
    Loads documents, splits them, creates embeddings, 
    and builds the FAISS vector store.
    """
    print(f"Loading documents from {DATA_PATH}...")
    
    if not os.path.isdir(DATA_PATH):
        print(f"Error: Directory not found at {DATA_PATH}")
        print("Please create the 'data' folder and add your text files.")
        sys.exit(1)

    loader = DirectoryLoader(DATA_PATH, glob=FILE_PATTERN)
    docs = loader.load()

    if not docs:
        print(f"No documents matching '{FILE_PATTERN}' found in {DATA_PATH}.")
        print("Please add some files to the 'data' folder.")
        sys.exit(1)

    print(f"Loaded {len(docs)} documents.")

    print("Splitting documents...")
    text_splitter = CharacterTextSplitter(
        separator="\n\n",  # Split on paragraphs
        chunk_size=500,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(docs)

    print("Initializing embeddings model...")
    embeddings = OpenAIEmbeddings(api_key=api_key, model=EMBEDDING_MODEL)

    print("Creating FAISS vector store... (This may take a moment)")
    try:
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        print("Vector store created successfully.")
        return vectorstore
    except Exception as e:
        print(f"An error occurred while creating the vector store: {e}")
        print("This might be due to an invalid API key or network issues.")
        sys.exit(1)

def create_rag_chain(vectorstore):
    """Creates the RAG chain using LangChain Expression Language (LCEL)."""
    
    llm = ChatOpenAI(model=LLM_MODEL)
    
    retriever = vectorstore.as_retriever()
    
    template = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, make something up.
    Keep the answer concise.

    Context: {context}

    Question: {question}

    Helpful Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

   
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def main():
    """Main function to run the RAG chatbot."""
    
    api_key = get_openai_api_key()
    
    vectorstore = setup_vector_store(api_key)
    
    rag_chain = create_rag_chain(vectorstore)

    print("\n--- RAG Chatbot is Ready ---")
    print("Ask questions about your documents.")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        try:
            query = input("\nYour question: ")
            
            if query.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if not query.strip():
                continue

            print("Thinking...")
            
            # 5. Invoke the chain and get the answer
            answer = rag_chain.invoke(query)
            
            print("\nAnswer:")
            print(answer)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()