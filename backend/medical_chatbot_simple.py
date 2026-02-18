# ================== imports ==================
import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import CallbackManagerForLLMRun
from typing import Any, Dict, Iterator
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================== env ==================
load_dotenv()

BASE_DIR = Path(__file__).parent
PDF_PATH = BASE_DIR / "Medical_book.pdf"
INDEX_ROOT = BASE_DIR / ".indices"
INDEX_ROOT.mkdir(exist_ok=True)

# ================== helpers ==================
def load_pdf(path: str):
    return PyPDFLoader(path).load()

def split_documents(docs, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)

def build_vectorstore(splits, embed_model_name: str="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
    return FAISS.from_documents(splits, embeddings)

# ================== cache helpers ==================
def _file_fingerprint(path: str) -> dict:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {
        "sha256": h.hexdigest(),
        "size": p.stat().st_size,
        "mtime": int(p.stat().st_mtime),
    }

def _index_key(pdf_path, chunk_size, chunk_overlap, embed_model_name):
    meta = {
        "pdf": _file_fingerprint(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding": embed_model_name,
    }
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()

# ================== index build/load ==================
def load_index(index_dir: Path, embed_model_name: str):
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
    return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)

def build_index(pdf_path, index_dir, chunk_size, chunk_overlap, embed_model_name):
    logger.info("Loading PDF...")
    docs = load_pdf(pdf_path)
    logger.info(f"Loaded {len(docs)} pages")
    
    logger.info("Splitting documents...")
    splits = split_documents(docs, chunk_size, chunk_overlap)
    logger.info(f"Created {len(splits)} chunks")
    
    logger.info("Building vector store...")
    vs = build_vectorstore(splits, embed_model_name)
    
    logger.info("Saving index...")
    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))
    return vs

def load_or_build_index(
    pdf_path,
    chunk_size=1000,
    chunk_overlap=150,
    embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
    force_rebuild=False,
):
    key = _index_key(pdf_path, chunk_size, chunk_overlap, embed_model_name)
    index_dir = INDEX_ROOT / key

    if index_dir.exists() and not force_rebuild:
        logger.info("Loading existing index...")
        return load_index(index_dir, embed_model_name)

    logger.info("Building new index...")
    return build_index(pdf_path, index_dir, chunk_size, chunk_overlap, embed_model_name)

# ================== LLM & Prompt ==================
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a medical assistant that ONLY answers questions based on the provided medical document context.

STRICT RULES:
1. ONLY use information from the provided medical document context
2. If the answer is not found in the context, respond with: "I cannot answer this question based on the provided medical document."
3. Do NOT use any external medical knowledge or general information
4. Do NOT make assumptions or provide information outside the document
5. Be concise and only include information that is explicitly stated in the context

Focus ONLY on the medical content provided in the context below."""),
    ("human", "Medical Question: {question}\n\nProvided Medical Document Context:\n{context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# ================== RAG PIPELINE ==================
def run_medical_rag(pdf_path, question):
    try:
        logger.info("Loading or building index...")
        vectorstore = load_or_build_index(pdf_path)
        
        logger.info("Setting up retriever...")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        logger.info("Creating chain...")
        chain = (
            RunnableParallel({
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            })
            | prompt
            | llm
            | StrOutputParser()
        )

        logger.info("Generating response...")
        response = chain.invoke(question)
        return response
    except Exception as e:
        logger.error(f"Error in medical RAG pipeline: {str(e)}")
        return f"Error processing your question: {str(e)}"

def run_medical_rag_stream(pdf_path, question):
    try:
        logger.info("Loading or building index...")
        vectorstore = load_or_build_index(pdf_path)
        
        logger.info("Setting up retriever...")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        logger.info("Creating chain...")
        chain = (
            RunnableParallel({
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            })
            | prompt
            | llm
            | StrOutputParser()
        )

        logger.info("Streaming response...")
        response = ""
        for chunk in chain.stream(question):
            response += chunk
            sys.stdout.write(chunk)
            sys.stdout.flush()
        
        return response
    except Exception as e:
        logger.error(f"Error in medical RAG pipeline: {str(e)}")
        return f"Error processing your question: {str(e)}"

# ================== CLI ==================
if __name__ == "__main__":
    print("🏥 Medical Chatbot Ready. Ask a medical question (Ctrl+C to exit)")
    print("=" * 50)
    
    # Pre-load the index to avoid delays on first query
    print("Pre-loading medical knowledge base...")
    try:
        vectorstore = load_or_build_index(PDF_PATH)
        print("✅ Knowledge base loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading knowledge base: {e}")
        exit(1)
    
    while True:
        try:
            q = input("\nMedical Question: ").strip()
            if not q:
                continue
            
            print("Processing your question...")
            print("\n📝 Response:")
            ans = run_medical_rag_stream(PDF_PATH, q)
            print("\n" + "=" * 50)
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            continue