# ================== imports ==================
import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from langsmith import traceable

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ✅ NEW correct import (fixes deprecation + error)
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
@traceable(name="load_pdf")
def load_pdf(path: str):
    return PyPDFLoader(path).load()

@traceable(name="split_documents")
def split_documents(docs, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)

@traceable(name="build_vectorstore")
def build_vectorstore(splits, embed_model_name: str="sentence-transformers/all-mpnet-base-v2"):
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model_name
    )
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
    return hashlib.sha256(
        json.dumps(meta, sort_keys=True).encode()
    ).hexdigest()

# ================== index build/load ==================
@traceable(name="load_index")
def load_index(index_dir: Path, embed_model_name: str):
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model_name
    )
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )

@traceable(name="build_index")
def build_index(pdf_path, index_dir, chunk_size, chunk_overlap, embed_model_name):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs, chunk_size, chunk_overlap)
    vs = build_vectorstore(splits, embed_model_name)

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
        return load_index(index_dir, embed_model_name)

    return build_index(
        pdf_path,
        index_dir,
        chunk_size,
        chunk_overlap,
        embed_model_name,
    )

# ================== LLM & Prompt ==================
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful medical assistant that provides accurate, structured responses based on medical document content. Always follow this exact format:

**Brief Summary:** [2-3 sentences summarizing the main medical points]

**Key Medical Information:**
* [Medical fact 1]
* [Medical fact 2]
* [Medical fact 3]

**Detailed Medical Content:**
### [Medical Topic/Section]
[Medical information with proper formatting]

### [Medical Topic/Section]
[Medical information with proper formatting]

### Medical Advice (if applicable)
⚠️ [Important medical disclaimer or advice]

Always start with "According to the medical document:" and only use information from the provided context. If medical information is not found, say "Medical information not found in the document." Use markdown formatting for headers, bullet points, and emphasize medical terminology appropriately. Remember to include appropriate medical disclaimers when giving health advice."""),

    ("human", "Medical Question: {question}\n\nMedical Context:\n{context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# ================== RAG PIPELINE ==================
@traceable(name="medical_rag_run")
def run_medical_rag(pdf_path, question):
    try:
        vectorstore = load_or_build_index(pdf_path)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        chain = (
            RunnableParallel({
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            })
            | prompt
            | llm
            | StrOutputParser()
        )

        return chain.invoke(question)
    except Exception as e:
        logger.error(f"Error in medical RAG pipeline: {str(e)}")
        return f"Error processing your medical question: {str(e)}"

# ================== CLI ==================
if __name__ == "__main__":
    print("🏥 Medical Chatbot Ready. Ask a medical question (Ctrl+C to exit)")
    print("=" * 50)
    while True:
        try:
            q = input("\nMedical Question: ").strip()
            if not q:
                continue
            ans = run_medical_rag(PDF_PATH, q)
            print("\n📝 Response:")
            print(ans)
            print("\n" + "=" * 50)
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            continue