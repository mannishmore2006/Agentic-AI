from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import logging

from medical_chatbot_simple import run_medical_rag_stream, PDF_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical Chatbot API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    status: str = "success"

class StreamQueryRequest(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "🏥 Medical Chatbot API is running!", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "medical-chatbot"}

@app.post("/ask-stream")
async def ask_medical_question_stream(request: StreamQueryRequest):
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        logger.info(f"Processing streaming medical question: {request.question}")
        
        async def generate_response():
            # Import here to avoid circular imports
            from medical_chatbot_simple import run_medical_rag_stream
            import asyncio
            
            # This is a simplified approach - in production, you'd want to modify
            # the chain to support async streaming properly
            response = run_medical_rag_stream(str(PDF_PATH), request.question.strip())
            yield response
            
        return StreamingResponse(generate_response(), media_type="text/plain")
    except Exception as e:
        logger.error(f"Error processing streaming question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing your question: {str(e)}")

@app.post("/ask", response_model=QueryResponse)
async def ask_medical_question(request: QueryRequest):
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        logger.info(f"Processing medical question: {request.question}")
        answer = run_medical_rag(str(PDF_PATH), request.question.strip())
        
        return QueryResponse(answer=answer)
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing your question: {str(e)}")

@app.get("/info")
async def get_info():
    return {
        "service": "Medical Chatbot",
        "description": "RAG-based medical question answering system",
        "knowledge_base": "Medical_book.pdf",
        "model": "LLaMA 3.1 8B",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")