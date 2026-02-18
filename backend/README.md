# 🏥 Medical Chatbot

A RAG (Retrieval-Augmented Generation) based medical question answering system that uses your medical knowledge base PDF to provide accurate medical information.

## Features

- **Knowledge Base**: Uses `Medical_book.pdf` as the medical knowledge source
- **Embedding**: Generates embeddings using Hugging Face sentence transformers
- **Vector Storage**: Stores embeddings in FAISS vector database
- **Retrieval**: Retrieves relevant medical context based on user queries
- **LLM Integration**: Uses Groq's LLaMA 3.1 8B model for response generation
- **Multiple Interfaces**: Command-line interface and web API
- **Caching**: Intelligent caching of vector indices for faster subsequent queries

## Tech Stack

- **LangChain**: Framework for building LLM applications
- **Hugging Face**: Embedding models and transformers
- **FAISS**: Vector similarity search
- **Groq**: Fast inference for LLMs
- **FastAPI**: Web API framework
- **Python**: Core programming language

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your `.env` file contains the required API keys:
```
GROQ_API_KEY="your_groq_api_key"
HUGGINGFACEHUB_API_TOKEN="your_huggingface_token"
```

## Usage

### Option 1: Simple Command Line Interface (Recommended)
```bash
python medical_chatbot_simple.py
```

### Option 2: Original Command Line Interface
```bash
python medical_chatbot.py
```

### Option 3: Web API
```bash
python api.py
```

Then open `web_interface.html` in your browser to use the web interface.

### Option 4: Using Batch Files (Windows)
- Double-click `run_chatbot.bat` for original CLI version
- Double-click `run_api.bat` for web API version

## API Endpoints

When running the API server:

- `GET /` - Health check
- `GET /health` - Service status
- `GET /info` - System information
- `POST /ask` - Ask medical questions

Example API usage:
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the symptoms of diabetes?"}'
```

## How It Works

1. **Document Processing**: Loads and splits the medical PDF into chunks
2. **Embedding Generation**: Creates vector embeddings using sentence transformers
3. **Index Building**: Stores embeddings in FAISS for efficient similarity search
4. **Query Processing**: Takes user questions and retrieves relevant medical context
5. **Response Generation**: Uses LLM to generate structured medical responses

## Project Structure

```
Medical_chatbot/
├── medical_chatbot.py      # Main CLI application
├── api.py                  # FastAPI web server
├── web_interface.html      # Web frontend
├── requirements.txt        # Python dependencies
├── run_chatbot.bat         # Windows CLI launcher
├── run_api.bat             # Windows API launcher
├── .env                   # API keys (already configured)
├── Medical_book.pdf       # Your medical knowledge base
└── .indices/              # Generated vector indices (created automatically)
```

## Notes

- **First run will take 2-5 minutes** as it processes the PDF, generates embeddings, and builds the vector index
- Subsequent runs use cached indices for faster response (few seconds)
- The system only uses information from your Medical_book.pdf
- Responses are formatted with medical structure and disclaimers
- For best performance, use `medical_chatbot_simple.py` which pre-loads the knowledge base

## Troubleshooting

If you encounter TensorFlow warnings, they can be ignored - the system will work correctly.
If the API times out on first request, wait for indexing to complete (check terminal logs).