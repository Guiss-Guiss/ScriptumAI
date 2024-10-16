import os
import torch
from pathlib import Path
import logging

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
INDEX_DIR = DATA_DIR / 'index'

# Ensure data directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama API endpoint

# Embedding model configuration
EMBEDDING_MODEL = "llama3.2"  # or another model that's available in your Ollama installation
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Added as per instructions

# LLM configuration for query processing
LLM_MODEL = "llama3.2:latest"  # Llama3.2 model in Ollama
LLM_MAX_TOKENS = 2048  # Maximum number of tokens for LLM response

# Vector store configuration
VECTOR_STORE_TYPE = "chroma"  # Using Chroma as per the source code
CHROMA_PERSIST_DIRECTORY = INDEX_DIR / "chroma"
CHROMA_COLLECTION_NAME = "buildragwithpython"

# Retrieval configuration
TOP_K_RESULTS = 5

# Ingestion configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Frontend configuration
FRONTEND_HOST = "localhost"
FRONTEND_PORT = 8501

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(BASE_DIR, "rag_app.log")

# Configure logging
logging.basicConfig(filename=LOG_FILE, level=LOG_LEVEL,
                    format='%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Flask API configuration (if you decide to use Flask for backend API)
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5001
FLASK_DEBUG = False

# Maximum file size for upload (in bytes)
MAX_UPLOAD_SIZE = 200 * 1024 * 1024  # 200 MB

# Supported file types for document ingestion
SUPPORTED_FILE_TYPES = [
    'application/pdf',
    'text/plain',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    
]

# Performance tuning
MAX_CONCURRENT_REQUESTS = 10
BATCH_SIZE = 32  # For batch processing in embedding and retrieval

# Error handling and retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Security settings
SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key-here")  # INPUT_REQUIRED {config_description}

# Ollama models to ensure are pulled
REQUIRED_OLLAMA_MODELS = [
    EMBEDDING_MODEL,
    LLM_MODEL,
]

# File upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'temp_uploads')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'html', 'md'}

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Add the missing API_BASE_URL configuration
API_BASE_URL = "http://localhost:5001" 

# Supported languages configuration
SUPPORTED_LANGUAGES = ['en', 'fr', 'es']
