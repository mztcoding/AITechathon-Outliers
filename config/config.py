import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone settings
PINECONE_INDEX_NAME = "freight-inspection"
PINECONE_DIMENSION = 768
PINECONE_METRIC = "cosine"

# Gemini model
GEMINI_MODEL = "gemini-2.5-flash"

# RAG settings
TOP_K_RESULTS = 3

# Risk thresholds
RISK_THRESHOLDS = {
    "LOW": (0, 30),
    "MEDIUM": (31, 60),
    "HIGH": (61, 80),
    "CRITICAL": (81, 100),
}

RISK_ACTIONS = {
    "LOW": "ALLOW PASSAGE",
    "MEDIUM": "INSPECT",
    "HIGH": "STOP FOR WEIGHING",
    "CRITICAL": "STOP FOR WEIGHING — IMMEDIATE ACTION REQUIRED",
}
