# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration générale
PIPELINE_STEPS = ["crawler", "pdf_doc_extractor", "embedding"]  # Etapes du pipeline

# Fournisseurs LLM et embeddings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai", "anthropic", "mistral"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")  # "openai", "mistral", "voyage"

# Clés API
API_KEYS = os.getenv("OPENAI_API_KEYS", "")  # Liste de clés séparées par des virgules
if API_KEYS:
    OPENAI_API_KEYS = [key.strip() for key in API_KEYS.split(',')]
else:
    OPENAI_API_KEYS = []

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")

# Modèle et tokens
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "10000"))

# Dossiers input/output
INPUT_DIR = os.getenv("INPUT_DIR", "input")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
CRAWLER_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "crawler_output")
PDF_DOC_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "pdf_doc_extracted")
EMBEDDING_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "embedding_output")

# Paramètres du crawler
START_URL = os.getenv("START_URL", "https://votre-site-exemple.com/fr-ca/")
MAX_DEPTH = int(os.getenv("MAX_DEPTH", "1"))
USE_PLAYWRIGHT = (os.getenv("USE_PLAYWRIGHT", "False").lower() == "true")
DOWNLOAD_PDF = (os.getenv("DOWNLOAD_PDF", "True").lower() == "true")
DOWNLOAD_DOC = (os.getenv("DOWNLOAD_DOC", "True").lower() == "true")
DOWNLOAD_IMAGE = (os.getenv("DOWNLOAD_IMAGE", "False").lower() == "true")
DOWNLOAD_OTHER = (os.getenv("DOWNLOAD_OTHER", "False").lower() == "true")
MAX_URLS = None  # Ex: None pour illimité, ou un entier pour limiter

# Chemin du checkpoint
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")

# Verbosité
VERBOSE = (os.getenv("VERBOSE", "False").lower() == "true")

# LLM Choice pour pdf/doc extractor
PDF_DOC_LLM_PROVIDER = LLM_PROVIDER

# LLM Choice pour embedding
EMBEDDING_LLM_PROVIDER = LLM_PROVIDER

# Embedding provider
EMBEDDING_CHOICE = EMBEDDING_PROVIDER
