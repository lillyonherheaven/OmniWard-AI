"""
Smart Visual-Acoustic Patient Monitor - FastAPI Server Application
Main entry point for the REST API backend. Configures CORS, lifecycle model loading, and routes.
"""

import os
import sys
import warnings

# Suppress environment and library warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_LOG_LEVEL"] = "error"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import logging
from pathlib import Path
from contextlib import asynccontextmanager
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from settings import MODEL_PATH, FASTAPI_HOST, FASTAPI_PORT
from routes import router as api_router, set_loaded_model
from build_triage_model import train_and_export_triage_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PATIENT MONITOR API] %(message)s"
)
logger = logging.getLogger("patient_monitor_main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.
    Loads triage_model.pkl on startup using joblib.load().
    If the model is not found, automatically trains and persists it.
    """
    logger.info("Initializing Smart Visual-Acoustic Patient Monitor Backend...")
    
    loaded_model = None
    if MODEL_PATH.exists():
        try:
            logger.info(f"Loading triage model from {MODEL_PATH}...")
            loaded_model = joblib.load(MODEL_PATH)
            logger.info("✓ Triage model successfully loaded into memory.")
        except Exception as e:
            logger.error(f"Failed to load existing triage model: {e}")
    
    if loaded_model is None:
        logger.warning(f"Triage model not found at {MODEL_PATH}. Initiating automatic training pipeline...")
        try:
            loaded_model = train_and_export_triage_model(MODEL_PATH)
            logger.info("✓ Triage model trained and loaded successfully.")
        except Exception as e:
            logger.critical(f"Critical error training triage model: {e}")

    # Inject model into routing subsystem
    set_loaded_model(loaded_model)
    logger.info("✓ Telemetry API routes ready for patient monitoring.")

    yield

    logger.info("Shutting down Patient Monitor Backend service.")


app = FastAPI(
    title="Smart Visual-Acoustic Patient Monitor API",
    description="Multimodal real-time patient posture, bed safety boundary, and acoustic triage API.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)


@app.get("/", tags=["Root"])
def read_root():
    return {
        "system": "Smart Visual-Acoustic Patient Monitor",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/api/v1/health",
        "endpoints": {
            "process_frame": "/api/v1/process-frame [POST]",
            "patient_logs": "/api/v1/patient-logs [GET]",
            "health": "/api/v1/health [GET]"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=FASTAPI_HOST,
        port=FASTAPI_PORT,
        reload=False
    )
