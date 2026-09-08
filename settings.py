"""
Smart Visual-Acoustic Patient Monitor - Global Configuration Settings
"""

import os
import warnings
from pathlib import Path

# -----------------------------------------------------------------------------
# GLOBAL WARNINGS & C++ LOGGING SUPPRESSION
# Must be configured prior to loading machine learning / vision frameworks
# -----------------------------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_LOG_LEVEL"] = "error"
os.environ["PYTHONWARNINGS"] = "ignore"

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
CSV_PATH = DATA_DIR / "patient_events.csv"
MODEL_PATH = SAVED_MODELS_DIR / "triage_model.pkl"
if not MODEL_PATH.exists() and (MODELS_DIR / "triage_model.pkl").exists():
    MODEL_PATH = MODELS_DIR / "triage_model.pkl"

# Camera and Vision Configuration
# Use integer index (0) or string path/URL for video streaming/testing
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
try:
    CAMERA_INDEX = int(CAMERA_SOURCE)
except ValueError:
    CAMERA_INDEX = CAMERA_SOURCE

FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
STREAM_FPS = int(os.getenv("STREAM_FPS", "30"))

# Bed Safety Area Boundaries (Normalized coordinates 0.0 - 1.0)
# Patient is safely confined within this virtual bounding box
BED_SAFETY_BOUNDS = {
    "x_min": float(os.getenv("BED_X_MIN", "0.20")),  # 20% from left
    "y_min": float(os.getenv("BED_Y_MIN", "0.20")),  # 20% from top
    "x_max": float(os.getenv("BED_X_MAX", "0.80")),  # 80% from left
    "y_max": float(os.getenv("BED_Y_MAX", "0.85")),  # 85% from top
}

# Pose Angle Thresholds for Fall Risk
# Torso angle relative to vertical axis (> 45° indicates horizontal/fallen or severe lean)
TORSO_ANGLE_FALL_THRESHOLD = float(os.getenv("TORSO_FALL_THRESHOLD", "45.0"))
TORSO_ANGLE_WARNING_THRESHOLD = float(os.getenv("TORSO_WARN_THRESHOLD", "30.0"))

# Audio Energy & Decibel Thresholds
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_SIZE = 1024
DECIBEL_SCREAM_THRESHOLD = float(os.getenv("DECIBEL_SCREAM_THRESHOLD", "85.0"))
DECIBEL_COUGH_THRESHOLD = float(os.getenv("DECIBEL_COUGH_THRESHOLD", "70.0"))
DECIBEL_GROAN_THRESHOLD = float(os.getenv("DECIBEL_GROAN_THRESHOLD", "58.0"))
DECIBEL_AMBIENT_BASELINE = float(os.getenv("DECIBEL_AMBIENT", "42.0"))

# Triage Severity Levels
# Level 0: Safe (Normal vitals & posture)
# Level 1: Low (Minor posture tilt or gentle cough)
# Level 2: Medium (Bed boundary violation or continuous cough/groan)
# Level 3: Critical (Fall detected, scream sound, or acute posture collapse)
LEVEL_0_SAFE = 0
LEVEL_1_LOW = 1
LEVEL_2_MEDIUM = 2
LEVEL_3_CRITICAL = 3

TRIAGE_SEVERITY_NAMES = {
    LEVEL_0_SAFE: "Safe",
    LEVEL_1_LOW: "Low",
    LEVEL_2_MEDIUM: "Medium",
    LEVEL_3_CRITICAL: "Critical",
}

TRIAGE_SEVERITY_COLORS = {
    LEVEL_0_SAFE: "#10b981",    # Emerald green
    LEVEL_1_LOW: "#0ea5e9",     # Sky blue
    LEVEL_2_MEDIUM: "#f59e0b",  # Amber orange
    LEVEL_3_CRITICAL: "#ef4444" # Crimson red
}

# Backend API Configuration
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "0.0.0.0")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))
BACKEND_API_URL = os.getenv("BACKEND_API_URL", f"http://127.0.0.1:{FASTAPI_PORT}")

# Streamlit Frontend Configuration
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# Notification Service Simulation Config
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "AC_MOCK_DEMO_SID_123456789")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "MOCK_AUTH_TOKEN_987654321")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "+18005550199")
EMERGENCY_DESK_PHONE = os.getenv("EMERGENCY_DESK_PHONE", "+18005550100")
PUSH_NOTIFICATION_WEBHOOK = os.getenv("PUSH_NOTIFICATION_WEBHOOK", "https://hospital.emergency.internal/api/v1/webhook")

# CSV Columns Schema
CSV_COLUMNS = [
    "timestamp",
    "room_id",
    "patient_id",
    "event_type",
    "decibels",
    "torso_angle",
    "severity_level",
    "status"
]
