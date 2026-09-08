"""
Smart Visual-Acoustic Patient Monitor - FastAPI Route Definitions
Endpoints for frame/audio telemetry processing, ML triage evaluation, and patient audit logs.
"""

import csv
import os
import threading
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Suppress runtime warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Query, status

from settings import (
    CSV_PATH,
    CSV_COLUMNS,
    TRIAGE_SEVERITY_NAMES,
    LEVEL_3_CRITICAL,
    TORSO_ANGLE_FALL_THRESHOLD,
    DECIBEL_SCREAM_THRESHOLD,
)
from notification_service import NotificationService

router = APIRouter(prefix="/api/v1", tags=["Patient Monitoring"])
notification_service = NotificationService()
csv_lock = threading.Lock()

# Global model reference (injected by main.py on startup)
ml_model = None


def set_loaded_model(model_instance):
    """Sets the globally cached scikit-learn triage model."""
    global ml_model
    ml_model = model_instance


class FrameMetricsRequest(BaseModel):
    room_id: str = Field(..., example="ICU-101", description="Patient Ward/Room identifier")
    patient_id: str = Field(..., example="PAT-8821", description="Unique Patient identifier")
    torso_angle: float = Field(..., ge=0.0, le=180.0, example=24.5, description="Torso inclination in degrees")
    bed_boundary_violation: int = Field(0, ge=0, le=1, example=0, description="1 if safety boundary crossed, else 0")
    audio_event: str = Field("none", example="cough", description="Acoustic signature: none, cough, groan, or scream")
    audio_decibel: float = Field(45.0, ge=20.0, le=140.0, example=71.2, description="Auditory sound level in dB")
    notes: Optional[str] = Field(None, description="Optional telemetry remarks or tags")


class TriageEvaluationResponse(BaseModel):
    timestamp: str
    room_id: str
    patient_id: str
    event_type: str
    decibels: float
    torso_angle: float
    bed_boundary_violation: int
    severity_level: int
    severity_name: str
    status: str
    critical_alert_triggered: bool
    alert_details: Optional[Dict[str, Any]] = None


def ensure_csv_initialized():
    """Guarantees patient_events.csv exists with correct header row."""
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)


def append_event_to_csv(
    timestamp: str,
    room_id: str,
    patient_id: str,
    event_type: str,
    decibels: float,
    torso_angle: float,
    severity_level: int,
    status_label: str
):
    """Safely appends a telemetry event record to CSV with thread-locking."""
    ensure_csv_initialized()
    with csv_lock:
        with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                room_id,
                patient_id,
                event_type,
                f"{decibels:.1f}",
                f"{torso_angle:.1f}",
                severity_level,
                status_label
            ])


def evaluate_triage_fallback(
    audio_event: str,
    audio_decibel: float,
    torso_angle: float,
    bed_violation: int
) -> int:
    """Deterministic clinical rule fallback if ML artifact is uninitialized."""
    # Critical criteria
    if torso_angle >= TORSO_ANGLE_FALL_THRESHOLD or audio_event == "scream" or audio_decibel >= DECIBEL_SCREAM_THRESHOLD:
        return 3
    # Medium risk criteria
    if bed_violation == 1 or audio_event == "groan" or torso_angle >= 30.0:
        return 2
    # Low risk criteria
    if audio_event == "cough" or torso_angle >= 20.0 or audio_decibel >= 65.0:
        return 1
    return 0


@router.post(
    "/process-frame",
    response_model=TriageEvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate patient frame and audio telemetry metrics"
)
def process_frame(payload: FrameMetricsRequest):
    """
    Ingests visual and acoustic telemetry metrics:
      1. Evaluates severity level using the trained Random Forest triage model.
      2. Categorizes event signature.
      3. Automatically logs the observation into patient_events.csv.
      4. Triggers Level 3 Code Blue emergency notifications if severity == 3.
    """
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # Determine event type
    event_type_components = []
    if payload.torso_angle >= TORSO_ANGLE_FALL_THRESHOLD:
        event_type_components.append("fall_detected")
    elif payload.bed_boundary_violation == 1:
        event_type_components.append("bed_boundary_violation")
    elif payload.torso_angle >= 25.0:
        event_type_components.append("posture_tilt")

    if payload.audio_event in ["scream", "cough", "groan"]:
        event_type_components.append(payload.audio_event)

    event_type = "_".join(event_type_components) if event_type_components else "normal_rest"

    # Evaluate ML Model
    severity_level = None
    if ml_model is not None:
        try:
            # Create feature DataFrame strictly matching training column names and types
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                features_df = pd.DataFrame([{
                    "audio_event": str(payload.audio_event if payload.audio_event in ["none", "cough", "groan", "scream"] else "none"),
                    "audio_decibel": float(payload.audio_decibel),
                    "torso_angle": float(payload.torso_angle),
                    "bed_boundary_violation": int(payload.bed_boundary_violation)
                }], columns=["audio_event", "audio_decibel", "torso_angle", "bed_boundary_violation"])
                prediction = ml_model.predict(features_df)
                severity_level = int(prediction[0])
        except Exception as err:
            severity_level = evaluate_triage_fallback(
                payload.audio_event,
                payload.audio_decibel,
                payload.torso_angle,
                payload.bed_boundary_violation
            )
    else:
        severity_level = evaluate_triage_fallback(
            payload.audio_event,
            payload.audio_decibel,
            payload.torso_angle,
            payload.bed_boundary_violation
        )

    # Escalation status & notifications
    critical_alert_triggered = False
    alert_details = None
    if severity_level == LEVEL_3_CRITICAL:
        critical_alert_triggered = True
        status_label = "Critical Alert Dispatched"
        alert_details = notification_service.send_critical_alert(
            room_id=payload.room_id,
            patient_id=payload.patient_id,
            event_type=event_type,
            details={
                "torso_angle": payload.torso_angle,
                "decibels": payload.audio_decibel,
                "audio_event": payload.audio_event,
                "boundary_breach": payload.bed_boundary_violation
            }
        )
    elif severity_level == 2:
        status_label = "Urgent Attention"
    elif severity_level == 1:
        status_label = "Advisory"
    else:
        status_label = "Logged"

    # Append to durable patient log CSV
    append_event_to_csv(
        timestamp=timestamp,
        room_id=payload.room_id,
        patient_id=payload.patient_id,
        event_type=event_type,
        decibels=payload.audio_decibel,
        torso_angle=payload.torso_angle,
        severity_level=severity_level,
        status_label=status_label
    )

    return TriageEvaluationResponse(
        timestamp=timestamp,
        room_id=payload.room_id,
        patient_id=payload.patient_id,
        event_type=event_type,
        decibels=payload.audio_decibel,
        torso_angle=payload.torso_angle,
        bed_boundary_violation=payload.bed_boundary_violation,
        severity_level=severity_level,
        severity_name=TRIAGE_SEVERITY_NAMES.get(severity_level, "Unknown"),
        status=status_label,
        critical_alert_triggered=critical_alert_triggered,
        alert_details=alert_details
    )


@router.get("/patient-logs", summary="Fetch patient monitoring audit records from CSV")
def get_patient_logs(
    limit: int = Query(50, ge=1, le=1000, description="Max records to return"),
    room_id: Optional[str] = Query(None, description="Filter by Room ID"),
    severity_level: Optional[int] = Query(None, ge=0, le=3, description="Filter by triage level")
):
    """Reads patient events from patient_events.csv and returns structured JSON."""
    ensure_csv_initialized()
    try:
        with csv_lock:
            df = pd.read_csv(CSV_PATH)

        if df.empty:
            return {"total": 0, "records": []}

        # Apply optional filters
        if isinstance(room_id, str) and room_id:
            df = df[df["room_id"] == room_id]
        if isinstance(severity_level, int):
            df = df[df["severity_level"] == severity_level]

        total_matching = len(df)
        limit_val = limit if isinstance(limit, int) else 50
        # Tail most recent records
        df_recent = df.tail(limit_val).iloc[::-1]

        records = df_recent.to_dict(orient="records")
        return {
            "total": total_matching,
            "returned": len(records),
            "records": records
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read patient logs: {str(e)}"
        )


@router.get("/health", summary="Service health and ML model status")
def health_check():
    """Health status and telemetry subsystem diagnostics."""
    ensure_csv_initialized()
    try:
        with csv_lock:
            log_count = sum(1 for _ in open(CSV_PATH, "r", encoding="utf-8")) - 1
            log_count = max(0, log_count)
    except Exception:
        log_count = 0

    return {
        "status": "healthy",
        "service": "Smart Visual-Acoustic Patient Monitor API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": ml_model is not None,
        "csv_logging": {
            "path": str(CSV_PATH),
            "event_count": log_count
        },
        "critical_alerts_sent": len(notification_service.dispatch_history)
    }
