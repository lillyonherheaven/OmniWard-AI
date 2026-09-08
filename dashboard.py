"""
Smart Visual-Acoustic Patient Monitor - Nurse Station Central Dashboard
A standalone, single-file Streamlit application providing real-time clinical telemetry,
visual-acoustic event monitoring, and triage management for hospital nurse stations.
"""

import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Patient Monitor — Nurse Station",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# 2. CLINICAL COLOR PALETTE & INJECTED CUSTOM CSS
# -----------------------------------------------------------------------------
# Palette specifications:
# - Main Background: #F8FAFC
# - Header & Primary Accents: #2563EB
# - Typography: #1E293B
# - Level 0 (Normal): #10B981
# - Level 1 (Low): #F59E0B
# - Level 2 (Medium): #EA580C
# - Level 3 (Critical): #DC2626
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    /* Global Body & Background */
    html, body, [class*="css"], .stApp {
        background-color: #F8FAFC !important;
        color: #1E293B !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Remove default Streamlit top header spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }

    /* Top Navigation Header Banner */
    .nurse-header-bar {
        background: #2563EB;
        color: #FFFFFF;
        padding: 18px 24px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.18);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }

    .nurse-header-title {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
        color: #FFFFFF;
    }

    .nurse-header-meta {
        display: flex;
        align-items: center;
        gap: 16px;
        font-size: 0.9rem;
        font-weight: 500;
        color: #EFF6FF;
    }

    .shift-tag {
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.35);
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .timestamp-tag {
        font-family: 'JetBrains Mono', monospace;
        background: rgba(15, 23, 42, 0.25);
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
    }

    /* Universal Clean White Cards */
    .med-card {
        background: #FFFFFF;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    .med-card:hover {
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
    }

    /* Triage Indicator Left Borders */
    .triage-border-0 {
        border-left: 6px solid #10B981 !important;
    }
    .triage-border-1 {
        border-left: 6px solid #F59E0B !important;
    }
    .triage-border-2 {
        border-left: 6px solid #EA580C !important;
    }
    .triage-border-3 {
        border-left: 6px solid #DC2626 !important;
    }

    /* Badges */
    .triage-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 3px 8px;
        border-radius: 4px;
    }

    .badge-level-0 {
        background: #ECFDF5;
        color: #10B981;
        border: 1px solid #A7F3D0;
    }
    .badge-level-1 {
        background: #FFFBEB;
        color: #F59E0B;
        border: 1px solid #FDE68A;
    }
    .badge-level-2 {
        background: #FFF7ED;
        color: #EA580C;
        border: 1px solid #FFEDD5;
    }
    .badge-level-3 {
        background: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FECACA;
        animation: pulse-red 2s infinite;
    }

    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.4); }
        70% { box-shadow: 0 0 0 8px rgba(220, 38, 38, 0); }
        100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }
    }

    /* Card Typography */
    .card-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .card-meta {
        font-size: 0.82rem;
        color: #64748B;
        line-height: 1.45;
    }

    .card-param {
        font-weight: 600;
        color: #334155;
    }

    /* Section Column Headers */
    .col-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 14px;
        padding-bottom: 8px;
        border-bottom: 2px solid #E2E8F0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Metric Stat Containers */
    .metric-container {
        display: flex;
        gap: 10px;
        margin-bottom: 16px;
    }

    .metric-card {
        flex: 1;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 14px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        text-align: center;
    }

    .metric-val {
        font-size: 1.7rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }

    .metric-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-top: 4px;
    }

    /* Video Player Box & Overlay */
    .video-viewport {
        position: relative;
        width: 100%;
        background: #0F172A;
        border-radius: 10px;
        border: 2px solid #334155;
        overflow: hidden;
        aspect-ratio: 16 / 9;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    .video-header-overlay {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 14px;
        background: linear-gradient(180deg, rgba(0,0,0,0.8) 0%, rgba(0,0,0,0) 100%);
        z-index: 2;
    }

    .rec-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        color: #EF4444;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }

    .rec-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #EF4444;
        animation: blink 1.2s infinite;
    }

    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.2; }
    }

    .video-simulation-note {
        position: absolute;
        bottom: 12px;
        left: 12px;
        right: 12px;
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(220, 38, 38, 0.4);
        color: #F8FAFC;
        padding: 8px 14px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
        display: flex;
        justify-content: space-between;
        align-items: center;
        z-index: 2;
    }

    /* Action Buttons Custom Styling */
    div.stButton > button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 18px !important;
        font-size: 0.9rem !important;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2) !important;
        transition: all 0.15s ease-in-out !important;
    }

    div.stButton > button:hover {
        background-color: #1D4ED8 !important;
        box-shadow: 0 4px 8px rgba(37, 99, 235, 0.35) !important;
        transform: translateY(-1px);
    }

    /* Zebra Striped Dataframe Container */
    .dataframe-wrapper {
        background: #FFFFFF;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. TOP NAVIGATION BANNER (#2563EB)
# -----------------------------------------------------------------------------
current_time_str = datetime.now().strftime("%Y-%m-%d • %H:%M:%S")

st.markdown(f"""
<div class="nurse-header-bar">
    <div class="nurse-header-title">
        <span>🏥</span> Smart Patient Monitor — Nurse Station
    </div>
    <div class="nurse-header-meta">
        <span class="shift-tag">🌙 Shift: Night</span>
        <span class="timestamp-tag">{current_time_str}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. INITIALIZE SESSION STATE FOR NURSE ACTIONS
# -----------------------------------------------------------------------------
if "alert_acknowledged" not in st.session_state:
    st.session_state.alert_acknowledged = False

if "doctor_called" not in st.session_state:
    st.session_state.doctor_called = False


# -----------------------------------------------------------------------------
# 5. DATA LOADING & FALLBACK LOGIC
# -----------------------------------------------------------------------------
def load_patient_events() -> pd.DataFrame:
    """Loads telemetry events from CSV if present, otherwise provides realistic records."""
    csv_candidates = [
        Path("data/patient_events.csv"),
        Path(__file__).resolve().parent / "data" / "patient_events.csv",
        Path(__file__).resolve().parent.parent / "data" / "patient_events.csv"
    ]

    for candidate in csv_candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            try:
                df = pd.read_csv(candidate)
                if not df.empty and "room_id" in df.columns:
                    return df
            except Exception:
                pass

    # Fallback clinical dataframe
    return pd.DataFrame([
        {
            "timestamp": "2026-09-08 03:54:12",
            "room_id": "Room 204",
            "patient_id": "PAT-8821",
            "patient_name": "Ahmed Hassan",
            "event_type": "scream_distress",
            "decibels": 92.4,
            "torso_angle": 52.0,
            "severity_level": 3,
            "status": "Critical Alert Dispatched"
        },
        {
            "timestamp": "2026-09-08 03:51:04",
            "room_id": "Room 102",
            "patient_id": "PAT-4412",
            "patient_name": "Maria Silva",
            "event_type": "bed_boundary_breach",
            "decibels": 64.0,
            "torso_angle": 36.5,
            "severity_level": 2,
            "status": "Medium Priority Follow-up"
        },
        {
            "timestamp": "2026-09-08 03:48:30",
            "room_id": "Room 105",
            "patient_id": "PAT-9903",
            "patient_name": "David Chen",
            "event_type": "cough_paroxysm",
            "decibels": 73.1,
            "torso_angle": 22.0,
            "severity_level": 1,
            "status": "Logged for Rounds"
        },
        {
            "timestamp": "2026-09-08 03:45:18",
            "room_id": "Room 301",
            "patient_id": "PAT-1189",
            "patient_name": "Eleanor Vance",
            "event_type": "normal_vital_check",
            "decibels": 38.5,
            "torso_angle": 12.0,
            "severity_level": 0,
            "status": "Stable"
        },
        {
            "timestamp": "2026-09-08 03:40:02",
            "room_id": "Room 201",
            "patient_id": "PAT-5531",
            "patient_name": "Arthur Dent",
            "event_type": "routine_posture_shift",
            "decibels": 41.2,
            "torso_angle": 15.0,
            "severity_level": 0,
            "status": "Stable"
        },
        {
            "timestamp": "2026-09-08 03:32:45",
            "room_id": "Room 108",
            "patient_id": "PAT-7720",
            "patient_name": "Fatima Al-Sayed",
            "event_type": "restless_movement",
            "decibels": 52.8,
            "torso_angle": 24.5,
            "severity_level": 1,
            "status": "Logged"
        },
        {
            "timestamp": "2026-09-08 03:25:10",
            "room_id": "Room 305",
            "patient_id": "PAT-6619",
            "patient_name": "John Watson",
            "event_type": "normal_rest",
            "decibels": 35.0,
            "torso_angle": 9.5,
            "severity_level": 0,
            "status": "Stable"
        }
    ])

df_events_raw = load_patient_events()


# -----------------------------------------------------------------------------
# 6. MAIN 3-COLUMN CLINICAL LAYOUT ARCHITECTURE
# -----------------------------------------------------------------------------
col_events, col_live, col_analytics = st.columns([1.0, 1.35, 1.25], gap="large")


# =============================================================================
# COLUMN 1: REAL-TIME EVENT STREAM (col_events)
# =============================================================================
with col_events:
    st.markdown('<div class="col-header"><span>⚡</span> Real-Time Event Stream</div>', unsafe_allow_html=True)

    # 1. Crimson Red Card (#DC2626) - Critical Level 3
    st.markdown("""
    <div class="med-card triage-border-3">
        <div class="card-title">
            <span>ROOM 204: Screaming Detected</span>
            <span class="triage-badge badge-level-3">🚨 Level 3 Critical</span>
        </div>
        <div class="card-meta">
            <div>Patient: <span class="card-param">Ahmed Hassan (PAT-8821)</span></div>
            <div>Acoustic Energy: <span class="card-param">92.4 dB SPL</span> (Acute Scream)</div>
            <div>Torso Tilt: <span class="card-param">52.0°</span> (Abrupt Fall Posture)</div>
            <div style="margin-top: 4px; font-size: 0.76rem; color: #DC2626; font-weight: 600;">
                ● Immediate Nurse Intervention Required
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Coral Orange Card (#EA580C) - Medium Level 2
    st.markdown("""
    <div class="med-card triage-border-2">
        <div class="card-title">
            <span>ROOM 102: Bed Boundary Crossed</span>
            <span class="triage-badge badge-level-2">⚠️ Level 2 Medium</span>
        </div>
        <div class="card-meta">
            <div>Patient: <span class="card-param">Maria Silva (PAT-4412)</span></div>
            <div>Perimeter Status: <span class="card-param">Right Bed Rail Breached</span></div>
            <div>Torso Tilt: <span class="card-param">36.5°</span> (Fall Hazard Risk)</div>
            <div style="margin-top: 4px; font-size: 0.76rem; color: #EA580C; font-weight: 600;">
                ● Repositioning Assistance Prompted
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3. Sage Green Card (#10B981) - Normal Level 0
    st.markdown("""
    <div class="med-card triage-border-0">
        <div class="card-title">
            <span>ROOM 301: Normal Vital Check</span>
            <span class="triage-badge badge-level-0">✓ Level 0 Normal</span>
        </div>
        <div class="card-meta">
            <div>Patient: <span class="card-param">Eleanor Vance (PAT-1189)</span></div>
            <div>Ambient Sound: <span class="card-param">38.5 dB</span> (Resting Ambient)</div>
            <div>Position: <span class="card-param">12.0° Supine</span> • Safe Zone Centered</div>
            <div style="margin-top: 4px; font-size: 0.76rem; color: #10B981; font-weight: 600;">
                ● Vitals Consistent & Steady
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Supplementary Warm Amber Card (#F59E0B) for complete 4-tier triage illustration
    st.markdown("""
    <div class="med-card triage-border-1">
        <div class="card-title">
            <span>ROOM 105: Cough Paroxysm</span>
            <span class="triage-badge badge-level-1">⚡ Level 1 Low</span>
        </div>
        <div class="card-meta">
            <div>Patient: <span class="card-param">David Chen (PAT-9903)</span></div>
            <div>Acoustic Energy: <span class="card-param">73.1 dB</span> (Repetitive Cough)</div>
            <div>Torso Angle: <span class="card-param">22.0°</span> (Elevated Fowler)</div>
            <div style="margin-top: 4px; font-size: 0.76rem; color: #D97706; font-weight: 600;">
                ● Telemetry Logged for Scheduled Rounds
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# COLUMN 2: MAIN FOCUS AREA & LIVE VIDEO (col_live)
# =============================================================================
with col_live:
    st.markdown('<div class="col-header"><span>📹</span> Main Focus Surveillance — Room 204</div>', unsafe_allow_html=True)

    # Large Video Feed Placeholder with MediaPipe Skeleton Overlay simulation
    st.markdown("""
    <div class="video-viewport">
        <!-- Top Overlay Header -->
        <div class="video-header-overlay">
            <span class="rec-badge">
                <span class="rec-dot"></span> LIVE 30 FPS • ROOM 204 (ICU Alpha)
            </span>
            <span style="color: #94A3B8; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace;">
                CAM-04 • 1080p HD
            </span>
        </div>

        <!-- Simulated Visual Canvas: Hospital Bed & Pose Skeleton Overlay -->
        <div style="flex: 1; position: relative; display: flex; align-items: center; justify-content: center; width: 100%;">
            <svg viewBox="0 0 640 360" style="width: 100%; height: 100%;" xmlns="http://www.w3.org/2000/svg">
                <!-- Room Background Dimming -->
                <rect width="640" height="360" fill="#0B1329" />
                
                <!-- Floor grid lines -->
                <line x1="0" y1="300" x2="640" y2="300" stroke="#1E293B" stroke-width="1" />
                <line x1="160" y1="300" x2="80" y2="360" stroke="#1E293B" stroke-width="1" />
                <line x1="480" y1="300" x2="560" y2="360" stroke="#1E293B" stroke-width="1" />
                
                <!-- Hospital Bed Frame -->
                <rect x="130" y="150" width="380" height="150" rx="10" fill="#1E293B" stroke="#334155" stroke-width="2" />
                <rect x="145" y="165" width="350" height="120" rx="6" fill="#334155" opacity="0.6" />
                
                <!-- Bed Safety Boundary Perimeter Box (Virtual Perimeter) -->
                <rect x="120" y="90" width="400" height="230" fill="none" stroke="#DC2626" stroke-width="2" stroke-dasharray="6,4" />
                <text x="130" y="112" fill="#DC2626" font-size="11" font-weight="700" font-family="Inter, sans-serif">
                    ⚠️ VIRTUAL BED SAFETY BOUNDARY (BREACHED)
                </text>

                <!-- MediaPipe Pose Skeleton Connections (Simulated Fall Angle: 52°) -->
                <!-- Spine / Torso line tilted at 52 deg -->
                <line x1="320" y1="130" x2="260" y2="230" stroke="#00F0FF" stroke-width="3" />
                <!-- Shoulders -->
                <line x1="280" y1="145" x2="360" y2="120" stroke="#00F0FF" stroke-width="3" />
                <!-- Left Arm -->
                <line x1="280" y1="145" x2="230" y2="190" stroke="#00FF66" stroke-width="2.5" />
                <line x1="230" y1="190" x2="190" y2="240" stroke="#00FF66" stroke-width="2.5" />
                <!-- Right Arm -->
                <line x1="360" y1="120" x2="410" y2="160" stroke="#00FF66" stroke-width="2.5" />
                <line x1="410" y1="160" x2="440" y2="210" stroke="#00FF66" stroke-width="2.5" />
                <!-- Hips -->
                <line x1="240" y1="235" x2="280" y2="225" stroke="#00F0FF" stroke-width="3" />
                <!-- Legs -->
                <line x1="240" y1="235" x2="190" y2="280" stroke="#FFCC00" stroke-width="2.5" />
                <line x1="280" y1="225" x2="230" y2="295" stroke="#FFCC00" stroke-width="2.5" />

                <!-- Landmarks Joints (Glowing keypoints) -->
                <!-- Head / Nose -->
                <circle cx="340" cy="100" r="14" fill="#E2E8F0" stroke="#00F0FF" stroke-width="2" />
                <circle cx="340" cy="100" r="4" fill="#00F0FF" />
                <!-- Shoulder Joints -->
                <circle cx="280" cy="145" r="5" fill="#00FF66" />
                <circle cx="360" cy="120" r="5" fill="#00FF66" />
                <!-- Elbows & Wrists -->
                <circle cx="230" cy="190" r="4" fill="#00FF66" />
                <circle cx="190" cy="240" r="4" fill="#DC2626" />
                <circle cx="410" cy="160" r="4" fill="#00FF66" />
                <circle cx="440" cy="210" r="4" fill="#00FF66" />
                <!-- Hips -->
                <circle cx="240" cy="235" r="5" fill="#00F0FF" />
                <circle cx="280" cy="225" r="5" fill="#00F0FF" />
                <!-- Knees -->
                <circle cx="190" cy="280" r="4" fill="#FFCC00" />
                <circle cx="230" cy="295" r="4" fill="#FFCC00" />

                <!-- Torso Angle Vector Arc Annotation -->
                <path d="M 260 230 L 260 150" stroke="#94A3B8" stroke-width="1" stroke-dasharray="3,3" />
                <path d="M 260 170 A 60 60 0 0 1 300 162" fill="none" stroke="#DC2626" stroke-width="2" />
                <rect x="272" y="166" width="60" height="20" rx="3" fill="#DC2626" />
                <text x="278" y="180" fill="#FFFFFF" font-size="11" font-weight="700" font-family="JetBrains Mono, monospace">
                    52.0°
                </text>
            </svg>
        </div>

        <!-- Overlay Simulation Note -->
        <div class="video-simulation-note">
            <span>🦴 MediaPipe Skeleton Overlay Active (Torso Angle: 52°)</span>
            <span style="color: #F87171; font-weight: 700;">● FALL DETECTED</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Patient Details Card Below Video
    st.markdown("""
    <div class="med-card" style="margin-top: 14px; border-left: 6px solid #DC2626;">
        <div class="card-title">
            <span>👤 Patient Record: Ahmed Hassan</span>
            <span class="triage-badge badge-level-3">Code Blue Escalation</span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 8px; font-size: 0.85rem;">
            <div>
                <span style="color: #64748B;">Patient ID:</span><br>
                <strong style="color: #0F172A;">PAT-8821</strong>
            </div>
            <div>
                <span style="color: #64748B;">Room / Bed:</span><br>
                <strong style="color: #0F172A;">Room 204 • Bed A</strong>
            </div>
            <div>
                <span style="color: #64748B;">Age & Gender:</span><br>
                <strong style="color: #0F172A;">68 Y / Male</strong>
            </div>
            <div>
                <span style="color: #64748B;">Admission Reason:</span><br>
                <strong style="color: #0F172A;">Post-Op Hip Rehab</strong>
            </div>
            <div>
                <span style="color: #64748B;">Assigned Nurse:</span><br>
                <strong style="color: #0F172A;">Nurse Sarah J., RN</strong>
            </div>
            <div>
                <span style="color: #64748B;">Risk Evaluation:</span><br>
                <strong style="color: #DC2626;">Severe Fall & Distress</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Action Buttons Row
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("✓ Acknowledge Alert", use_container_width=True):
            st.session_state.alert_acknowledged = True
            st.toast("Alert for Room 204 Acknowledged by Night Shift Staff", icon="✅")

    with btn_col2:
        if st.button("📞 Call On-Duty Doctor", use_container_width=True):
            st.session_state.doctor_called = True
            st.toast("Emergency Page sent to Dr. Robert Vance (On-Duty Physician)", icon="🚨")

    # Feedback indicators if actions taken
    if st.session_state.alert_acknowledged:
        st.success("✓ Alert Acknowledged — Nurse En Route to Room 204", icon="🩺")
    if st.session_state.doctor_called:
        st.info("🚨 On-Duty Physician Contacted via Vocera Emergency Dispatch (Room 204)", icon="📟")


# =============================================================================
# COLUMN 3: OVERALL ANALYTICS & DATA LOGS (col_analytics)
# =============================================================================
with col_analytics:
    st.markdown('<div class="col-header"><span>📊</span> Unit Analytics & Telemetry Logs</div>', unsafe_allow_html=True)

    # Summary Metric Cards: Total active rooms (12), Stable (10), Needs attention (2)
    st.markdown("""
    <div class="metric-container">
        <div class="metric-card">
            <div class="metric-val" style="color: #2563EB;">12</div>
            <div class="metric-label">Active Rooms</div>
        </div>
        <div class="metric-card">
            <div class="metric-val" style="color: #10B981;">10</div>
            <div class="metric-label">Stable</div>
        </div>
        <div class="metric-card">
            <div class="metric-val" style="color: #DC2626;">2</div>
            <div class="metric-label">Needs Attention</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Filtering Section
    st.markdown('<div style="font-weight: 700; font-size: 0.88rem; color: #334155; margin-bottom: 8px;">Filter Telemetry Logs:</div>', unsafe_allow_html=True)
    filter_c1, filter_c2 = st.columns(2)

    with filter_c1:
        unique_rooms = ["All Rooms"] + sorted(list(df_events_raw["room_id"].unique()))
        selected_room = st.selectbox("Select Room:", unique_rooms, index=0)

    with filter_c2:
        severity_options = [
            "All Levels",
            "Level 0 (Normal)",
            "Level 1 (Low)",
            "Level 2 (Medium)",
            "Level 3 (Critical)"
        ]
        selected_severity = st.selectbox("Severity Level:", severity_options, index=0)

    # Filter Logic
    filtered_df = df_events_raw.copy()

    if selected_room != "All Rooms":
        filtered_df = filtered_df[filtered_df["room_id"] == selected_room]

    if selected_severity != "All Levels":
        target_lvl = int(selected_severity.split(" ")[1])
        filtered_df = filtered_df[filtered_df["severity_level"] == target_lvl]

    # Display Styled Event History Table
    st.markdown('<div style="font-weight: 700; font-size: 0.85rem; color: #475569; margin-top: 10px; margin-bottom: 4px;">Recent Incident History:</div>', unsafe_allow_html=True)

    columns_to_show = ["timestamp", "room_id", "patient_name", "event_type", "torso_angle", "decibels", "severity_level", "status"]
    available_cols = [c for c in columns_to_show if c in filtered_df.columns]

    # Format dataframe for display
    display_df = filtered_df[available_cols].rename(columns={
        "timestamp": "Timestamp",
        "room_id": "Room",
        "patient_name": "Patient",
        "event_type": "Event Detected",
        "torso_angle": "Angle (°)",
        "decibels": "Sound (dB)",
        "severity_level": "Level",
        "status": "Triage Action"
    })

    if not display_df.empty:
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=260
        )
    else:
        st.info("No clinical incidents found matching the selected filters.", icon="ℹ️")

    # CSV Log Download Button
    csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Telemetry Log (.CSV)",
        data=csv_bytes,
        file_name=f"patient_telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
