import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# -----------------------------------------------------------------------------
# 1. STREAMLIT CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="OmniWard AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# 2. CLINICAL NEUTRAL PALETTE & SAFE HTML RENDERING GUARDRAIL
# -----------------------------------------------------------------------------
def render_html(html_str: str):
    """
    Guarantees zero code leakage and clean HTML/SVG rendering in Streamlit.
    Strips leading line indentation to prevent CommonMark from misinterpreting HTML as 4-space indented code blocks.
    Prefers st.html() when available (Streamlit 1.33+), falling back to st.markdown(..., unsafe_allow_html=True).
    """
    clean_html = "\n".join(line.strip() for line in html_str.strip().splitlines() if line.strip())
    if hasattr(st, "html"):
        st.html(clean_html)
    else:
        st.markdown(clean_html, unsafe_allow_html=True)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"], .stApp {
    background-color: #F8FAFC !important;
    color: #1E293B !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 2rem !important;
    padding-left: 1.8rem !important;
    padding-right: 1.8rem !important;
    max-width: 100% !important;
}

/* Top Navigation Banner (#2563EB) */
.nurse-nav-bar {
    background: #2563EB;
    color: #FFFFFF;
    padding: 16px 24px;
    border-radius: 12px;
    margin-bottom: 22px;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.20);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}

.nurse-nav-title {
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 10px;
    color: #FFFFFF;
}

.nurse-nav-meta {
    display: flex;
    align-items: center;
    gap: 14px;
    font-size: 0.9rem;
    font-weight: 500;
    color: #EFF6FF;
}

.shift-pill {
    background: rgba(255, 255, 255, 0.18);
    border: 1px solid rgba(255, 255, 255, 0.35);
    padding: 4px 12px;
    border-radius: 9999px;
    font-weight: 600;
}

.time-pill {
    font-family: 'JetBrains Mono', monospace;
    background: rgba(15, 23, 42, 0.25);
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 0.85rem;
}

/* Clinical Cards */
.med-card {
    background: #FFFFFF;
    border-radius: 10px;
    border: 1px solid #E2E8F0;
    padding: 15px 18px;
    margin-bottom: 14px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.med-card:hover {
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
}

/* Severity Left Borders */
.triage-border-0 { border-left: 6px solid #10B981 !important; }
.triage-border-1 { border-left: 6px solid #F59E0B !important; }
.triage-border-2 { border-left: 6px solid #EA580C !important; }
.triage-border-3 { border-left: 6px solid #DC2626 !important; }

/* Triage Badges */
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

.badge-0 { background: #ECFDF5; color: #10B981; border: 1px solid #A7F3D0; }
.badge-1 { background: #FFFBEB; color: #F59E0B; border: 1px solid #FDE68A; }
.badge-2 { background: #FFF7ED; color: #EA580C; border: 1px solid #FFEDD5; }
.badge-3 {
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

/* Metric Cards */
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

/* Acoustic Monitor Panel */
.acoustic-panel {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 14px 18px;
    margin-top: 14px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.decibel-bar-bg {
    width: 100%;
    height: 14px;
    background: #E2E8F0;
    border-radius: 9999px;
    overflow: hidden;
    margin: 8px 0;
}

.decibel-bar-fill {
    height: 100%;
    width: 94.4%;
    background: linear-gradient(90deg, #10B981 0%, #F59E0B 55%, #DC2626 88%);
    border-radius: 9999px;
    transition: width 0.3s ease;
}

/* Action Buttons (#2563EB) */
div.stButton > button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 10px 18px !important;
    font-size: 0.9rem !important;
    box-shadow: 0 2px 4px rgba(37, 99, 235, 0.20) !important;
    transition: all 0.15s ease-in-out !important;
}

div.stButton > button:hover {
    background-color: #1D4ED8 !important;
    box-shadow: 0 4px 8px rgba(37, 99, 235, 0.35) !important;
    transform: translateY(-1px);
}

/* Active Room Visual Highlight */
.active-room-card {
    border: 2px solid #2563EB !important;
    background: #EFF6FF !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.22) !important;
}

.active-room-pill {
    background: #2563EB;
    color: #FFFFFF;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 4px;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}
</style>
"""
render_html(CUSTOM_CSS)

# -----------------------------------------------------------------------------
# 3. TOP NAVIGATION BANNER
# -----------------------------------------------------------------------------
current_time_str = datetime.now().strftime("%Y-%m-%d • %H:%M:%S")

HEADER_HTML = f"""
<div class="nurse-nav-bar">
    <div class="nurse-nav-title">
        <span>🏥</span> OmniWard AI
    </div>
    <div class="nurse-nav-meta">
        <span class="shift-pill">🌙 Shift: Night</span>
        <span class="time-pill">{current_time_str}</span>
    </div>
</div>
"""
render_html(HEADER_HTML)

# -----------------------------------------------------------------------------
# 4. SESSION STATE & PATIENT ROOM DATABASE
# -----------------------------------------------------------------------------
if "active_room" not in st.session_state:
    st.session_state.active_room = "Room 204"
if "acknowledged_rooms" not in st.session_state:
    st.session_state.acknowledged_rooms = set()
if "doctor_called_rooms" not in st.session_state:
    st.session_state.doctor_called_rooms = set()

# Unified Clinical Patient & Room Database
ROOMS_DATABASE = {
    "Room 204": {
        "patient_id": "PAT-8821",
        "name": "Ahmed Hassan",
        "age_gender": "68 Y / Male",
        "unit": "ICU Alpha • Bed A",
        "admission": "Post-Op Rehab & Cardiac Monitoring",
        "nurse": "Sarah Jenkins, RN",
        "cam_id": "CAM-04 (ICU Alpha)",
        "severity_level": 3,
        "severity_label": "Critical Alert",
        "badge_class": "badge-3",
        "border_class": "triage-border-3",
        "color": "#DC2626",
        "event_type": "Screaming & Acute Fall Detected",
        "decibels": 94.4,
        "decibel_category": "Acute Scream (Emergency)",
        "torso_angle": 52.0,
        "posture_desc": "Abrupt Fall / Incline Exceeded",
        "boundary_status": "PERIMETER BREACHED",
        "boundary_color": "#DC2626",
        "boundary_icon": "⚠️",
        "alert_message": "Immediate Staff Intervention Required",
        "status": "Critical Alert Dispatched",
        "skeleton_state": "fall_52"
    },
    "Room 102": {
        "patient_id": "PAT-4412",
        "name": "Maria Silva",
        "age_gender": "74 Y / Female",
        "unit": "Stepdown Unit • Bed B",
        "admission": "Hip Replacement Recovery & Mobility",
        "nurse": "Michael Torres, RN",
        "cam_id": "CAM-02 (Stepdown West)",
        "severity_level": 2,
        "severity_label": "Level 2 Medium",
        "badge_class": "badge-2",
        "border_class": "triage-border-2",
        "color": "#EA580C",
        "event_type": "Bed Boundary Crossed / Fall Risk",
        "decibels": 64.0,
        "decibel_category": "Bed Rail Clatter / Restless",
        "torso_angle": 36.5,
        "posture_desc": "Leaning Over Bed Perimeter",
        "boundary_status": "RIGHT BED RAIL BREACHED",
        "boundary_color": "#EA580C",
        "boundary_icon": "⚠️",
        "alert_message": "Repositioning Assistance Prompted",
        "status": "Follow-up Required",
        "skeleton_state": "breach_36"
    },
    "Room 105": {
        "patient_id": "PAT-9903",
        "name": "David Chen",
        "age_gender": "59 Y / Male",
        "unit": "Telemetry Wing • Bed A",
        "admission": "Acute Bronchitis & COPD Observation",
        "nurse": "Elena Rostova, RN",
        "cam_id": "CAM-05 (Telemetry North)",
        "severity_level": 1,
        "severity_label": "Level 1 Low",
        "badge_class": "badge-1",
        "border_class": "triage-border-1",
        "color": "#F59E0B",
        "event_type": "Cough Paroxysm & Respiration Spike",
        "decibels": 73.1,
        "decibel_category": "Repetitive Cough Episodes",
        "torso_angle": 22.0,
        "posture_desc": "Elevated Fowler Incline",
        "boundary_status": "PERIMETER SECURE",
        "boundary_color": "#10B981",
        "boundary_icon": "🛡️",
        "alert_message": "Telemetry Logged for Scheduled Rounds",
        "status": "Logged for Rounds",
        "skeleton_state": "fowler_22"
    },
    "Room 301": {
        "patient_id": "PAT-1189",
        "name": "Eleanor Vance",
        "age_gender": "62 Y / Female",
        "unit": "General Ward 3 • Bed C",
        "admission": "Post-Procedure Observation",
        "nurse": "David Kim, RN",
        "cam_id": "CAM-01 (Ward 3 East)",
        "severity_level": 0,
        "severity_label": "Safe / Stable",
        "badge_class": "badge-0",
        "border_class": "triage-border-0",
        "color": "#10B981",
        "event_type": "Normal Vital Check & Resting Supine",
        "decibels": 38.5,
        "decibel_category": "Resting Ambient",
        "torso_angle": 12.0,
        "posture_desc": "Supine Resting Centered",
        "boundary_status": "PERIMETER SECURE",
        "boundary_color": "#10B981",
        "boundary_icon": "🛡️",
        "alert_message": "Vitals Consistent & Steady",
        "status": "Stable",
        "skeleton_state": "supine_12"
    },
    "Room 201": {
        "patient_id": "PAT-5531",
        "name": "Arthur Dent",
        "age_gender": "51 Y / Male",
        "unit": "Telemetry Wing • Bed B",
        "admission": "Cardiac Rhythm Evaluation",
        "nurse": "Sarah Jenkins, RN",
        "cam_id": "CAM-03 (Telemetry South)",
        "severity_level": 0,
        "severity_label": "Safe / Stable",
        "badge_class": "badge-0",
        "border_class": "triage-border-0",
        "color": "#10B981",
        "event_type": "Routine Posture Shift",
        "decibels": 41.2,
        "decibel_category": "Pillow / Posture Adjustment",
        "torso_angle": 15.0,
        "posture_desc": "Side-Angle Neutral",
        "boundary_status": "PERIMETER SECURE",
        "boundary_color": "#10B981",
        "boundary_icon": "🛡️",
        "alert_message": "Patient Repositioned Safely",
        "status": "Stable",
        "skeleton_state": "shift_15"
    },
    "Room 108": {
        "patient_id": "PAT-7720",
        "name": "Fatima Al-Sayed",
        "age_gender": "45 Y / Female",
        "unit": "Neurology Ward • Bed A",
        "admission": "Post-Concussion Observation",
        "nurse": "Rachel Adams, RN",
        "cam_id": "CAM-08 (Neuro North)",
        "severity_level": 1,
        "severity_label": "Level 1 Low",
        "badge_class": "badge-1",
        "border_class": "triage-border-1",
        "color": "#F59E0B",
        "event_type": "Restless Movement",
        "decibels": 52.8,
        "decibel_category": "Blanket Rustle & Movement",
        "torso_angle": 24.5,
        "posture_desc": "Semi-Fowler Active Shift",
        "boundary_status": "PERIMETER SECURE",
        "boundary_color": "#10B981",
        "boundary_icon": "🛡️",
        "alert_message": "Continuous Neuro Monitoring",
        "status": "Logged",
        "skeleton_state": "restless_24"
    },
    "Room 305": {
        "patient_id": "PAT-6619",
        "name": "John Watson",
        "age_gender": "71 Y / Male",
        "unit": "General Ward 3 • Bed A",
        "admission": "Pneumonia Convalescence",
        "nurse": "David Kim, RN",
        "cam_id": "CAM-09 (Ward 3 West)",
        "severity_level": 0,
        "severity_label": "Safe / Stable",
        "badge_class": "badge-0",
        "border_class": "triage-border-0",
        "color": "#10B981",
        "event_type": "Normal Sleep & Rest",
        "decibels": 35.0,
        "decibel_category": "Quiet Ambient Sleep",
        "torso_angle": 9.5,
        "posture_desc": "Full Supine Rest",
        "boundary_status": "PERIMETER SECURE",
        "boundary_color": "#10B981",
        "boundary_icon": "🛡️",
        "alert_message": "Steady Nocturnal Respiration",
        "status": "Stable",
        "skeleton_state": "sleep_9"
    }
}

# -----------------------------------------------------------------------------
# 5. DYNAMIC SURVEILLANCE FEED GENERATOR
# -----------------------------------------------------------------------------
def generate_surveillance_html(room_id: str, room_data: dict) -> str:
    """
    Generates high-precision visual surveillance feed with custom MediaPipe pose skeleton,
    boundary perimeter, and real-time biometric telemetry overlays for each specific room.
    """
    torso_angle = room_data.get("torso_angle", 12.0)
    cam_id = room_data.get("cam_id", "CAM-01 • 1080p HD")
    severity = room_data.get("severity_level", 0)
    unit_label = room_data.get("unit", "General Unit").split("•")[0].strip()

    if severity == 3:
        bound_stroke = "#DC2626"
        bound_dash = "6,4"
        bound_text = "⚠️ VIRTUAL BED SAFETY BOUNDARY (PERIMETER BREACHED)"
        badge_color = "#DC2626"
        sim_note_text = f"🦴 MediaPipe Skeleton Overlay Active (Torso Angle: {torso_angle}°) ● FALL DETECTED"
        sim_note_color = "#F87171"
        spine_x1, spine_y1, spine_x2, spine_y2 = 320, 130, 260, 230
        sh_l_x, sh_l_y, sh_r_x, sh_r_y = 280, 145, 360, 120
        arm_l1_x, arm_l1_y, arm_l2_x, arm_l2_y = 230, 190, 190, 240
        arm_r1_x, arm_r1_y, arm_r2_x, arm_r2_y = 410, 160, 440, 210
        pelvis_x1, pelvis_y1, pelvis_x2, pelvis_y2 = 240, 235, 280, 225
        leg_l_x, leg_l_y, leg_r_x, leg_r_y = 190, 280, 230, 295
        head_cx, head_cy = 340, 100
        arc_svg = '<path d="M 260 230 L 260 150" stroke="#94A3B8" stroke-width="1" stroke-dasharray="3,3" /><path d="M 260 170 A 60 60 0 0 1 300 162" fill="none" stroke="#DC2626" stroke-width="2" /><rect x="272" y="166" width="60" height="20" rx="3" fill="#DC2626" /><text x="278" y="180" fill="#FFFFFF" font-size="11" font-weight="700" font-family="JetBrains Mono, monospace">52.0°</text>'
    elif severity == 2:
        bound_stroke = "#EA580C"
        bound_dash = "6,4"
        bound_text = "⚠️ RIGHT BED RAIL BREACHED (PERIMETER HAZARD)"
        badge_color = "#EA580C"
        sim_note_text = f"🦴 MediaPipe Skeleton Overlay Active (Torso Angle: {torso_angle}°) ● BOUNDARY BREACH"
        sim_note_color = "#FB923C"
        spine_x1, spine_y1, spine_x2, spine_y2 = 310, 140, 260, 230
        sh_l_x, sh_l_y, sh_r_x, sh_r_y = 270, 150, 350, 130
        arm_l1_x, arm_l1_y, arm_l2_x, arm_l2_y = 225, 185, 205, 235
        arm_r1_x, arm_r1_y, arm_r2_x, arm_r2_y = 415, 175, 470, 235
        pelvis_x1, pelvis_y1, pelvis_x2, pelvis_y2 = 245, 235, 280, 225
        leg_l_x, leg_l_y, leg_r_x, leg_r_y = 210, 280, 250, 290
        head_cx, head_cy = 330, 110
        arc_svg = '<path d="M 260 230 L 260 160" stroke="#94A3B8" stroke-width="1" stroke-dasharray="3,3" /><path d="M 260 180 A 50 50 0 0 1 295 175" fill="none" stroke="#EA580C" stroke-width="2" /><rect x="270" y="178" width="60" height="20" rx="3" fill="#EA580C" /><text x="276" y="192" fill="#FFFFFF" font-size="11" font-weight="700" font-family="JetBrains Mono, monospace">36.5°</text>'
    elif severity == 1:
        bound_stroke = "#F59E0B"
        bound_dash = "4,4"
        bound_text = "🛡️ BED SAFETY BOUNDARY (PERIMETER SECURE)"
        badge_color = "#F59E0B"
        sim_note_text = f"🦴 MediaPipe Skeleton Overlay Active (Torso Angle: {torso_angle}°) ● {room_data['event_type'].upper()}"
        sim_note_color = "#FBBF24"
        spine_x1, spine_y1, spine_x2, spine_y2 = 295, 160, 255, 230
        sh_l_x, sh_l_y, sh_r_x, sh_r_y = 260, 165, 330, 155
        arm_l1_x, arm_l1_y, arm_l2_x, arm_l2_y = 225, 195, 210, 240
        arm_r1_x, arm_r1_y, arm_r2_x, arm_r2_y = 365, 185, 380, 235
        pelvis_x1, pelvis_y1, pelvis_x2, pelvis_y2 = 240, 235, 275, 230
        leg_l_x, leg_l_y, leg_r_x, leg_r_y = 210, 280, 240, 290
        head_cx, head_cy = 310, 130
        arc_svg = f'<path d="M 255 230 L 255 170" stroke="#94A3B8" stroke-width="1" stroke-dasharray="3,3" /><path d="M 255 190 A 40 40 0 0 1 280 188" fill="none" stroke="#F59E0B" stroke-width="2" /><rect x="260" y="190" width="56" height="18" rx="3" fill="#F59E0B" /><text x="265" y="203" fill="#FFFFFF" font-size="10" font-weight="700" font-family="JetBrains Mono, monospace">{torso_angle}°</text>'
    else:
        bound_stroke = "#10B981"
        bound_dash = "4,4"
        bound_text = "🛡️ BED SAFETY BOUNDARY (PERIMETER SECURE)"
        badge_color = "#10B981"
        sim_note_text = f"🦴 MediaPipe Skeleton Overlay Active (Torso Angle: {torso_angle}°) ● ALL VITALS STABLE"
        sim_note_color = "#4ADE80"
        spine_x1, spine_y1, spine_x2, spine_y2 = 280, 185, 245, 230
        sh_l_x, sh_l_y, sh_r_x, sh_r_y = 250, 185, 310, 185
        arm_l1_x, arm_l1_y, arm_l2_x, arm_l2_y = 220, 205, 205, 240
        arm_r1_x, arm_r1_y, arm_r2_x, arm_r2_y = 340, 205, 355, 240
        pelvis_x1, pelvis_y1, pelvis_x2, pelvis_y2 = 235, 235, 265, 230
        leg_l_x, leg_l_y, leg_r_x, leg_r_y = 205, 280, 235, 285
        head_cx, head_cy = 295, 155
        arc_svg = f'<path d="M 245 230 L 245 185" stroke="#94A3B8" stroke-width="1" stroke-dasharray="3,3" /><path d="M 245 200 A 30 30 0 0 1 265 198" fill="none" stroke="#10B981" stroke-width="2" /><rect x="248" y="200" width="52" height="18" rx="3" fill="#10B981" /><text x="253" y="213" fill="#FFFFFF" font-size="10" font-weight="700" font-family="JetBrains Mono, monospace">{torso_angle}°</text>'

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        background: transparent;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        overflow: hidden;
    }}
    .video-viewport {{
        position: relative;
        width: 100%;
        height: 380px;
        background: #0B1329;
        border-radius: 10px;
        border: 2px solid #334155;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    }}
    .video-header-overlay {{
        position: absolute;
        top: 0; left: 0; right: 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 16px;
        background: linear-gradient(180deg, rgba(11,19,41,0.95) 0%, rgba(11,19,41,0) 100%);
        z-index: 5;
    }}
    .rec-badge {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: {badge_color};
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }}
    .rec-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: {badge_color};
        animation: blink 1.2s infinite;
    }}
    @keyframes blink {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.2; }}
    }}
    .cam-meta {{
        color: #94A3B8;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
    }}
    .video-canvas-container {{
        flex: 1;
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        height: 100%;
    }}
    .video-simulation-note {{
        position: absolute;
        bottom: 10px;
        left: 12px;
        right: 12px;
        background: rgba(15, 23, 42, 0.90);
        backdrop-filter: blur(6px);
        border: 1px solid rgba(148, 163, 184, 0.25);
        color: #F8FAFC;
        padding: 8px 14px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
        display: flex;
        justify-content: space-between;
        align-items: center;
        z-index: 5;
    }}
    svg {{
        width: 100%;
        height: 100%;
        display: block;
    }}
</style>
</head>
<body>
<div class="video-viewport">
    <div class="video-header-overlay">
        <span class="rec-badge">
            <span class="rec-dot"></span> LIVE 30 FPS • {room_id} ({unit_label})
        </span>
        <span class="cam-meta">
            {cam_id} • 1080p HD
        </span>
    </div>

    <div class="video-canvas-container">
        <svg viewBox="0 0 640 360" xmlns="http://www.w3.org/2000/svg">
            <rect width="640" height="360" fill="#0B1329" />
            <line x1="0" y1="300" x2="640" y2="300" stroke="#1E293B" stroke-width="1" />
            <line x1="160" y1="300" x2="80" y2="360" stroke="#1E293B" stroke-width="1" />
            <line x1="480" y1="300" x2="560" y2="360" stroke="#1E293B" stroke-width="1" />

            <rect x="130" y="150" width="380" height="150" rx="10" fill="#1E293B" stroke="#334155" stroke-width="2" />
            <rect x="145" y="165" width="350" height="120" rx="6" fill="#334155" opacity="0.6" />

            <rect x="120" y="90" width="400" height="230" fill="none" stroke="{bound_stroke}" stroke-width="2" stroke-dasharray="{bound_dash}" />
            <text x="130" y="112" fill="{bound_stroke}" font-size="11" font-weight="700" font-family="Inter, sans-serif">
                {bound_text}
            </text>

            <!-- MediaPipe Pose Skeleton Lines -->
            <line x1="{spine_x1}" y1="{spine_y1}" x2="{spine_x2}" y2="{spine_y2}" stroke="#00F0FF" stroke-width="3" />
            <line x1="{sh_l_x}" y1="{sh_l_y}" x2="{sh_r_x}" y2="{sh_r_y}" stroke="#00F0FF" stroke-width="3" />
            <line x1="{sh_l_x}" y1="{sh_l_y}" x2="{arm_l1_x}" y2="{arm_l1_y}" stroke="#00FF66" stroke-width="2.5" />
            <line x1="{arm_l1_x}" y1="{arm_l1_y}" x2="{arm_l2_x}" y2="{arm_l2_y}" stroke="#00FF66" stroke-width="2.5" />
            <line x1="{sh_r_x}" y1="{sh_r_y}" x2="{arm_r1_x}" y2="{arm_r1_y}" stroke="#00FF66" stroke-width="2.5" />
            <line x1="{arm_r1_x}" y1="{arm_r1_y}" x2="{arm_r2_x}" y2="{arm_r2_y}" stroke="#00FF66" stroke-width="2.5" />
            <line x1="{pelvis_x1}" y1="{pelvis_y1}" x2="{pelvis_x2}" y2="{pelvis_y2}" stroke="#00F0FF" stroke-width="3" />
            <line x1="{pelvis_x1}" y1="{pelvis_y1}" x2="{leg_l_x}" y2="{leg_l_y}" stroke="#FFCC00" stroke-width="2.5" />
            <line x1="{pelvis_x2}" y1="{pelvis_y2}" x2="{leg_r_x}" y2="{leg_r_y}" stroke="#FFCC00" stroke-width="2.5" />

            <circle cx="{head_cx}" cy="{head_cy}" r="14" fill="#E2E8F0" stroke="#00F0FF" stroke-width="2" />
            <circle cx="{head_cx}" cy="{head_cy}" r="4" fill="#00F0FF" />
            <circle cx="{sh_l_x}" cy="{sh_l_y}" r="5" fill="#00FF66" />
            <circle cx="{sh_r_x}" cy="{sh_r_y}" r="5" fill="#00FF66" />
            <circle cx="{arm_l1_x}" cy="{arm_l1_y}" r="4" fill="#00FF66" />
            <circle cx="{arm_l2_x}" cy="{arm_l2_y}" r="4" fill="{bound_stroke if severity >= 2 else '#00FF66'}" />
            <circle cx="{arm_r1_x}" cy="{arm_r1_y}" r="4" fill="#00FF66" />
            <circle cx="{arm_r2_x}" cy="{arm_r2_y}" r="4" fill="{bound_stroke if severity >= 2 else '#00FF66'}" />
            <circle cx="{pelvis_x1}" cy="{pelvis_y1}" r="5" fill="#00F0FF" />
            <circle cx="{pelvis_x2}" cy="{pelvis_y2}" r="5" fill="#00F0FF" />
            <circle cx="{leg_l_x}" cy="{leg_l_y}" r="4" fill="#FFCC00" />
            <circle cx="{leg_r_x}" cy="{leg_r_y}" r="4" fill="#FFCC00" />

            {arc_svg}
        </svg>
    </div>

    <div class="video-simulation-note">
        <span>{sim_note_text}</span>
        <span style="color: {sim_note_color}; font-weight: 700;">● {room_data['status'].upper()}</span>
    </div>
</div>
</body>
</html>"""

# -----------------------------------------------------------------------------
# 6. DATA LOADING & FALLBACK
# -----------------------------------------------------------------------------
def load_patient_events() -> pd.DataFrame:
    rows = []
    for r_id, r_info in ROOMS_DATABASE.items():
        rows.append({
            "timestamp": "2026-09-08 04:36:12" if r_id == "Room 204" else "2026-09-08 04:31:05" if r_id == "Room 102" else "2026-09-08 04:22:30" if r_id == "Room 105" else "2026-09-08 04:15:18" if r_id == "Room 301" else "2026-09-08 04:02:40" if r_id == "Room 201" else "2026-09-08 03:51:10" if r_id == "Room 108" else "2026-09-08 03:40:00",
            "room_id": r_id,
            "patient_id": r_info["patient_id"],
            "patient_name": r_info["name"],
            "event_type": r_info["event_type"],
            "decibels": r_info["decibels"],
            "torso_angle": r_info["torso_angle"],
            "severity_level": r_info["severity_level"],
            "status": r_info["status"]
        })
    return pd.DataFrame(rows)

df_events_raw = load_patient_events()

# -----------------------------------------------------------------------------
# 7. WIDE 3-COLUMN ARCHITECTURE
# -----------------------------------------------------------------------------
col_events, col_live, col_analytics = st.columns([1.05, 1.4, 1.15], gap="large")

# =============================================================================
# LEFT COLUMN: REAL-TIME EVENT STREAM (col_events)
# =============================================================================
with col_events:
    render_html('<div class="col-header"><span>⚡</span> Real-Time Event Stream & Room Triage</div>')

    # Iterate through all rooms to make every patient room selectable
    for r_id, r_info in ROOMS_DATABASE.items():
        is_active = (r_id == st.session_state.active_room)
        active_class = "active-room-card" if is_active else ""
        active_tag = '<span class="active-room-pill">👁️ CURRENTLY VIEWING</span>' if is_active else ''

        card_html = f"""
        <div class="med-card {r_info['border_class']} {active_class}">
            <div class="card-title">
                <span>{r_id}: {r_info['event_type']}</span>
                <span class="triage-badge {r_info['badge_class']}">{r_info['severity_label']}</span>
            </div>
            <div class="card-meta">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                    <div>Patient: <span class="card-param">{r_info['name']} ({r_info['patient_id']})</span></div>
                    {active_tag}
                </div>
                <div>Acoustic Energy: <span class="card-param">{r_info['decibels']} dB SPL</span> ({r_info['decibel_category']})</div>
                <div>Torso Inclination: <span class="card-param">{r_info['torso_angle']}°</span> ({r_info['posture_desc']})</div>
                <div style="margin-top: 4px; font-size: 0.76rem; color: {r_info['color']}; font-weight: 700;">
                    ● {r_info['alert_message']}
                </div>
            </div>
        </div>
        """
        render_html(card_html)

        if not is_active:
            if st.button(f"👁️ View {r_id} Feed ({r_info['severity_label']})", key=f"btn_switch_ev_{r_id}", use_container_width=True):
                st.session_state.active_room = r_id
                st.rerun()
        else:
            st.button(f"✓ Currently Monitoring {r_id}", key=f"btn_cur_ev_{r_id}", disabled=True, use_container_width=True)

# =============================================================================
# MIDDLE COLUMN: PATIENT CANVAS & VISUAL OVERLAY (col_live)
# =============================================================================
with col_live:
    # Ensure active_room is valid
    if st.session_state.active_room not in ROOMS_DATABASE:
        st.session_state.active_room = "Room 204"

    cur_room = st.session_state.active_room
    cur_data = ROOMS_DATABASE[cur_room]

    # Room Navigation & Title Header
    render_html(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <div class="col-header" style="margin-bottom: 0; padding-bottom: 0; border-bottom: none;">
            <span>📹</span> Live Patient Surveillance — <strong style="color: #2563EB;">{cur_room}</strong>
        </div>
        <span class="triage-badge {cur_data['badge_class']}">{cur_data['severity_label']}</span>
    </div>
    """)

    # Quick Switch Room Navigator Toolbar
    render_html('<div style="font-size: 0.8rem; font-weight: 700; color: #475569; margin-bottom: 6px;">Switch Active Room:</div>')

    room_keys = list(ROOMS_DATABASE.keys())

    # Dropdown selector
    sel_room = st.selectbox(
        "Select Active Room:",
        room_keys,
        index=room_keys.index(cur_room),
        key="main_room_select",
        label_visibility="collapsed"
    )
    if sel_room != cur_room:
        st.session_state.active_room = sel_room
        st.rerun()

    # One-click quick jump chips (Row 1: Critical & Urgent, Row 2: Routine & Normal)
    q_c1, q_c2, q_c3, q_c4 = st.columns(4)
    for idx, r_k in enumerate(room_keys[:4]):
        r_obj = ROOMS_DATABASE[r_k]
        is_cur = (r_k == cur_room)
        icon = "🚨" if r_obj["severity_level"] == 3 else "⚠️" if r_obj["severity_level"] == 2 else "🟡" if r_obj["severity_level"] == 1 else "🟢"
        label = f"{'▶ ' if is_cur else ''}{r_k} {icon}"
        with [q_c1, q_c2, q_c3, q_c4][idx]:
            if st.button(label, key=f"qchip1_{r_k}", use_container_width=True):
                st.session_state.active_room = r_k
                st.rerun()

    q_c5, q_c6, q_c7 = st.columns(3)
    for idx, r_k in enumerate(room_keys[4:]):
        r_obj = ROOMS_DATABASE[r_k]
        is_cur = (r_k == cur_room)
        icon = "🚨" if r_obj["severity_level"] == 3 else "⚠️" if r_obj["severity_level"] == 2 else "🟡" if r_obj["severity_level"] == 1 else "🟢"
        label = f"{'▶ ' if is_cur else ''}{r_k} {icon}"
        with [q_c5, q_c6, q_c7][idx]:
            if st.button(label, key=f"qchip2_{r_k}", use_container_width=True):
                st.session_state.active_room = r_k
                st.rerun()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Dynamic Video Feed rendered via components.html to prevent any Markdown code leakage
    live_feed_code = generate_surveillance_html(cur_room, cur_data)
    components.html(live_feed_code, height=385, scrolling=False)

    # Patient Profile Card dynamically matching selected room
    PATIENT_CARD_HTML = f"""
    <div class="med-card" style="border-left: 6px solid {cur_data['color']}; margin-top: 12px;">
        <div class="card-title">
            <span>👤 Patient Record: {cur_data['name']}</span>
            <span class="triage-badge {cur_data['badge_class']}">{cur_data['severity_label']}</span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 8px; font-size: 0.85rem;">
            <div>
                <span style="color: #64748B;">Patient ID:</span><br>
                <strong style="color: #0F172A;">{cur_data['patient_id']}</strong>
            </div>
            <div>
                <span style="color: #64748B;">Room / Bed:</span><br>
                <strong style="color: #0F172A;">{cur_room} • {cur_data['unit']}</strong>
            </div>
            <div>
                <span style="color: #64748B;">Age & Gender:</span><br>
                <strong style="color: #0F172A;">{cur_data['age_gender']}</strong>
            </div>
            <div>
                <span style="color: #64748B;">Admission:</span><br>
                <strong style="color: #0F172A;">{cur_data['admission']}</strong>
            </div>
            <div>
                <span style="color: #64748B;">Assigned Nurse:</span><br>
                <strong style="color: #0F172A;">{cur_data['nurse']}</strong>
            </div>
            <div>
                <span style="color: #64748B;">Risk Evaluation:</span><br>
                <strong style="color: {cur_data['color']};">{cur_data['event_type']}</strong>
            </div>
        </div>
    </div>
    """
    render_html(PATIENT_CARD_HTML)

    # Dedicated Acoustic Monitoring Panel dynamically matching selected room
    decibels = cur_data["decibels"]
    fill_pct = min(100, max(12, int((decibels / 100.0) * 100)))

    if decibels >= 85:
        sound_color = "#DC2626"
        sound_badge = f'<span class="triage-badge badge-3">🚨 {cur_data["decibel_category"]}</span>'
    elif decibels >= 70:
        sound_color = "#EA580C"
        sound_badge = f'<span class="triage-badge badge-2">⚠️ {cur_data["decibel_category"]}</span>'
    elif decibels >= 50:
        sound_color = "#F59E0B"
        sound_badge = f'<span class="triage-badge badge-1">🟡 {cur_data["decibel_category"]}</span>'
    else:
        sound_color = "#10B981"
        sound_badge = f'<span class="triage-badge badge-0">🟢 {cur_data["decibel_category"]}</span>'

    ACOUSTIC_PANEL_HTML = f"""
    <div class="acoustic-panel">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: 700; font-size: 0.92rem; color: #1E293B;">
                🎙️ Acoustic Distress Monitor — {cur_room}
            </span>
            {sound_badge}
        </div>
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 8px;">
            <span style="font-size: 0.82rem; color: #64748B;">Decibel Reading:</span>
            <span style="font-size: 1.25rem; font-weight: 800; color: {sound_color}; font-family: 'JetBrains Mono', monospace;">
                {decibels} dB SPL
            </span>
        </div>
        <div class="decibel-bar-bg">
            <div class="decibel-bar-fill" style="width: {fill_pct}%;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: #64748B;">
            <span>30 dB (Ambient)</span>
            <span>65 dB (Speaking)</span>
            <span>85 dB (Distress)</span>
            <strong style="color: {sound_color};">{decibels} dB Current</strong>
        </div>
    </div>
    """
    render_html(ACOUSTIC_PANEL_HTML)

    # Action Buttons Row (#2563EB)
    is_ack = cur_room in st.session_state.acknowledged_rooms
    is_doc = cur_room in st.session_state.doctor_called_rooms

    btn_c1, btn_c2 = st.columns(2)
    with btn_c1:
        ack_btn_text = f"✓ {cur_room} Acknowledged" if is_ack else f"✅ Acknowledge Alert ({cur_room})"
        if st.button(ack_btn_text, key=f"btn_ack_{cur_room}", use_container_width=True, disabled=is_ack):
            st.session_state.acknowledged_rooms.add(cur_room)
            st.toast(f"Alert for {cur_room} ({cur_data['name']}) Acknowledged by Nurse Station", icon="✅")
            st.rerun()

    with btn_c2:
        doc_btn_text = f"✓ Doctor Alerted ({cur_room})" if is_doc else f"📞 Call Doctor ({cur_room})"
        if st.button(doc_btn_text, key=f"btn_doc_{cur_room}", use_container_width=True, disabled=is_doc):
            st.session_state.doctor_called_rooms.add(cur_room)
            st.toast(f"Emergency Page dispatched to On-Duty Physician for {cur_room}", icon="🚨")
            st.rerun()

    if is_ack:
        st.success(f"✓ Alert Acknowledged — Staff Dispatched to {cur_room} ({cur_data['name']})", icon="🩺")
    if is_doc:
        st.info(f"🚨 On-Duty Doctor Alerted via Hospital Emergency System ({cur_room})", icon="📟")

# =============================================================================
# RIGHT COLUMN: OVERALL ANALYTICS & DATA LOGS (col_analytics)
# =============================================================================
with col_analytics:
    render_html('<div class="col-header"><span>📊</span> Unit Analytics & Historical Logs</div>')

    METRICS_HTML = """
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
    """
    render_html(METRICS_HTML)

    # Filter Controls
    render_html('<div style="font-weight: 700; font-size: 0.88rem; color: #334155; margin-bottom: 8px;">Filter Telemetry Logs:</div>')
    fc1, fc2 = st.columns(2)

    with fc1:
        room_list = ["All Rooms"] + sorted(list(df_events_raw["room_id"].unique()))
        selected_room = st.selectbox("Filter by Room:", room_list, index=0)

    with fc2:
        severity_options = [
            "All Levels",
            "Level 0 (Safe)",
            "Level 1 (Low)",
            "Level 2 (Medium)",
            "Level 3 (Critical)"
        ]
        selected_severity = st.selectbox("Severity Level:", severity_options, index=0)

    # Quick Switch button from filter if different from active room
    if selected_room != "All Rooms" and selected_room != st.session_state.active_room:
        if st.button(f"📹 Switch Live Surveillance Feed to {selected_room}", key="btn_sync_filter_to_live", use_container_width=True):
            st.session_state.active_room = selected_room
            st.rerun()

    # Filter Application
    filtered_df = df_events_raw.copy()

    if selected_room != "All Rooms":
        filtered_df = filtered_df[filtered_df["room_id"] == selected_room]

    if selected_severity != "All Levels":
        target_lvl = int(selected_severity.split(" ")[1])
        filtered_df = filtered_df[filtered_df["severity_level"] == target_lvl]

    # Display Dataframe
    render_html('<div style="font-weight: 700; font-size: 0.85rem; color: #475569; margin-top: 10px; margin-bottom: 4px;">Telemetry Event History:</div>')

    display_cols = ["timestamp", "room_id", "patient_name", "event_type", "torso_angle", "decibels", "severity_level", "status"]
    available_cols = [c for c in display_cols if c in filtered_df.columns]

    display_df = filtered_df[available_cols].rename(columns={
        "timestamp": "Timestamp",
        "room_id": "Room",
        "patient_name": "Patient",
        "event_type": "Event Detected",
        "torso_angle": "Tilt (°)",
        "decibels": "Sound (dB)",
        "severity_level": "Level",
        "status": "Triage Status"
    })

    if not display_df.empty:
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=260
        )
    else:
        st.info("No telemetry logs found matching the selected filter criteria.", icon="ℹ️")

    # CSV Download Button
    csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Telemetry Log (.CSV)",
        data=csv_bytes,
        file_name=f"patient_telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
