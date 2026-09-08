# OmniWard AI
### A Multimodal Visual-Acoustic Patient Monitoring System

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20Estimation-0078D4?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![Huawei Club ECU](https://img.shields.io/badge/Huawei%20Club-ECU%20HCIA--AI-C7081B?style=for-the-badge&logo=huawei&logoColor=white)](https://www.ecu.edu.eg/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **OmniWard AI** is a real-time, non-invasive patient safety monitoring and clinical triage platform. By fusing computer vision pose estimation with acoustic distress analysis, OmniWard AI detects bed boundary breaches, patient falls, and vocal distress, empowering hospital staff to intervene before serious injury occurs.

---

##  Table of Contents
- [About The Project](#-about-the-project)
- [Context & Academic Foundation](#-context--academic-foundation)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Multimodal Detection Pipeline](#-multimodal-detection-pipeline)
  - [1. Visual Kinematics (MediaPipe)](#1-visual-kinematics-mediapipe)
  - [2. Acoustic Telemetry (Distress Analysis)](#2-acoustic-telemetry-distress-analysis)
  - [3. Central Nurse Station Dashboard](#3-central-nurse-station-dashboard)
- [Installation & Setup Guide](#-installation--setup-guide)
  - [Prerequisites](#prerequisites)
  - [Step 1: Clone the Repository](#step-1-clone-the-repository)
  - [Step 2: Create a Virtual Environment](#step-2-create-a-virtual-environment)
  - [Step 3: Install Dependencies](#step-3-install-dependencies)
  - [Step 4: Running the Backend (FastAPI)](#step-4-running-the-backend-fastapi)
  - [Step 5: Running the Frontend (Streamlit)](#step-5-running-the-frontend-streamlit)
- [Project Directory Structure](#-project-directory-structure)
- [Roadmap & Future Enhancements](#-roadmap--future-enhancements)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

##  About The Project

Patient falls and unobserved medical distress in hospital rooms are among the leading causes of preventable inpatient morbidity and prolonged hospitalization. Traditional nurse call buttons require conscious patient initiation, which is impossible during acute disorientation, sudden syncope, or severe trauma.

**OmniWard AI** addresses this challenge through an autonomous, multimodal surveillance architecture:
- **Vision-Based Safety**: Continuous bed boundary perimeter tracking and pose angle estimation without relying on wearable sensors.
- **Audio-Based Safety**: Real-time acoustic energy monitoring to detect screams, sharp cries of distress, and abnormal coughing fits.
- **Nurse Station Integration**: A centralized, low-latency dashboard that color-codes severity levels (Normal, Low, Medium, Critical) and provides immediate one-click response dispatch.

---

## Context & Academic Foundation

This project was developed as the final practical capstone project for the **HCIA-AI (Huawei Certified ICT Associate - Artificial Intelligence)** course hosted by **Huawei Club ECU** at the **Egyptian Chinese University (ECU)**.

The project demonstrates practical mastery of:
- Deep learning and computer vision pipeline design.
- Audio signal thresholding and feature extraction for distress classification.
- Real-time full-stack AI deployment integrating modern API architectures (FastAPI) and interactive clinical telemetry visualization (Streamlit).

---

##  Key Features

1. **MediaPipe Pose Estimation & Fall Kinematics**
   - Extracts real-time 33 anatomical landmarks (shoulders, hips, elbows, knees).
   - Computes dynamic torso inclination angles (e.g., $52^\circ$ threshold for acute fall classification).
   - Enforces virtual bed safety boundaries, detecting rail crossings before a complete fall occurs.

2. **Acoustic Distress & Decibel Telemetry**
   - Monitors decibel levels (dB SPL) in patient rooms.
   - Accurately classifies acute vocal distress (screams $>85\text{ dB}$, persistent paroxysmal coughing $70\text{--}80\text{ dB}$, and ambient sleep $<45\text{ dB}$).

3. **Interactive Multi-Room Nurse Station Dashboard (Streamlit)**
   - Live camera surveillance feed with visual skeleton overlays and breached perimeter indicators.
   - Dynamic switching across all patient rooms (ICU, Stepdown, Telemetry, General Wards).
   - Comprehensive triage queue and historical telemetry filtering.
   - Instant response workflows: **"Acknowledge Alert"** and **"Call On-Duty Doctor"**.

4. **Lightweight FastAPI Backend Service**
   - Provides RESTful endpoints for streaming telemetry data, room status updates, and emergency alert dispatches.

---

## System Architecture

OmniWard AI decouples data ingestion and processing into a synchronized multimodal pipeline:

```text
       ┌────────────────────────┐         ┌────────────────────────┐
       │   Video Feed (Camera)  │         │   Audio Feed (Mic)     │
       └───────────┬────────────┘         └───────────┬────────────┘
                   │                                  │
                   ▼                                  ▼
      ┌─────────────────────────┐        ┌─────────────────────────┐
      │  MediaPipe Pose Model   │        │ Acoustic Signal Engine  │
      │  - 33 Body Landmarks    │        │ - Decibel (dB SPL) Calc │
      │  - Torso Inclination    │        │ - Scream/Cough Detector │
      │  - Bed Boundary Check   │        └────────────┬────────────┘
      └────────────┬────────────┘                     │
                   │                                  │
                   └─────────────────┬────────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │  FastAPI Telemetry Gateway   │
                      │  - Event Ingestion & Triage  │
                      │  - Multi-Room State Store    │
                      └──────────────┬───────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │  Streamlit Nurse Station GUI │
                      │  - Live Surveillance Viewport│
                      │  - Event Stream & Log Filter │
                      │  - Clinical Alert Escalation │
                      └──────────────────────────────┘
```

---

##  Multimodal Detection Pipeline

### 1. Visual Kinematics (MediaPipe)
The system calculates the torso angle relative to the vertical bed axis using the 3D coordinates of the shoulders and hips:
$$\theta = \arccos\left(\frac{\mathbf{v}_{\text{torso}} \cdot \mathbf{v}_{\text{vertical}}}{\|\mathbf{v}_{\text{torso}}\| \|\mathbf{v}_{\text{vertical}}\|}\right)$$
- **$\theta < 20^\circ$**: Stable resting supine/elevated posture.
- **$20^\circ \le \theta < 45^\circ$**: Restless movement or partial bed-boundary breach.
- **$\theta \ge 45^\circ$ (with rapid descent)**: Critical fall detected $\rightarrow$ Level 3 Alert.

### 2. Acoustic Telemetry (Distress Analysis)
Audio frames are sampled at $16\text{ kHz}$ to compute root-mean-square (RMS) energy and peak sound pressure level (dB SPL):
- **Ambient baseline**: $30\text{--}45\text{ dB}$ (Normal breathing/rest).
- **Coughing episodes**: $65\text{--}78\text{ dB}$ (Logged for routine rounds).
- **Acoustic scream / acute distress**: $>85\text{ dB}$ (Immediate escalation).

### 3. Central Nurse Station Dashboard
- **Color-Coded Triage**:
  - 🟢 **Level 0 (Normal / Safe)**: Steady resting vitals and secure perimeter.
  - 🟡 **Level 1 (Low)**: Routine restlessness or elevated coughing.
  - 🟠 **Level 2 (Medium)**: Bed rail boundary hazard; repositioning required.
  - 🔴 **Level 3 (Critical)**: Immediate staff dispatch triggered.

---

##  Installation & Setup Guide

### Prerequisites
- Python **3.9+** (recommended: Python 3.10 or 3.11)
- `pip` package manager
- Webcam / microphone (or sample video/audio feeds)

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/omniward-ai.git
cd omniward-ai
```

### Step 2: Create a Virtual Environment
```bash
# On Linux / macOS
python3 -m venv venv
source venv/bin/activate

# On Windows (Command Prompt / PowerShell)
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(If `requirements.txt` is not yet created, install core dependencies directly:)*
```bash
pip install streamlit fastapi uvicorn mediapipe opencv-python numpy pandas librosa
```

### Step 4: Running the Backend (FastAPI)
Launch the FastAPI telemetry server:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
The interactive API documentation will be available at `http://localhost:8000/docs`.

### Step 5: Running the Frontend (Streamlit)
In a separate terminal (with the virtual environment activated), start the Nurse Station dashboard:
```bash
streamlit run gui.py
```
Open your browser at `http://localhost:8501` to access the central monitoring console.

---

##  Project Directory Structure

```text
omniward-ai/
├── data/
│   └── patient_events.csv        # Historical and simulated patient telemetry records
├── gui.py                        # Streamlit Nurse Station Central Dashboard
├── main.py                       # Unified dashboard launcher
├── main_camera_stream.py         # MediaPipe visual capture & pose estimation loop
├── server.py                     # FastAPI telemetry and event routing service
├── requirements.txt              # Project dependencies
├── README.md                     # Documentation & project manual
└── LICENSE                       # MIT License
```

---

##  Roadmap & Future Enhancements

- [ ] **Thermal Vision Integration**: Enable zero-light nocturnal surveillance using FLIR thermal cameras.
- [ ] **Huawei Ascend / MindSpore Optimization**: Accelerate pose inference pipelines utilizing Huawei Atlas hardware.
- [ ] **EHR / FHIR Interoperability**: Direct HL7/FHIR export to hospital electronic health records.
- [ ] **Predictive Agitation Analytics**: Sequence models (LSTM/Transformers) to predict bed exit intent prior to rail crossing.

---

##  License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

##  Acknowledgments

- **Huawei Club ECU (Egyptian Chinese University)**: For organizing the HCIA-AI course and providing technical guidance and mentoring.
- **Instructors & Mentors**: Special gratitude to the Huawei certified instructors for their invaluable insights into AI deployment, computer vision pipelines, and system optimization.
- **MediaPipe Team**: For the open-source pose estimation library.
- **Streamlit & FastAPI Communities**: For the developer-friendly frameworks that made real-time clinical prototyping seamless.

---
<p align="center">
  Developed with ❤️ as part of the <strong>HCIA-AI Course</strong> at <strong>Huawei Club ECU</strong>.
</p>
