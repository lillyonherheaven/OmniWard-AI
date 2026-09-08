"""
Smart Visual-Acoustic Patient Monitor - Real-Time Camera & Audio Stream Engine
Captures video from webcam (or synthesized clinical simulation stream), processes
MediaPipe pose landmarks, analyzes acoustic levels, and transmits telemetry to the backend.
"""

import os
import sys
import warnings

# -----------------------------------------------------------------------------
# C++ & LIBRARY WARNING SUPPRESSION
# Must precede library imports (OpenCV, MediaPipe, etc.)
# -----------------------------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_LOG_LEVEL"] = "error"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass

import time
import signal
import argparse
import math
import numpy as np
import cv2
import requests

from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from settings import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    STREAM_FPS,
    BACKEND_API_URL,
    BED_SAFETY_BOUNDS,
    TORSO_ANGLE_FALL_THRESHOLD,
)
from vision_processor import VisionProcessor
from audio_processor import AudioProcessor


def generate_simulated_clinical_frame(
    step: int,
    posture_mode: str = "normal",
    width: int = 640,
    height: int = 480
) -> np.ndarray:
    """
    Generates a realistic clinical room synthetic camera frame with an animated
    patient avatar for testing environments lacking physical camera hardware.
    """
    # Soft hospital room backdrop
    frame = np.full((height, width, 3), (238, 242, 245), dtype=np.uint8)

    # Hospital bed outline
    bed_x1 = int(BED_SAFETY_BOUNDS["x_min"] * width)
    bed_y1 = int(BED_SAFETY_BOUNDS["y_min"] * height)
    bed_x2 = int(BED_SAFETY_BOUNDS["x_max"] * width)
    bed_y2 = int(BED_SAFETY_BOUNDS["y_max"] * height)

    cv2.rectangle(frame, (bed_x1 - 10, bed_y1 - 10), (bed_x2 + 10, bed_y2 + 10), (200, 208, 216), -1)
    cv2.rectangle(frame, (bed_x1, bed_y1), (bed_x2, bed_y2), (255, 255, 255), -1)
    # Pillow
    cv2.rectangle(frame, (bed_x1 + 30, bed_y1 + 15), (bed_x2 - 30, bed_y1 + 80), (220, 230, 240), -1)

    # Patient body position depending on posture mode
    center_x = (bed_x1 + bed_x2) // 2
    center_y = (bed_y1 + bed_y2) // 2

    t = step * 0.05
    breathe = int(3 * math.sin(t))

    if posture_mode == "normal":
        # Resting upright / slight incline in bed
        head_pos = (center_x, center_y - 80 + breathe)
        torso_end = (center_x, center_y + 40)
        l_hip = (center_x - 30, center_y + 80)
        r_hip = (center_x + 30, center_y + 80)
    elif posture_mode == "breach":
        # Patient leaning far out to the right edge of bed
        offset_x = int((bed_x2 - center_x) * 1.05)
        head_pos = (center_x + offset_x, center_y - 20)
        torso_end = (center_x + offset_x - 30, center_y + 60)
        l_hip = (center_x + offset_x - 50, center_y + 110)
        r_hip = (center_x + offset_x, center_y + 110)
    elif posture_mode == "fall":
        # Patient fallen across floor (horizontal, angle > 65°)
        head_pos = (center_x + 120, bed_y2 + 35)
        torso_end = (center_x - 20, bed_y2 + 35)
        l_hip = (center_x - 80, bed_y2 + 25)
        r_hip = (center_x - 80, bed_y2 + 45)
    else:
        head_pos = (center_x, center_y - 50)
        torso_end = (center_x, center_y + 30)
        l_hip = (center_x - 25, center_y + 70)
        r_hip = (center_x + 25, center_y + 70)

    # Draw synthetic stylized person silhouette
    cv2.circle(frame, head_pos, 22, (70, 90, 120), -1)
    cv2.line(frame, head_pos, torso_end, (70, 90, 120), 10)
    cv2.line(frame, torso_end, l_hip, (70, 90, 120), 8)
    cv2.line(frame, torso_end, r_hip, (70, 90, 120), 8)

    return frame


def run_camera_stream(
    camera_source=CAMERA_INDEX,
    backend_url=BACKEND_API_URL,
    room_id="ICU-101",
    patient_id="PAT-8821",
    simulate_mode=False,
    max_iterations=None
):
    """
    Main loop executing frame ingestion, MediaPipe posture analysis,
    acoustic feature evaluation, and telemetry dispatch.
    """
    print("=" * 65)
    print("📹 Initializing Smart Visual-Acoustic Camera Stream...")
    print(f"   Target Backend API: {backend_url}")
    print(f"   Room: {room_id} | Patient: {patient_id}")
    print("=" * 65)

    vision = VisionProcessor()
    audio = AudioProcessor()

    cap = None
    if not simulate_mode:
        print(f"Attempting to open camera device source: {camera_source}...")
        try:
            cap = cv2.VideoCapture(camera_source)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            if not cap.isOpened():
                print("⚠️ Physical camera device unavailable. Switching seamlessly to simulation stream mode.")
                simulate_mode = True
        except Exception as e:
            print(f"⚠️ Camera capture initialization failed ({e}). Entering simulation stream.")
            simulate_mode = True

    step = 0
    posture_modes = ["normal", "normal", "breach", "normal", "fall"]
    current_posture_idx = 0
    audio_sim_events = ["none", "cough", "none", "groan", "scream"]
    current_audio_idx = 0

    last_api_dispatch = time.time()
    api_endpoint = f"{backend_url.rstrip('/')}/api/v1/process-frame"

    try:
        while True:
            if max_iterations is not None and step >= max_iterations:
                print(f"Reached max iterations limit ({max_iterations}). Stopping.")
                break

            step += 1
            loop_start = time.time()

            # Cycle through simulation states every 90 frames (~3 sec)
            if step % 90 == 0:
                current_posture_idx = (current_posture_idx + 1) % len(posture_modes)
                current_audio_idx = (current_audio_idx + 1) % len(audio_sim_events)

            active_posture = posture_modes[current_posture_idx]
            simulated_audio_event = audio_sim_events[current_audio_idx]

            # 1. Grab Frame
            if simulate_mode or cap is None or not cap.isOpened():
                frame = generate_simulated_clinical_frame(
                    step, posture_mode=active_posture, width=FRAME_WIDTH, height=FRAME_HEIGHT
                )
            else:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Failed to read frame from camera. Falling back to synthetic frame.")
                    frame = generate_simulated_clinical_frame(step, active_posture, FRAME_WIDTH, FRAME_HEIGHT)

            # 2. Vision Processing (MediaPipe Pose + Boundaries)
            vision_results = vision.analyze_frame(frame)
            torso_angle = vision_results["torso_angle"]
            is_fall = vision_results["is_fall_detected"]
            is_breached = vision_results["is_bed_boundary_crossed"]
            display_frame = vision_results["frame_overlay"]

            # If simulation mode, adjust angle to reflect simulation state
            if simulate_mode:
                if active_posture == "fall":
                    torso_angle = 68.5
                    is_fall = True
                elif active_posture == "breach":
                    is_breached = True
                    torso_angle = 34.0
                else:
                    torso_angle = 12.0
                    is_fall = False

            # 3. Audio Processing
            synthetic_sound = audio.synthesize_test_signal(event_type=simulated_audio_event)
            audio_results = audio.analyze_audio(synthetic_sound)
            audio_event = audio_results["event_type"]
            decibels = audio_results["decibels"]

            # 4. Telemetry API Dispatch (Throttled to 2 Hz to avoid network saturation)
            now = time.time()
            if now - last_api_dispatch >= 0.5:
                payload = {
                    "room_id": room_id,
                    "patient_id": patient_id,
                    "torso_angle": float(torso_angle),
                    "bed_boundary_violation": 1 if is_breached else 0,
                    "audio_event": audio_event,
                    "audio_decibel": float(decibels),
                    "notes": f"Simulated posture: {active_posture}" if simulate_mode else "Live Camera Feed"
                }

                try:
                    resp = requests.post(api_endpoint, json=payload, timeout=0.8)
                    if resp.status_code == 200:
                        data = resp.json()
                        sev = data.get("severity_level", 0)
                        sev_name = data.get("severity_name", "Safe")
                        status_str = data.get("status", "")
                        print(
                            f"[TELEMETRY] Room: {room_id} | Torso: {torso_angle:.1f}° | "
                            f"Sound: {audio_event} ({decibels:.1f} dB) | Triage: L{sev} ({sev_name}) - {status_str}"
                        )
                except Exception as err:
                    # Non-blocking if server isn't running yet
                    pass

                last_api_dispatch = now

            # 5. Render GUI window if display server available
            try:
                if "DISPLAY" in os.environ or sys.platform in ["win32", "darwin"]:
                    cv2.imshow("Smart Visual-Acoustic Patient Monitor", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord("q"), 27]:  # 'q' or ESC
                        print("User terminated stream via hotkey.")
                        break
                    elif key == ord("f"):
                        current_posture_idx = 4  # Fall
                    elif key == ord("s"):
                        current_audio_idx = 4    # Scream
            except Exception:
                # Running in headless container environment
                pass

            # Maintain target framerate
            elapsed = time.time() - loop_start
            delay = max(0.001, (1.0 / STREAM_FPS) - elapsed)
            time.sleep(delay)

    except KeyboardInterrupt:
        print("\nExiting Camera Stream loop...")
    finally:
        if cap is not None:
            cap.release()
        vision.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("✓ Vision & Audio Stream resources cleanly released.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Patient Monitor Real-Time Stream")
    parser.add_argument("--camera", default=CAMERA_INDEX, help="Camera index or RTSP video path")
    parser.add_argument("--room", default="ICU-101", help="Clinical Room ID")
    parser.add_argument("--patient", default="PAT-8821", help="Patient Identification Number")
    parser.add_argument("--simulate", action="store_true", help="Force synthetic clinical stream")
    parser.add_argument("--max-iterations", type=int, default=None, help="Stop after N frames (useful for test runs)")
    args = parser.parse_args()

    run_camera_stream(
        camera_source=args.camera,
        room_id=args.room,
        patient_id=args.patient,
        simulate_mode=args.simulate,
        max_iterations=args.max_iterations
    )
