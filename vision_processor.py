"""
Smart Visual-Acoustic Patient Monitor - Vision Processor Module
Utilizes MediaPipe Pose and OpenCV to extract 3D landmarks, compute torso angles,
detect bed safety boundary violations, and identify patient fall risks.
"""

import os
import sys
import math
import warnings
from typing import Dict, Any, Tuple, Optional

# -----------------------------------------------------------------------------
# C++ & LIBRARY WARNING SUPPRESSION
# Must be set before importing OpenCV or MediaPipe
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

import cv2
import numpy as np
import contextlib

@contextlib.contextmanager
def suppress_native_output():
    """
    Context manager to redirect file descriptors 1 (stdout) and 2 (stderr)
    at the OS level to /dev/null. Completely silences low-level C/C++ logging
    from MediaPipe, TensorFlow Lite, and Abseil during model load and warm-up.
    """
    try:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
        saved_stdout = os.dup(stdout_fd)
        saved_stderr = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        try:
            yield
        finally:
            os.dup2(saved_stdout, stdout_fd)
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stdout)
            os.close(saved_stderr)
    except Exception:
        yield

try:
    with suppress_native_output():
        import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

from settings import (
    BED_SAFETY_BOUNDS,
    TORSO_ANGLE_FALL_THRESHOLD,
    TORSO_ANGLE_WARNING_THRESHOLD,
)


class VisionProcessor:
    """
    Computer Vision subsystem for real-time patient posture and boundary analytics.
    Extracts 3D pose landmarks, assesses patient safety perimeter, and identifies fall hazards.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1
    ):
        self.bed_bounds = BED_SAFETY_BOUNDS
        self.fall_threshold = TORSO_ANGLE_FALL_THRESHOLD
        self.warning_threshold = TORSO_ANGLE_WARNING_THRESHOLD

        if MEDIAPIPE_AVAILABLE and mp is not None and hasattr(mp, "solutions"):
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = getattr(mp.solutions, "drawing_utils", None)
            self.mp_drawing_styles = getattr(mp.solutions, "drawing_styles", None)
            try:
                with suppress_native_output():
                    self.pose = self.mp_pose.Pose(
                        static_image_mode=False,
                        model_complexity=model_complexity,
                        smooth_landmarks=True,
                        enable_segmentation=False,
                        smooth_segmentation=False,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence
                    )
                    # Warm-up inference inside suppression block to silence first-run TFLite delegate creation
                    warmup_arr = np.zeros((480, 640, 3), dtype=np.uint8)
                    self.pose.process(warmup_arr)
            except Exception:
                self.pose = None
        else:
            self.mp_pose = None
            self.mp_drawing = None
            self.mp_drawing_styles = None
            self.pose = None

    def calculate_torso_angle(
        self,
        left_shoulder: Any,
        right_shoulder: Any,
        left_hip: Any,
        right_hip: Any
    ) -> float:
        """
        Calculates the torso inclination angle relative to the upward vertical axis (in degrees).
        0° = Perfectly upright posture (sitting or standing vertical)
        90° = Horizontal posture (flat on bed or fallen on floor)
        """
        # Shoulder midpoint
        s_x = (left_shoulder.x + right_shoulder.x) / 2.0
        s_y = (left_shoulder.y + right_shoulder.y) / 2.0

        # Hip midpoint
        h_x = (left_hip.x + right_hip.x) / 2.0
        h_y = (left_hip.y + right_hip.y) / 2.0

        # Vector pointing from Hip to Shoulder
        dx = s_x - h_x
        dy = s_y - h_y

        magnitude = math.sqrt(dx * dx + dy * dy)
        if magnitude < 1e-6:
            return 0.0

        # Vertical upward vector in normalized image space (where y points downward) is (0, -1)
        # Dot product with (0, -1) is -dy
        cos_angle = -dy / magnitude
        # Clamp to avoid numerical floating errors outside [-1, 1]
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)

        return float(round(angle_deg, 2))

    def check_bed_boundary_breach(self, landmarks, image_w: int, image_h: int) -> Tuple[bool, list]:
        """
        Determines whether critical patient keypoints have crossed outside the virtual bed safety zone.
        Keypoints evaluated: Head (Nose), Shoulders, Hips, Ankles.
        """
        x_min = self.bed_bounds["x_min"]
        y_min = self.bed_bounds["y_min"]
        x_max = self.bed_bounds["x_max"]
        y_max = self.bed_bounds["y_max"]

        if not landmarks:
            return False, []

        eval_indices = [
            0,   # Nose
            11,  # Left Shoulder
            12,  # Right Shoulder
            23,  # Left Hip
            24,  # Right Hip
            27,  # Left Ankle
            28,  # Right Ankle
        ]

        breached_points = []
        is_breached = False

        for idx in eval_indices:
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                if lm.visibility > 0.4:
                    if lm.x < x_min or lm.x > x_max or lm.y < y_min or lm.y > y_max:
                        is_breached = True
                        breached_points.append((int(lm.x * image_w), int(lm.y * image_h)))

        return is_breached, breached_points

    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Processes a single BGR camera frame and performs multimodal visual safety analytics.
        
        Returns:
            dict containing:
              - is_fall_detected (bool)
              - is_bed_boundary_crossed (bool)
              - torso_angle (float)
              - landmark_count (int)
              - frame_overlay (np.ndarray with clinical HUD and telemetry bounding boxes)
        """
        h, w, _ = frame.shape
        annotated_frame = frame.copy()

        # Coordinates for the virtual safety bed area
        box_x1 = int(self.bed_bounds["x_min"] * w)
        box_y1 = int(self.bed_bounds["y_min"] * h)
        box_x2 = int(self.bed_bounds["x_max"] * w)
        box_y2 = int(self.bed_bounds["y_max"] * h)

        is_fall_detected = False
        is_bed_boundary_crossed = False
        torso_angle = 0.0
        patient_detected = False

        if self.pose is not None:
            # Convert BGR to RGB for MediaPipe with contiguous memory layout to eliminate warnings
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
            rgb_frame.flags.writeable = False
            results = self.pose.process(rgb_frame)
            rgb_frame.flags.writeable = True

            if results.pose_landmarks:
                patient_detected = True
                lm = results.pose_landmarks.landmark

                # Left & Right Shoulders (11, 12), Left & Right Hips (23, 24)
                l_shoulder = lm[11]
                r_shoulder = lm[12]
                l_hip = lm[23]
                r_hip = lm[24]

                torso_angle = self.calculate_torso_angle(l_shoulder, r_shoulder, l_hip, r_hip)
                
                # Check for fall: torso tilted severely or horizontal
                if torso_angle >= self.fall_threshold:
                    is_fall_detected = True

                # Check bed boundary violations
                is_bed_boundary_crossed, breached_pts = self.check_bed_boundary_breach(
                    results.pose_landmarks, w, h
                )

                # Draw Pose Skeleton
                if self.mp_drawing and self.mp_pose:
                    self.mp_drawing.draw_landmarks(
                        annotated_frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(0, 255, 230), thickness=2, circle_radius=3
                        ),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(0, 200, 100), thickness=2
                        )
                    )

                # Highlight any breached landmarks with glowing markers
                for pt in breached_pts:
                    cv2.circle(annotated_frame, pt, 8, (0, 0, 255), -1)
                    cv2.circle(annotated_frame, pt, 14, (0, 0, 255), 2)

        # Draw Safety Boundary Box
        # Color: Green = Secure inside, Red = Perimeter Violation
        bed_box_color = (0, 0, 255) if is_bed_boundary_crossed else (0, 255, 120)
        cv2.rectangle(annotated_frame, (box_x1, box_y1), (box_x2, box_y2), bed_box_color, 2)
        cv2.putText(
            annotated_frame,
            "BED SAFETY ZONE" if not is_bed_boundary_crossed else "SAFETY ZONE BREACHED!",
            (box_x1 + 10, box_y1 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            bed_box_color,
            2,
            cv2.LINE_AA
        )

        # Draw Clinical HUD Telemetry Banner
        self._render_hud(
            annotated_frame,
            patient_detected,
            torso_angle,
            is_fall_detected,
            is_bed_boundary_crossed
        )

        return {
            "is_fall_detected": bool(is_fall_detected),
            "is_bed_boundary_crossed": bool(is_bed_boundary_crossed),
            "torso_angle": float(torso_angle),
            "patient_detected": bool(patient_detected),
            "frame_overlay": annotated_frame
        }

    def _render_hud(
        self,
        frame: np.ndarray,
        patient_detected: bool,
        torso_angle: float,
        is_fall_detected: bool,
        is_boundary_breached: bool
    ) -> None:
        """Renders an overlay information banner on top of the video feed."""
        h, w, _ = frame.shape
        
        # Semi-transparent top HUD bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (20, 24, 33), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Status text colors
        stat_color = (255, 255, 255)
        angle_color = (0, 0, 255) if is_fall_detected else ((0, 200, 255) if torso_angle > self.warning_threshold else (0, 255, 120))

        # Line 1 info
        cv2.putText(
            frame,
            f"TORSO ANGLE: {torso_angle:.1f}°",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            angle_color,
            2,
            cv2.LINE_AA
        )

        boundary_status = "BREACH" if is_boundary_breached else "OK"
        boundary_color = (0, 0, 255) if is_boundary_breached else (0, 255, 120)
        cv2.putText(
            frame,
            f"BED PERIMETER: {boundary_status}",
            (w // 2 - 80, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            boundary_color,
            2,
            cv2.LINE_AA
        )

        status_text = "FALL DETECTED!" if is_fall_detected else ("MONITORING" if patient_detected else "NO SUBJECT")
        status_color = (0, 0, 255) if is_fall_detected else (0, 255, 120)
        cv2.putText(
            frame,
            status_text,
            (w - 180, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            status_color,
            2,
            cv2.LINE_AA
        )

        # Emergency flashing warning banner if fall detected
        if is_fall_detected:
            cv2.rectangle(frame, (w // 4, h - 60), (3 * w // 4, h - 15), (0, 0, 220), -1)
            cv2.putText(
                frame,
                "CRITICAL: FALL DETECTED!",
                (w // 4 + 20, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

    def close(self):
        """Releases MediaPipe and vision resources cleanly."""
        if self.pose is not None:
            try:
                self.pose.close()
            except Exception:
                pass
            self.pose = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
