"""
Smart Visual-Acoustic Patient Monitor - Audio Processor Module
Analyzes acoustic waveforms to extract RMS Volume (dB), Pitch (Hz), and Energy,
and classifies critical distress sounds such as screams, coughs, and groans.
"""

import os
import math
import warnings
from typing import Dict, Any, Tuple
import numpy as np

# Suppress runtime numerical warnings
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

from settings import (
    AUDIO_SAMPLE_RATE,
    DECIBEL_SCREAM_THRESHOLD,
    DECIBEL_COUGH_THRESHOLD,
    DECIBEL_GROAN_THRESHOLD,
    DECIBEL_AMBIENT_BASELINE,
)


class AudioProcessor:
    """
    Acoustic Signal Processing subsystem for patient acoustic surveillance.
    Extracts decibel level, fundamental pitch frequency, and energy characteristics
    to detect auditory distress indicators.
    """

    def __init__(self, sample_rate: int = AUDIO_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.scream_threshold = DECIBEL_SCREAM_THRESHOLD
        self.cough_threshold = DECIBEL_COUGH_THRESHOLD
        self.groan_threshold = DECIBEL_GROAN_THRESHOLD
        self.ambient_baseline = DECIBEL_AMBIENT_BASELINE

    def compute_rms_and_decibels(self, audio_data: np.ndarray) -> Tuple[float, float]:
        """
        Computes Root Mean Square (RMS) energy and converts to calibrated Decibel (dB SPL).
        """
        if len(audio_data) == 0:
            return 0.0, self.ambient_baseline

        # Normalize signal to [-1.0, 1.0] if integer format
        if audio_data.dtype in [np.int16, np.int32]:
            max_val = np.iinfo(audio_data.dtype).max
            audio_data = audio_data.astype(np.float32) / max_val

        rms = float(np.sqrt(np.mean(audio_data ** 2)))
        if rms < 1e-7:
            return 0.0, self.ambient_baseline

        # Calibrate Decibel value on 30 - 110 dB clinical scale
        # 20 * log10(rms / ref_p0)
        p0 = 20e-6
        raw_db = 20.0 * math.log10(rms / p0)
        # Scale to typical room acoustic bounds (30 dB whisper to 110 dB severe scream)
        scaled_db = float(np.clip(raw_db - 20.0, 30.0, 115.0))
        return float(rms), round(scaled_db, 1)

    def estimate_pitch_autocorr(self, audio_data: np.ndarray) -> float:
        """
        Estimates the fundamental frequency (pitch in Hz) via normalized autocorrelation.
        Applicable range: 70 Hz (low groan/vocal tone) to 2500 Hz (high-pitched scream).
        """
        if len(audio_data) < 256:
            return 0.0

        # Remove DC bias
        signal = audio_data - np.mean(audio_data)

        # Autocorrelation
        corr = np.correlate(signal, signal, mode='full')
        corr = corr[len(corr) // 2:]

        d = np.diff(corr)
        start_indices = np.where(d > 0)[0]
        if len(start_indices) == 0:
            return 0.0
        start = start_indices[0]

        peak = np.argmax(corr[start:]) + start
        if peak == 0:
            return 0.0

        f0 = float(self.sample_rate / peak)
        if 60.0 <= f0 <= 3000.0:
            return round(f0, 1)
        return 0.0

    def compute_spectral_energy(self, audio_data: np.ndarray) -> float:
        """Computes short-time total spectral energy."""
        if len(audio_data) == 0:
            return 0.0
        return float(np.sum(audio_data ** 2))

    def analyze_audio(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Performs rule-based acoustic feature extraction and classification.
        
        Returns:
            dict containing:
              - event_type: 'scream', 'cough', 'groan', or 'silent'
              - decibels: float (dB)
              - is_critical_sound: bool
              - pitch: float (Hz)
              - energy: float
        """
        rms, decibels = self.compute_rms_and_decibels(audio_data)
        pitch = self.estimate_pitch_autocorr(audio_data)
        energy = self.compute_spectral_energy(audio_data)

        # Rule-based classification:
        # Scream: High decibels (>85 dB) combined with high pitch (>650 Hz) or extreme volume (>90 dB)
        if decibels >= self.scream_threshold or (decibels >= 80.0 and pitch >= 600.0):
            event_type = "scream"
            is_critical = True
        # Cough: Explosive impulsive transient energy, moderate-high dB (70-85 dB), wide frequency
        elif decibels >= self.cough_threshold:
            # Coughs typically have sudden onset and lower tonal stability
            event_type = "cough"
            is_critical = False
        # Groan: Prolonged low pitch (80-350 Hz) with audible distress dB (58-70 dB)
        elif decibels >= self.groan_threshold and (70.0 <= pitch <= 450.0 or decibels >= 62.0):
            event_type = "groan"
            is_critical = False
        else:
            event_type = "silent"
            is_critical = False

        return {
            "event_type": event_type,
            "decibels": float(decibels),
            "is_critical_sound": bool(is_critical),
            "pitch_hz": float(pitch),
            "energy": float(round(energy, 4))
        }

    @staticmethod
    def synthesize_test_signal(
        event_type: str = "silent",
        duration_sec: float = 1.0,
        sample_rate: int = AUDIO_SAMPLE_RATE
    ) -> np.ndarray:
        """
        Generates synthetic clinical audio waveforms for testing, calibration,
        and headless demonstration when physical microphones are unavailable.
        """
        t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)

        if event_type == "scream":
            # High-amplitude, high-frequencyFM modulated sound with turbulence noise
            carrier = 0.85 * np.sin(2 * np.pi * (850 + 250 * np.sin(2 * np.pi * 5 * t)) * t)
            noise = 0.15 * np.random.normal(0, 0.2, len(t))
            signal = (carrier + noise).astype(np.float32)
        elif event_type == "cough":
            # Repetitive impulsive bursts with decaying envelope
            envelope = np.exp(-12 * (t % 0.35))
            noise = np.random.normal(0, 0.7, len(t))
            burst = 0.5 * np.sin(2 * np.pi * 320 * t)
            signal = ((burst + noise) * envelope).astype(np.float32)
        elif event_type == "groan":
            # Low-frequency droning harmonic sound
            f0 = 140.0
            signal = (
                0.55 * np.sin(2 * np.pi * f0 * t) +
                0.30 * np.sin(2 * np.pi * 2 * f0 * t) +
                0.15 * np.random.normal(0, 0.05, len(t))
            ).astype(np.float32)
        else:
            # Silent ambient ward background noise (soft hum)
            signal = (0.02 * np.random.normal(0, 0.1, len(t))).astype(np.float32)

        return signal
