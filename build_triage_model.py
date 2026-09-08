"""
Smart Visual-Acoustic Patient Monitor - Triage Model Training Pipeline
Generates synthetic clinical data, trains a Random Forest classifier, and exports triage_model.pkl.
"""

import os
import sys
import warnings

# Suppress sklearn and runtime warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from settings import MODEL_PATH, SAVED_MODELS_DIR


def generate_synthetic_triage_data(n_samples: int = 4000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates realistic clinical multimodal patient monitoring data.
    Features:
      - audio_event: 'none', 'cough', 'groan', 'scream'
      - audio_decibel: float (30.0 - 110.0 dB)
      - torso_angle: float (0.0 - 90.0 degrees from vertical)
      - bed_boundary_violation: int (0 or 1)
    Target:
      - target_severity: int (0: Safe, 1: Low, 2: Medium, 3: Critical)
    """
    np.random.seed(random_seed)
    
    samples_per_class = n_samples // 4
    records = []

    # Class 0: Safe (Normal sleeping or relaxed resting)
    for _ in range(samples_per_class):
        audio_event = "none" if np.random.rand() > 0.05 else "cough"
        decibels = np.random.uniform(32.0, 48.0)
        torso_angle = np.random.uniform(0.0, 24.0)
        bed_violation = 0 if np.random.rand() > 0.02 else 1
        records.append({
            "audio_event": audio_event,
            "audio_decibel": decibels,
            "torso_angle": torso_angle,
            "bed_boundary_violation": bed_violation,
            "target_severity": 0
        })

    # Class 1: Low Risk (Minor agitation, gentle cough, sitting up slightly)
    for _ in range(samples_per_class):
        audio_choice = np.random.choice(["cough", "none", "groan"], p=[0.60, 0.30, 0.10])
        decibels = np.random.uniform(48.0, 72.0)
        torso_angle = np.random.uniform(15.0, 35.0)
        bed_violation = 0 if np.random.rand() > 0.08 else 1
        records.append({
            "audio_event": audio_choice,
            "audio_decibel": decibels,
            "torso_angle": torso_angle,
            "bed_boundary_violation": bed_violation,
            "target_severity": 1
        })

    # Class 2: Medium Risk (Bed departure attempt, noticeable distress, loud groan, persistent cough)
    for _ in range(samples_per_class):
        audio_choice = np.random.choice(["groan", "cough", "none"], p=[0.55, 0.35, 0.10])
        decibels = np.random.uniform(58.0, 80.0)
        torso_angle = np.random.uniform(28.0, 48.0)
        bed_violation = 1 if np.random.rand() > 0.30 else 0
        records.append({
            "audio_event": audio_choice,
            "audio_decibel": decibels,
            "torso_angle": torso_angle,
            "bed_boundary_violation": bed_violation,
            "target_severity": 2
        })

    # Class 3: Critical Risk (Fall detected: angle > 45°, screaming, or severe bed egress with collapse)
    for _ in range(samples_per_class):
        audio_choice = np.random.choice(["scream", "groan", "none"], p=[0.70, 0.20, 0.10])
        # Acute decibel spike or fall signature
        if audio_choice == "scream":
            decibels = np.random.uniform(82.0, 108.0)
        else:
            decibels = np.random.uniform(55.0, 92.0)
        torso_angle = np.random.uniform(46.0, 90.0)
        bed_violation = 1 if np.random.rand() > 0.20 else 0
        records.append({
            "audio_event": audio_choice,
            "audio_decibel": decibels,
            "torso_angle": torso_angle,
            "bed_boundary_violation": bed_violation,
            "target_severity": 3
        })

    df = pd.DataFrame(records)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return df


def train_and_export_triage_model(output_path: Path = MODEL_PATH):
    """
    Builds and trains the Random Forest Triage Classifier pipeline and persists the artifact.
    """
    print("=" * 65)
    print("🏥 Starting Smart Patient Monitor Triage Model Training Pipeline...")
    print("=" * 65)

    df = generate_synthetic_triage_data(n_samples=5000, random_seed=42)
    print(f"✓ Generated {len(df)} synthetic patient observation records.")
    print("Class distribution:\n", df['target_severity'].value_counts().sort_index())

    X = df[["audio_event", "audio_decibel", "torso_angle", "bed_boundary_violation"]]
    y = df["target_severity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    categorical_features = ["audio_event"]
    numeric_features = ["audio_decibel", "torso_angle", "bed_boundary_violation"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(categories=[["none", "cough", "groan", "scream"]], handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_split=4,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    print("✓ Training Random Forest Triage Classifier...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✓ Model Evaluation - Overall Test Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Level 0 (Safe)", "Level 1 (Low)", "Level 2 (Medium)", "Level 3 (Critical)"]))

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"✓ Triage Model successfully exported to:\n  --> {output_path.resolve()}")
    print("=" * 65)
    return pipeline


if __name__ == "__main__":
    train_and_export_triage_model()
