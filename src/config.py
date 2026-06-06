"""
config.py
---------
Central configuration for Handwritten Character Recognition project.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR   = os.path.join(OUTPUTS_DIR, "plots")
MODELS_DIR  = os.path.join(OUTPUTS_DIR, "models")

# ── Dataset ───────────────────────────────────────────────────────────────────
# After downloading from Kaggle, place the CSV files here:
#   data/emnist-letters-train.csv
#   data/emnist-letters-test.csv
TRAIN_CSV = os.path.join(DATA_DIR, "emnist-letters-train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "emnist-letters-test.csv")

# ── EMNIST Letters label map ──────────────────────────────────────────────────
# Labels are 1–26 → map to A–Z
NUM_CLASSES   = 26
LABEL_OFFSET  = 1          # EMNIST letters start at 1, not 0
CLASS_NAMES   = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # ['A'..'Z']

# ── Image settings ────────────────────────────────────────────────────────────
IMG_SIZE      = 28         # 28×28 pixels
IMG_CHANNELS  = 1          # grayscale
INPUT_SHAPE   = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)

# ── Training ──────────────────────────────────────────────────────────────────
TEST_SIZE     = 0.15
VAL_SIZE      = 0.10
RANDOM_STATE  = 42
BATCH_SIZE    = 64
EPOCHS        = 50
LEARNING_RATE = 0.001
