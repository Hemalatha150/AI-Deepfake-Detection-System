"""Central configuration for the Deepfake Detection System."""
import os
from pathlib import Path

# Keep TensorFlow's console output quiet.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # original images / videos (real/, fake/)
FACES_DIR = DATA_DIR / "faces"      # extracted face crops (train/val/test)
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "deepfake_detector.keras"      # face model
SCENE_MODEL_PATH = MODEL_DIR / "scene_detector.keras"   # whole-image AI-generation model
SCENES_DIR = DATA_DIR / "scenes"                        # whole-image dataset
REPORT_DIR = ROOT / "reports"

# Model input
IMG_SIZE = 224
DEFAULT_ARCH = "efficientnet"       # efficientnet | xception | cnn

# Label convention: 0 = real (Original), 1 = fake (Deepfake).
# The network outputs P(fake).
CLASS_NAMES = ["real", "fake"]
DISPLAY_NAMES = {0: "Original", 1: "Deepfake"}
DEFAULT_THRESHOLD = 0.5

# Face extraction
FACE_MARGIN = 0.6                   # border around the face box; matches the framing of the training faces
MIN_FACE_CONFIDENCE = 0.90          # MTCNN confidence threshold
DETECT_MAX_SIDE = 720               # frames are downscaled to this before detection

# Video analysis
FRAMES_PER_VIDEO = 20
MAX_FACES_PER_FRAME = 2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
