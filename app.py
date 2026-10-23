import os
import zipfile
from io import BytesIO
from pathlib import Path

import face_recognition
import gdown
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Celebrity Face Recognition",
    page_icon="📸",
    layout="wide",
)

st.title("📸 Celebrity Face Detection and Recognition")
st.write(
    "Upload a photo or take a picture to detect faces and compare them "
    "with the registered celebrity gallery."
)

# Google Drive file ID from the shared Celebrity_Images.zip link.
GOOGLE_DRIVE_FILE_ID = "1tf2oMArRQvLm3ufiFxdvYi8-j7RRh-X_"

# Store the downloaded ZIP and extracted images in Streamlit's temporary folder.
DATA_DIR = Path("/tmp/celebrity_dataset")
ZIP_PATH = DATA_DIR / "Celebrity_Images.zip"
EXTRACT_DIR = DATA_DIR / "extracted"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# A lower face distance means a closer match. Tune this only after testing.
RECOGNITION_THRESHOLD = 0.48


# --------------------------------------------------
# 2. DOWNLOAD AND EXTRACT THE DATASET
# --------------------------------------------------

def folder_has_image_files(folder: Path) -> bool:
    """Return True if this folder contains supported image files directly."""
    try:
        return any(
            item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
            for item in folder.iterdir()
        )
    except OSError:
        return False


def find_dataset_root(extract_dir: Path):
    """
    Find the folder that contains the celebrity identity folders.
    Expected structure:
        Celebrity Images/
            virat_kohli/
                image1.jpg
            messi/
                image1.jpg
    """
    candidates = [
        extract_dir / "Celebrity Images",
        extract_dir / "archive" / "Celebrity Images",
        extract_dir,
    ]

    # Also check nested folders named "Celebrity Images".
    try:
        candidates.extend(
            p for p in extract_dir.rglob("Celebrity Images") if p.is_dir()
        )
    except OSError:
        pass

    # Remove duplicate paths while preserving order.
    unique_candidates = []
    seen = set()
    for path in candidates:
        key = str(path)
        if key not in seen and path.is_dir():
            unique_candidates.append(path)
            seen.add(key)

    # The dataset root should have subfolders that contain images,
    # either directly or inside nested folders.
    for candidate in unique_candidates:
        try:
            identity_folders = [
