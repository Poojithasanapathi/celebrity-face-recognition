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
                        [(left, top), (right, bottom)],
        outline=color,
        width=3,
    )

    # Display the name and approximate similarity score.
    label = f"{name} ({similarity:.1f}%)" if similarity is not None else name

    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    label_x = max(0, min(left, annotated_image.width - text_w - 8))
    label_y = top - text_h - 10

    if label_y < 0:
        label_y = min(
            bottom + 2,
            max(0, annotated_image.height - text_h - 8),
        )

    draw.rectangle(
        [
            (label_x, label_y),
            (label_x + text_w + 8, label_y + text_h + 8),
        ],
        fill=color,
    )
    draw.text(
        (label_x + 4, label_y + 4),
        label,
        fill="black",
        font=font,
    )

    results.append(
        {
            "face_number": face_number,
            "name": name,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "face_distance": round(distance, 4) if distance is not None else None,
            "similarity_estimate_percent": (
                round(similarity, 2) if similarity is not None else None
            ),
        }
    )


# --------------------------------------------------
# 8. DISPLAY RESULTS
# --------------------------------------------------

recognized_count = sum(row["name"] != "Unknown" for row in results)
unknown_count = len(results) - recognized_count

col1, col2, col3 = st.columns(3)
col1.metric("Faces detected", len(results))
col2.metric("Recognized", recognized_count)
col3.metric("Unknown", unknown_count)

st.subheader("Annotated photo")
st.image(annotated_image, use_container_width=True)

if not results:
    st.warning("No faces detected. Try a clearer photo with larger, well-lit faces.")


# --------------------------------------------------
# 9. TABLE AND DOWNLOADS
# --------------------------------------------------

st.subheader("Face coordinates and recognition details")

if results:
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results CSV",
        data=csv_bytes,
        file_name="face_recognition_results.csv",
        mime="text/csv",
    )

    image_buffer = BytesIO()
    annotated_image.save(image_buffer, format="JPEG")
    st.download_button(
        "Download annotated photo",
        data=image_buffer.getvalue(),
        file_name="annotated_faces.jpg",
        mime="image/jpeg",
    )

st.caption(
    "Similarity is an approximate distance-based score, not a calibrated "
    "probability. Incorrect identification is possible. An Unknown result "
    "does not prove that a person is absent from the gallery."
)

