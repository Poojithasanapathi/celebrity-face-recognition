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
                                item for item in candidate.iterdir() if item.is_dir()
            ]
        except OSError:
            continue

        for identity_folder in identity_folders:
            if folder_has_image_files(identity_folder):
                return candidate
            try:
                if any(
                    folder_has_image_files(subfolder)
                    for subfolder in identity_folder.rglob("*")
                    if subfolder.is_dir()
                ):
                    return candidate
            except OSError:
                continue

    return None
    @st.cache_resource(show_spinner=False)
def download_and_prepare_dataset():
    """Download the ZIP from Drive, extract it, and locate the image gallery."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
   # Download the dataset if it is not already available in this instance.
    if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size == 0:
        if ZIP_PATH.exists():
            ZIP_PATH.unlink()

        downloaded_path = gdown.download(
            id=GOOGLE_DRIVE_FILE_ID,
            output=str(ZIP_PATH),
            quiet=False,
            fuzzy=True,
        )

        if not downloaded_path or not ZIP_PATH.exists() or ZIP_PATH.stat().st_size == 0:
            raise RuntimeError(
                "Google Drive download failed. Check that the file's General "
                "access is set to 'Anyone with the link' (Viewer)."
            )

    # Check that the downloaded file is really a ZIP.
    if not zipfile.is_zipfile(ZIP_PATH):
        raise RuntimeError(
            "The downloaded file is not a valid ZIP archive. Check the Google "
            "Drive sharing link and make sure it points to Celebrity_Images.zip."
        )

        # Extract only if the expected gallery cannot already be found.
    dataset_root = find_dataset_root(EXTRACT_DIR)
    if dataset_root is None:
        # Clear a partial extraction from an earlier failed run.
        for child in EXTRACT_DIR.iterdir():
            if child.is_dir():
                import shutil
                shutil.rmtree(child)
            else:
                child.unlink()

        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)

        dataset_root = find_dataset_root(EXTRACT_DIR)

    if dataset_root is None:
        raise RuntimeError(
            "The ZIP was downloaded and extracted, but celebrity folders were "
            "not found. The ZIP should contain a 'Celebrity Images' folder "
            "with one subfolder per identity, such as 'messi' or 'virat_kohli'."
        )

    return dataset_root


try:
    with st.spinner("Downloading and preparing the celebrity dataset..."):
        DATASET_PATH = download_and_prepare_dataset()
except Exception as error:
    st.error("Could not load the celebrity dataset.")
    st.exception(error)
    st.info(
        "Open the Google Drive ZIP's Share settings and set General access "
        "to 'Anyone with the link' and role to 'Viewer'."
    )
    st.stop()

st.success(f"Dataset folder found: `{DATASET_PATH}`")


# --------------------------------------------------
# 3. BUILD THE CELEBRITY REFERENCE GALLERY
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def build_gallery(dataset_path_string):
    encodings_list = []
    names_list = []
    skipped = 0
    dataset_path = Path(dataset_path_string)

    for root, _, files in os.walk(dataset_path):
        image_files = sorted(
            filename
            for filename in files
            if filename.lower().endswith(IMAGE_EXTENSIONS)
        )

        if not image_files:
            continue

        # The folder containing the image is used as the identity label.
        celebrity_name = Path(root).name

        for filename in image_files:
            image_path = Path(root) / filename

            try:
                image = face_recognition.load_image_file(str(image_path))
                locations = face_recognition.face_locations(image, model="hog")

                if not locations:
                    skipped += 1
                    continue

                encodings = face_recognition.face_encodings(
                    image,
                    known_face_locations=locations,
                )

                if not encodings:
                    skipped += 1
                    continue
                # If a reference image has several faces, use its largest face.
                areas = [
                    (bottom - top) * (right - left)
                    for top, right, bottom, left in locations
                ]
                largest_index = int(np.argmax(areas))

                encodings_list.append(encodings[largest_index])
                names_list.append(celebrity_name)

            except Exception as error:
                skipped += 1
                print(f"Skipped reference image {image_path}: {error}")

    return encodings_list, names_list, skipped


with st.spinner("Building the celebrity reference gallery..."):
    known_face_encodings, known_face_names, skipped = build_gallery(
        str(DATASET_PATH)
    )

if not known_face_encodings:
    st.error(
        f"No usable face encodings were found in `{DATASET_PATH}`. "
        "Check that each celebrity has a separate folder containing clear images."
    )
    st.stop()

st.success(
    f"Gallery loaded: {len(known_face_encodings)} reference faces from "
    f"{len(set(known_face_names))} identities."
)

if skipped:
    st.warning(f"{skipped} reference images were skipped because a face could not be encoded.")

with st.expander("View registered identities"):
    for identity in sorted(set(known_face_names)):
        count = known_face_names.count(identity)
        st.write(f"{identity}: {count} reference images")

# --------------------------------------------------
# 4. UPLOAD OR CAPTURE A PHOTO
# --------------------------------------------------

st.subheader("Choose a test photo")

input_method = st.radio(
    "Input method",
    ["Upload image", "Take photo with camera"],
    horizontal=True,
)

if input_method == "Upload image":
    photo = st.file_uploader(
        "Select a group photo or portrait",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
    )
else:
    photo = st.camera_input("Take a photo")

if photo is None:
    st.info("Upload or capture a photo to begin.")
    st.stop()
         
# 5. LOAD THE PHOTO
# --------------------------------------------------

try:
    pil_image = Image.open(photo).convert("RGB")
    image_rgb = np.array(pil_image)
except Exception as error:
    st.error(f"Could not open the photo: {error}")
    st.stop()


# 6. DETECT FACES
# --------------------------------------------------

try:
    with st.spinner("Detecting and recognizing faces..."):
        face_locations = face_recognition.face_locations(
            image_rgb,
            model="hog",
        )
        face_encodings = face_recognition.face_encodings(
            image_rgb,
            known_face_locations=face_locations,
        )
except Exception as error:
    st.error(f"Face detection failed: {error}")
    st.stop()


# --------------------------------------------------
# 7. RECOGNIZE FACES AND RECORD COORDINATES
# --------------------------------------------------

annotated_image = pil_image.copy()
draw = ImageDraw.Draw(annotated_image)

try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
except OSError:
    font = ImageFont.load_default()

results = []

for face_number, (location, encoding) in enumerate(
    zip(face_locations, face_encodings),
    start=1,
):
    # face_locations format: top, right, bottom, left.
    top, right, bottom, left = location

    # Convert the box to x, y, width, and height.
    x = int(left)
    y = int(top)
    w = int(right - left)
    h = int(bottom - top)

    distances = face_recognition.face_distance(
        known_face_encodings,
        encoding,
    )

    if len(distances):
        best_index = int(np.argmin(distances))
        distance = float(distances[best_index])

        # This is a rough distance-based similarity estimate, not a probability.
        similarity = max(0.0, (1.0 - distance) * 100.0)

        if distance < RECOGNITION_THRESHOLD:
            name = known_face_names[best_index]
        else:
            name = "Unknown"
    else:
        distance = None
        similarity = None
        name = "Unknown"

    color = "lime" if name != "Unknown" else "red"

    # Draw the face rectangle.
    draw.rectangle(
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
