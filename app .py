import zipfile
import gdown
import os
from pathlib import Path
from io import BytesIO

import streamlit as st
import face_recognition
import numpy as np
import pandas as pd

from PIL import Image, ImageDraw, ImageFont


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Celebrity Face Recognition",
    page_icon="📸",
    layout="wide"
)

st.title("📸 Celebrity Face Detection and Recognition")
st.write(
    "Upload a photo or take a picture to detect faces "
    "and compare them with the registered celebrity gallery."
)


# --------------------------------------------------
# 2. FIND THE DATASET FOLDER
# --------------------------------------------------

APP_DIR = Path(__file__).parent

possible_paths = [
    APP_DIR / "Celebrity Images",
    Path("/content/archive/Celebrity Images"),
    Path("/content/celebrity_dataset"),
]

DATASET_PATH = next(
    (path for path in possible_paths if path.is_dir()),
    None
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# Lower distance means a closer match.
# Adjust only after evaluating false matches.
RECOGNITION_THRESHOLD = 0.48


# --------------------------------------------------
# 3. BUILD THE CELEBRITY REFERENCE GALLERY
# --------------------------------------------------

@st.cache_resource
def build_gallery(dataset_path):

    encodings_list = []
    names_list = []
    skipped = 0

    for root, dirs, files in os.walk(dataset_path):

        image_files = sorted([
            f for f in files
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ])

        if not image_files:
            continue

        # The image-containing folder name is the identity label.
        celebrity_name = os.path.basename(root)

        for filename in image_files:

            image_path = os.path.join(root, filename)

            try:
                image = face_recognition.load_image_file(image_path)

                locations = face_recognition.face_locations(
                    image,
                    model="hog"
                )

                if not locations:
                    skipped += 1
                    continue

                encodings = face_recognition.face_encodings(
                    image,
                    known_face_locations=locations
                )

                if not encodings:
                    skipped += 1
                    continue

                # Select the largest face in a reference image.
                areas = [
                    (bottom - top) * (right - left)
                    for top, right, bottom, left in locations
                ]

                largest_index = int(np.argmax(areas))

                encodings_list.append(encodings[largest_index])
                names_list.append(celebrity_name)

            except Exception as error:
                skipped += 1
                print("Skipped:", image_path, error)

    return encodings_list, names_list, skipped


if DATASET_PATH is None:
    st.error("Celebrity dataset folder was not found.")
    st.write("In Colab, check that this folder exists:")
    st.code("/content/archive/Celebrity Images")
    st.write(
        "For GitHub deployment, place the Celebrity Images "
        "folder beside app.py."
    )
    st.stop()

with st.spinner("Loading celebrity reference images..."):
    known_face_encodings, known_face_names, skipped = build_gallery(
        str(DATASET_PATH)
    )

if len(known_face_encodings) == 0:
    st.error(
        f"No usable face encodings found in {DATASET_PATH}. "
        "Check the dataset folders and images."
    )
    st.stop()

st.success(
    f"Gallery loaded: {len(known_face_encodings)} reference faces "
    f"from {len(set(known_face_names))} identities."
)

if skipped:
    st.warning(f"{skipped} reference images were skipped.")

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
    horizontal=True
)

photo = None

if input_method == "Upload image":
    photo = st.file_uploader(
        "Select a group photo or portrait",
        type=["jpg", "jpeg", "png", "webp", "bmp"]
    )
else:
    photo = st.camera_input("Take a photo")

if photo is None:
    st.info("Upload or capture a photo to begin.")
    st.stop()


# --------------------------------------------------
# 5. LOAD THE PHOTO
# --------------------------------------------------

try:
    pil_image = Image.open(photo).convert("RGB")
    image_rgb = np.array(pil_image)
except Exception as error:
    st.error(f"Could not open the photo: {error}")
    st.stop()


# --------------------------------------------------
# 6. DETECT FACES
# --------------------------------------------------

with st.spinner("Detecting and recognizing faces..."):

    face_locations = face_recognition.face_locations(
        image_rgb,
        model="hog"
    )

    face_encodings = face_recognition.face_encodings(
        image_rgb,
        known_face_locations=face_locations
    )


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
    zip(face_locations, face_encodings), start=1
):

    # face_locations format: top, right, bottom, left
    top, right, bottom, left = location

    # Convert to x, y, width and height
    x = int(left)
    y = int(top)
    w = int(right - left)
    h = int(bottom - top)

    distances = face_recognition.face_distance(
        known_face_encodings,
        encoding
    )

    if len(distances):
        best_index = int(np.argmin(distances))
        distance = float(distances[best_index])

        # This is a similarity estimate, NOT a probability.
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
        width=3
    )

    if similarity is not None:
        label = f"{name} ({similarity:.1f}%)"
    else:
        label = name

    # Keep the label within the image boundaries.
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    label_x = max(
        0,
        min(left, annotated_image.width - text_w - 8)
    )

    label_y = top - text_h - 10

    if label_y < 0:
        label_y = min(
            bottom + 2,
            max(0, annotated_image.height - text_h - 8)
        )

    draw.rectangle(
        [
            (label_x, label_y),
            (label_x + text_w + 8, label_y + text_h + 8)
        ],
        fill=color
    )

    draw.text(
        (label_x + 4, label_y + 4),
        label,
        fill="black",
        font=font
    )

    results.append({
        "face_number": face_number,
        "name": name,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "face_distance": (
            round(distance, 4) if distance is not None else None
        ),
        "similarity_estimate_percent": (
            round(similarity, 2) if similarity is not None else None
        )
    })


# --------------------------------------------------
# 8. DISPLAY RESULTS
# --------------------------------------------------

recognized_count = sum(
    row["name"] != "Unknown" for row in results
)
unknown_count = len(results) - recognized_count

col1, col2, col3 = st.columns(3)

col1.metric("Faces detected", len(results))
col2.metric("Recognized", recognized_count)
col3.metric("Unknown", unknown_count)

st.subheader("Annotated photo")

st.image(
    annotated_image,
    use_container_width=True
)

if not results:
    st.warning("No faces detected. Try a clearer photo.")

# --------------------------------------------------
# 9. TABLE AND DOWNLOADS
# --------------------------------------------------

st.subheader("Face coordinates and recognition details")

if results:

    results_df = pd.DataFrame(results)

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )

    csv_bytes = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download results CSV",
        data=csv_bytes,
        file_name="face_recognition_results.csv",
        mime="text/csv"
    )

    image_buffer = BytesIO()
    annotated_image.save(image_buffer, format="JPEG")

    st.download_button(
        "Download annotated photo",
        data=image_buffer.getvalue(),
        file_name="annotated_faces.jpg",
        mime="image/jpeg"
    )

st.caption(
    "Similarity is an approximate distance-based score, not a calibrated "
    "probability. Incorrect identification is possible. Unknown faces are "
    "not proof that a person is absent from the gallery."
)
