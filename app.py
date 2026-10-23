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


