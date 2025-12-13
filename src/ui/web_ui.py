"""Streamlit web UI for image analysis results."""

import streamlit as st
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.utils.thumbnails import ThumbnailGenerator


def load_data(json_path: str) -> list:
    """Load analysis JSON data."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    st.set_page_config(
        page_title="Nivo Image Analysis",
        page_icon="🖼️",
        layout="wide",
    )

    st.title("🖼️ Nivo Image Analysis")

    # Sidebar: Data selection
    with st.sidebar:
        st.header("Data Source")

        # File input
        data_file = st.text_input(
            "Analysis JSON",
            value="batch1_tags.json",
            help="Path to analysis results with tags",
        )

        if not Path(data_file).exists():
            st.error(f"File not found: {data_file}")
            st.stop()

        # Load data
        with st.spinner("Loading data..."):
            data = load_data(data_file)
            st.success(f"Loaded {len(data)} images")

        # Filters
        st.header("Filters")

        # Tag filter
        all_tags = set()
        for img in data:
            all_tags.update(img.get("flat_tags", []))

        selected_tags = st.multiselect(
            "Tags",
            options=sorted(all_tags),
            help="Filter by tags (AND logic)",
        )

        # Face filter
        face_filter = st.selectbox(
            "Faces",
            options=["All", "With Faces", "No Faces"],
        )

        # Quality filter
        quality_min = st.slider(
            "Min Quality",
            min_value=0,
            max_value=100,
            value=0,
            help="Filter by quality score",
        )

    # Apply filters
    filtered_data = data

    if selected_tags:
        filtered_data = [
            img for img in filtered_data
            if all(tag in img.get("flat_tags", []) for tag in selected_tags)
        ]

    if face_filter == "With Faces":
        filtered_data = [
            img for img in filtered_data
            if img.get("face_count", 0) > 0
        ]
    elif face_filter == "No Faces":
        filtered_data = [
            img for img in filtered_data
            if img.get("face_count", 0) == 0
        ]

    if quality_min > 0:
        filtered_data = [
            img for img in filtered_data
            if img.get("quality_score", 0) >= quality_min
        ]

    # Main area: Image grid
    st.header(f"Images ({len(filtered_data)})")

    if not filtered_data:
        st.warning("No images match the filters")
        st.stop()

    # Generate thumbnails
    with st.spinner("Generating thumbnails..."):
        thumb_gen = ThumbnailGenerator(size=(256, 256))
        image_paths = [img["file_path"] for img in filtered_data if "file_path" in img]
        thumbnails = thumb_gen.generate_batch(image_paths, show_progress=False)

    # Display images in grid (3 columns)
    cols_per_row = 3
    for i in range(0, len(filtered_data), cols_per_row):
        cols = st.columns(cols_per_row)

        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(filtered_data):
                break

            img_data = filtered_data[idx]
            file_path = img_data.get("file_path", "")
            filename = Path(file_path).name

            with col:
                # Thumbnail
                thumb_path = thumbnails.get(file_path)
                if thumb_path and thumb_path.exists():
                    st.image(str(thumb_path), use_container_width=True)
                else:
                    st.info("No preview")

                # Filename
                st.caption(f"**{filename}**")

                # Metadata in expander
                with st.expander("Details"):
                    # Quality
                    quality = img_data.get("quality_score", 0)
                    st.metric("Quality", f"{quality:.0f}/100")

                    # Scene
                    scene = img_data.get("primary_scene", "unknown")
                    st.text(f"Scene: {scene}")

                    # Faces
                    face_count = img_data.get("face_count", 0)
                    if face_count > 0:
                        st.text(f"👤 {face_count} face(s)")

                    # Tags (top 5)
                    tags = img_data.get("flat_tags", [])
                    if tags:
                        st.text("Tags:")
                        st.text(", ".join(tags[:5]))
                        if len(tags) > 5:
                            st.text(f"... +{len(tags) - 5} more")

                    # Dimensions
                    width = img_data.get("width")
                    height = img_data.get("height")
                    if width and height:
                        st.text(f"Size: {width}x{height}")


if __name__ == "__main__":
    main()
