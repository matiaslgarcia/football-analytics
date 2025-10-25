from typing import Tuple, Optional
from pathlib import Path
import streamlit as st


def source_selector(videos_dir: Path) -> Tuple[str, Optional[object], Optional[Path]]:
    """Selector de origen de video: subida de archivo o videos locales de SoccerNet."""
    source_mode = st.radio(
        "Origen del video", options=["Subir archivo", "SoccerNet local"], horizontal=True
    )

    uploaded_file = None
    selected_soccernet_path = None

    if source_mode == "Subir archivo":
        uploaded_file = st.file_uploader(
            "Sube tu video", type=["mp4", "mov", "avi", "mkv"]
        )
        if uploaded_file is not None:
            st.video(uploaded_file)
    else:
        videos_dir.mkdir(exist_ok=True)
        candidates = [
            p
            for p in videos_dir.glob("**/*")
            if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")
        ]
        if len(candidates) == 0:
            st.info(
                "No se encontraron videos en 'videos/'. Copia allí un .mkv/.mp4 de SoccerNet."
            )
        else:
            selected_soccernet_path = st.selectbox(
                "Selecciona video SoccerNet", options=candidates, format_func=lambda p: p.name
            )
            if selected_soccernet_path:
                st.video(str(selected_soccernet_path))

    return source_mode, uploaded_file, selected_soccernet_path