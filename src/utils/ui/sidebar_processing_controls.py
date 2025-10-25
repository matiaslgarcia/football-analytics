from typing import Tuple
import streamlit as st


def sidebar_processing_controls() -> Tuple[str, float, bool, int, bool, float, float]:
    """Renderiza los controles de procesamiento en la barra lateral y retorna sus valores."""
    model_name = st.sidebar.selectbox("Modelo YOLOv8", options=["yolov8n.pt", "yolov8s.pt"], index=0)
    conf = st.sidebar.slider("Umbral de confianza", 0.1, 0.9, 0.25, 0.05)
    only_person = st.sidebar.checkbox("Solo personas", value=True)
    img_size = st.sidebar.selectbox("Tamaño de imagen", options=[640, 720, 960], index=0)

    st.sidebar.markdown("---")
    segment_mode = st.sidebar.checkbox(
        "Procesar solo un segmento",
        value=False,
        help="Procesa sólo un tramo del video.",
    )
    start_s = st.sidebar.number_input(
        "Inicio (seg)", min_value=0.0, value=0.0, step=1.0, format="%.1f"
    )
    duration_s = st.sidebar.number_input(
        "Duración (seg)", min_value=1.0, value=10.0, step=1.0, format="%.1f"
    )
    return model_name, conf, only_person, img_size, segment_mode, float(start_s), float(duration_s)