from typing import Tuple
import streamlit as st


def download_controls(default_game: str) -> Tuple[str, str, str, str, bool, float, float, bool]:
    """Renderiza controles de descarga de SoccerNet y retorna los valores junto al click del botón."""
    with st.expander("Descargar video de SoccerNet"):
        st.markdown(
            "Ingresa la información del juego para descargar el video al directorio `videos/`."
        )
        game_path = st.text_input("Ruta del juego (league/season/game)", value=default_game)
        quality_download = st.selectbox("Calidad", options=["720p", "224p"], index=0)
        half_choice = st.selectbox("Mitad", options=["both", "1", "2"], index=0)
        password_download = st.text_input(
            "Password (NDA)", type="password", help="Requerido para descargar videos del dataset."
        )

        st.markdown("---")
        recortar_download = st.checkbox(
            "Recortar tras descarga",
            value=True,
            help="Crear un clip MP4 del tramo indicado por cada mitad descargada.",
        )
        start_dl_s = st.number_input(
            "Inicio clip (seg)", min_value=0.0, value=0.0, step=1.0, format="%.1f"
        )
        duration_dl_s = st.number_input(
            "Duración clip (seg)", min_value=1.0, value=10.0, step=1.0, format="%.1f"
        )
        clicked = st.button("Descargar a videos/")

    return (
        game_path,
        quality_download,
        half_choice,
        password_download,
        recortar_download,
        float(start_dl_s),
        float(duration_dl_s),
        clicked,
    )