import streamlit as st
from pathlib import Path
from src.models.load_model import load_model
from src.controllers.download_game import download_game
from src.controllers.process_video import process_video
from src.controllers.process_video_segment import process_video_segment
from src.utils.config import INPUTS_DIR, OUTPUTS_DIR, VIDEOS_DIR
from src.utils.ui.sidebar_processing_controls import sidebar_processing_controls
from src.utils.ui.source_selector import source_selector
from src.utils.ui.download_controls import download_controls

st.set_page_config(page_title="Detección y tracking de jugadores", layout="centered")

st.title("Detección y tracking de jugadores (YOLOv8 + Supervision)")
st.caption("Sube un video de fútbol, detecta jugadores y realiza tracking por ID.")

# Controles en la barra lateral
model_name, conf, only_person, img_size, segment_mode, start_s, duration_s = sidebar_processing_controls()

# NUEVO: Selector de origen de video (Subir archivo o SoccerNet local)
source_mode, uploaded_file, selected_soccernet_path = source_selector(VIDEOS_DIR)

# load_model movido a src.models.load_model


inputs_dir = INPUTS_DIR
outputs_dir = OUTPUTS_DIR

if st.button("Procesar video"):
    if source_mode == "Subir archivo":
        if uploaded_file is None:
            st.warning("Sube un archivo primero.")
        else:
            in_path = inputs_dir / uploaded_file.name
            with open(in_path, "wb") as f:
                f.write(uploaded_file.read())
    else:
        if selected_soccernet_path is None:
            st.warning("Selecciona un video de SoccerNet.")
        else:
            in_path = selected_soccernet_path

    if 'in_path' in locals():
        out_stem = Path(in_path).stem
        if segment_mode:
            out_path = outputs_dir / f"result_{out_stem}_seg_{int(start_s)}-{int(duration_s)}.mp4"
        else:
            out_path = outputs_dir / f"result_{out_stem}.mp4"

        with st.spinner("Procesando el video, esto puede tardar..."):
            model = load_model(model_name)
            if segment_mode:
                process_video_segment(
                    str(in_path), str(out_path), model,
                    conf=conf, only_person=only_person, img_size=img_size,
                    start_s=float(start_s), duration_s=float(duration_s)
                )
            else:
                process_video(str(in_path), str(out_path), model, conf=conf, only_person=only_person, img_size=img_size)

        st.success("Listo: procesamiento completado.")
        st.video(str(out_path))
        with open(out_path, "rb") as f:
            st.download_button(
                label="Descargar video anotado",
                data=f,
                file_name=Path(out_path).name,
                mime="video/mp4"
            )
    else:
        st.stop()

default_game = "europe_uefa-champions-league/2016-2017/2017-04-18 - 21-45 Real Madrid 4 - 2 Bayern Munich"
game_path, quality_download, half_choice, password_download, recortar_download, start_dl_s, duration_dl_s, clicked = download_controls(default_game)
if clicked:
    try:
        with st.spinner("Descargando desde SoccerNet, puede tardar..."):
            result = download_game(
                game_path=game_path,
                quality=quality_download,
                half_choice=half_choice,
                password=password_download or None,
                local_dir=VIDEOS_DIR,
                recortar=recortar_download,
                start_s=float(start_dl_s),
                duration_s=float(duration_dl_s),
            )
        if recortar_download and len(result["clips"]) > 0:
            st.success(f"Descarga y recorte completados. Clips creados: {len(result['clips'])}")
        else:
            st.success("Descarga completada. Los archivos están en `videos/`.")

        st.experimental_rerun()
    except Exception as e:
        st.error(f"Error al descargar: {e}")