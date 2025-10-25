import streamlit as st
import numpy as np
import supervision as sv
from ultralytics import YOLO
from pathlib import Path
import os
import cv2
from SoccerNet.Downloader import SoccerNetDownloader

st.set_page_config(page_title="Detección y tracking de jugadores", layout="centered")

st.title("Detección y tracking de jugadores (YOLOv8 + Supervision)")
st.caption("Sube un video de fútbol, detecta jugadores y realiza tracking por ID.")

# Controles en la barra lateral
model_name = st.sidebar.selectbox("Modelo YOLOv8", options=["yolov8n.pt", "yolov8s.pt"], index=0)
conf = st.sidebar.slider("Umbral de confianza", 0.1, 0.9, 0.25, 0.05)
only_person = st.sidebar.checkbox("Solo personas", value=True)
img_size = st.sidebar.selectbox("Tamaño de imagen", options=[640, 720, 960], index=0)

st.sidebar.markdown("---")
segment_mode = st.sidebar.checkbox("Procesar solo un segmento", value=False, help="Procesa sólo un tramo del video.")
start_s = st.sidebar.number_input("Inicio (seg)", min_value=0.0, value=0.0, step=1.0, format="%.1f")
duration_s = st.sidebar.number_input("Duración (seg)", min_value=1.0, value=10.0, step=1.0, format="%.1f")

# NUEVO: Selector de origen de video (Subir archivo o SoccerNet local)
source_mode = st.radio("Origen del video", options=["Subir archivo", "SoccerNet local"], horizontal=True)

uploaded_file = None
selected_soccernet_path = None

if source_mode == "Subir archivo":
    uploaded_file = st.file_uploader("Sube tu video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_file is not None:
        st.video(uploaded_file)
else:
    videos_dir = Path("videos"); videos_dir.mkdir(exist_ok=True)
    candidates = [p for p in videos_dir.glob("**/*") if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")]
    if len(candidates) == 0:
        st.info("No se encontraron videos en 'videos/'. Copia allí un .mkv/.mp4 de SoccerNet.")
    else:
        selected_soccernet_path = st.selectbox("Selecciona video SoccerNet", options=candidates, format_func=lambda p: p.name)
        if selected_soccernet_path:
            st.video(str(selected_soccernet_path))

@st.cache_resource
def load_model(name: str):
    model = YOLO(name)
    # pequeña optimización
    try:
        model.fuse()
    except Exception:
        pass
    return model


def process_video(source_path: str, target_path: str, model: YOLO, conf: float, only_person: bool, img_size: int):
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    names = getattr(model.model, "names", None)

    def callback(frame: np.ndarray, _: int) -> np.ndarray:
        results = model(frame, conf=conf, imgsz=img_size)[0]
        detections = sv.Detections.from_ultralytics(results)

        if only_person:
            # Determinar el id de clase para "person" (COCO)
            person_id = 0
            if names and isinstance(names, dict):
                for k, v in names.items():
                    if v == "person":
                        person_id = k
                        break
            mask = detections.class_id == person_id
            detections = detections[mask]

        # Actualizar tracking
        detections = tracker.update_with_detections(detections)

        # Construir etiquetas con tracker ID y nombre de clase
        class_names = None
        try:
            class_names = detections.data.get("class_name", None)
        except Exception:
            class_names = None
        if class_names is None and names and isinstance(names, dict):
            class_names = [names.get(int(cid), str(cid)) for cid in detections.class_id]
        if class_names is None:
            class_names = [str(cid) for cid in detections.class_id]

        labels = []
        for class_name, tracker_id in zip(class_names, detections.tracker_id):
            if tracker_id is None:
                labels.append(f"{class_name}")
            else:
                labels.append(f"#{tracker_id} {class_name}")

        annotated = box_annotator.annotate(frame.copy(), detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        return annotated

    sv.process_video(source_path=source_path, target_path=target_path, callback=callback)


def process_video_segment(source_path: str, target_path: str, model: YOLO, conf: float, only_person: bool, img_size: int, start_s: float, duration_s: float):
    import math
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {source_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0 else math.inf

    start_frame = max(0, int(start_s * fps))
    max_frames = int(duration_s * fps)
    end_frame = start_frame + max_frames - 1 if max_frames > 0 else total_frames - 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(target_path, fourcc, fps, (width, height))

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    names = getattr(model.model, "names", None)

    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, imgsz=img_size)[0]
        detections = sv.Detections.from_ultralytics(results)

        if only_person:
            person_id = 0
            if names and isinstance(names, dict):
                for k, v in names.items():
                    if v == "person":
                        person_id = k
                        break
            mask = detections.class_id == person_id
            detections = detections[mask]

        detections = tracker.update_with_detections(detections)

        class_names = None
        try:
            class_names = detections.data.get("class_name", None)
        except Exception:
            class_names = None
        if class_names is None and names and isinstance(names, dict):
            class_names = [names.get(int(cid), str(cid)) for cid in detections.class_id]
        if class_names is None:
            class_names = [str(cid) for cid in detections.class_id]

        labels = []
        for class_name, tracker_id in zip(class_names, detections.tracker_id):
            if tracker_id is None:
                labels.append(f"{class_name}")
            else:
                labels.append(f"#{tracker_id} {class_name}")

        annotated = box_annotator.annotate(frame.copy(), detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        writer.write(annotated)

        current_frame += 1

    cap.release()
    writer.release()


def clip_video_simple(source_path: str, target_path: str, start_s: float, duration_s: float):
    import math
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {source_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0 else math.inf

    start_frame = max(0, int(start_s * fps))
    max_frames = int(duration_s * fps)
    end_frame = start_frame + max_frames - 1 if max_frames > 0 else total_frames - 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(target_path, fourcc, fps, (width, height))

    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        current_frame += 1

    cap.release()
    writer.release()


inputs_dir = Path("inputs"); inputs_dir.mkdir(exist_ok=True)
outputs_dir = Path("outputs"); outputs_dir.mkdir(exist_ok=True)

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

with st.expander("Cómo usar SoccerNet en esta app"):
    st.markdown(
        "- Coloca los videos de SoccerNet en la carpeta `videos/`.\n"
        "- Los videos del dataset son 25 fps y resolución 720p o 224p.\n"
        "- Para descargar los videos oficiales, primero completa el NDA y sigue las instrucciones del sitio de SoccerNet: https://www.soccer-net.org/data\n"
    )

with st.expander("Descargar video de SoccerNet"):
    st.markdown("Ingresa la información del juego para descargar el video al directorio `videos/`.")
    default_game = "europe_uefa-champions-league/2016-2017/2017-04-18 - 21-45 Real Madrid 4 - 2 Bayern Munich"
    game_path = st.text_input("Ruta del juego (league/season/game)", value=default_game)
    quality_download = st.selectbox("Calidad", options=["720p", "224p"], index=0)
    half_choice = st.selectbox("Mitad", options=["both", "1", "2"], index=0)
    password_download = st.text_input("Password (NDA)", type="password", help="Requerido para descargar videos del dataset.")

    st.markdown("---")
    recortar_download = st.checkbox("Recortar tras descarga", value=True, help="Crear un clip MP4 del tramo indicado por cada mitad descargada.")
    start_dl_s = st.number_input("Inicio clip (seg)", min_value=0.0, value=0.0, step=1.0, format="%.1f")
    duration_dl_s = st.number_input("Duración clip (seg)", min_value=1.0, value=10.0, step=1.0, format="%.1f")

    if st.button("Descargar a videos/"):
        videos_dir = Path("videos"); videos_dir.mkdir(exist_ok=True)
        files = []
        if half_choice in ("1", "both"):
            files.append("1_720p.mkv" if quality_download == "720p" else "1_224p.mkv")
        if half_choice in ("2", "both"):
            files.append("2_720p.mkv" if quality_download == "720p" else "2_224p.mkv")

        if not password_download:
            # fallback a variables de entorno
            password_download = os.getenv("SOCCERNET_PASSWORD") or os.getenv("SOCCERNET_PW")
        if not password_download:
            st.error("Debes ingresar el password del NDA para descargar videos.")
        else:
            try:
                with st.spinner("Descargando desde SoccerNet, puede tardar..."):
                    downloader = SoccerNetDownloader(LocalDirectory=str(videos_dir))
                    downloader.password = password_download
                    downloader.downloadGame(files=files, game=game_path)

                clips_created = []
                if recortar_download:
                    game_dir = Path(videos_dir) / Path(game_path)
                    for fname in files:
                        src_path = game_dir / fname
                        if src_path.exists():
                            base = Path(fname).stem
                            clip_name = f"{base}_clip_{int(start_dl_s)}-{int(duration_dl_s)}.mp4"
                            dst_path = game_dir / clip_name
                            try:
                                clip_video_simple(str(src_path), str(dst_path), float(start_dl_s), float(duration_dl_s))
                                clips_created.append(dst_path)
                            except Exception as e:
                                st.warning(f"No se pudo recortar {fname}: {e}")

                if recortar_download and len(clips_created) > 0:
                    st.success(f"Descarga y recorte completados. Clips creados: {len(clips_created)}")
                else:
                    st.success("Descarga completada. Los archivos están en `videos/`.")

                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error al descargar: {e}")