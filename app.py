import streamlit as st
from pathlib import Path
from src.models.load_model import load_roboflow_model
from src.controllers.process_video import process_video
from src.utils.config import INPUTS_DIR, OUTPUTS_DIR
from ultralytics import YOLO

st.set_page_config(page_title="Soccer Analytics AI", layout="wide")

st.title("⚽ Soccer Analytics AI")
st.caption("Detección de jugadores, equipos, pelota y proyección a radar 2D.")

# === SIDEBAR CONFIGURATION ===
st.sidebar.header("⚙️ Configuración")

# 1. PLAYERS MODEL
st.sidebar.subheader("1. Jugadores")
player_source = st.sidebar.radio(
    "Modelo de Jugadores",
    ["YOLOv8 Genérico (COCO)", "Subir Modelo Custom (.pt)"],
    index=0
)

player_model = None
if player_source == "YOLOv8 Genérico (COCO)":
    # Default to Medium for balance
    model_size = st.sidebar.selectbox("Tamaño del modelo", ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"], index=2)
    with st.spinner(f"Cargando {model_size}..."):
        player_model = load_roboflow_model(model_size)
else:
    uploaded_player = st.sidebar.file_uploader("Subir modelo jugadores (.pt)", type=["pt"])
    if uploaded_player:
        p_path = Path("models") / uploaded_player.name
        p_path.parent.mkdir(exist_ok=True)
        with open(p_path, "wb") as f:
            f.write(uploaded_player.read())
        player_model = YOLO(p_path)
        st.sidebar.success(f"Cargado: {uploaded_player.name}")

# 2. BALL MODEL
st.sidebar.subheader("2. Pelota")
ball_source = st.sidebar.radio(
    "Modelo de Pelota",
    ["Heurística (Clase 'sports ball')", "Subir Modelo Custom (.pt)"],
    index=0
)

ball_model = None
if ball_source == "Subir Modelo Custom (.pt)":
    uploaded_ball = st.sidebar.file_uploader("Subir modelo pelota (.pt)", type=["pt"])
    if uploaded_ball:
        b_path = Path("models") / uploaded_ball.name
        b_path.parent.mkdir(exist_ok=True)
        with open(b_path, "wb") as f:
            f.write(uploaded_ball.read())
        ball_model = YOLO(b_path)
        st.sidebar.success(f"Cargado: {uploaded_ball.name}")

# 3. RADAR / PITCH
st.sidebar.subheader("3. Radar View")
enable_radar = st.sidebar.checkbox("Habilitar Radar", value=False)
pitch_model = None
full_field_approx = False

if enable_radar:
    pitch_source = st.sidebar.radio(
        "Modelo de Campo",
        ["Subir Modelo Custom (.pt)", "Aproximación Pantalla Completa (Experimental)"],
        index=0
    )
    
    if pitch_source == "Subir Modelo Custom (.pt)":
        uploaded_pitch = st.sidebar.file_uploader("Subir modelo campo (.pt)", type=["pt"])
        if uploaded_pitch:
            pi_path = Path("models") / uploaded_pitch.name
            pi_path.parent.mkdir(exist_ok=True)
            with open(pi_path, "wb") as f:
                f.write(uploaded_pitch.read())
            pitch_model = YOLO(pi_path)
            st.sidebar.success(f"Cargado: {uploaded_pitch.name}")
        else:
            st.sidebar.warning("Sin modelo de campo, el radar no funcionará.")
    else:
        st.sidebar.info("ℹ️ Asume que los 4 bordes del video coinciden con los 4 bordes del campo.")
        full_field_approx = True

# === MAIN AREA ===
col1, col2 = st.columns(2)

with col1:
    st.subheader("📺 Video de Entrada")
    uploaded_video = st.file_uploader("Arrastra un video aquí", type=["mp4", "mov", "avi"])

if uploaded_video:
    # Save input video
    input_path = INPUTS_DIR / uploaded_video.name
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(input_path, "wb") as f:
        f.write(uploaded_video.read())
    
    with col1:
        st.video(str(input_path))

    # Process Button
    if st.button("🚀 Iniciar Procesamiento", type="primary"):
        if player_model is None:
            st.error("❌ Debes cargar un modelo de jugadores (o usar el genérico).")
        else:
            with col2:
                st.subheader("⚙️ Procesando...")
                status_text = st.empty()
                progress_bar = st.progress(0)
                
                output_filename = f"processed_{uploaded_video.name}"
                target_path = OUTPUTS_DIR / output_filename
                OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

                try:
                    status_text.text("Iniciando motor de IA...")
                    
                    # Pass full_field_approx to process_video (we need to update process_video signature or handle it)
                    # To keep it clean, we'll pass a special flag or handle it inside process_video
                    # Let's modify process_video to accept a config dict or specific arg
                    
                    process_video(
                        source_path=str(input_path),
                        target_path=str(target_path),
                        player_model=player_model,
                        ball_model=ball_model,
                        pitch_model=pitch_model,
                        conf=0.3,
                        detection_mode="players_and_ball",
                        full_field_approx=full_field_approx  # We will add this argument
                    )
                    
                    progress_bar.progress(100)
                    status_text.success("✅ ¡Procesamiento completado!")
                    st.video(str(target_path))
                    
                    with open(target_path, "rb") as f:
                        st.download_button(
                            "⬇️ Descargar Video Procesado",
                            f,
                            file_name=output_filename
                        )
                        
                except Exception as e:
                    st.error(f"Ocurrió un error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

else:
    st.info("👈 Sube un video para comenzar.")
