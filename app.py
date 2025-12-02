"""
Soccer Analytics AI - Versión con Estadísticas Tácticas
Interfaz Streamlit mejorada con análisis táctico completo
"""

import streamlit as st
from pathlib import Path
from src.models.load_model import load_roboflow_model
from src.controllers.process_video import process_video
from src.utils.config import INPUTS_DIR, OUTPUTS_DIR
from ultralytics import YOLO
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Soccer Analytics AI", layout="wide", initial_sidebar_state="expanded")

st.title("⚽ Soccer Analytics AI")
st.caption("Análisis táctico completo: Tracking, Formaciones y Métricas de Comportamiento Colectivo")

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
enable_radar = st.sidebar.checkbox("Habilitar Radar", value=True)
enable_analytics = st.sidebar.checkbox("Habilitar Análisis Táctico", value=True,
                                       help="Calcula formaciones y métricas tácticas")

pitch_model = None
full_field_approx = False

if enable_radar:
    pitch_source = st.sidebar.radio(
        "Modelo de Campo",
        [
            "Homography.pt (32 keypoints) - ⭐ Recomendado",
            "Soccana Keypoint (29 keypoints)",
            "Aproximación Pantalla Completa (Experimental)"
        ],
        index=0,
        help="Homography.pt detecta más keypoints y tiene mayor tasa de éxito"
    )

    if pitch_source == "Homography.pt (32 keypoints) - ⭐ Recomendado":
        homography_path = Path("models/homography.pt")
        if homography_path.exists():
            with st.spinner("Cargando modelo Homography (32 keypoints)..."):
                pitch_model = YOLO(str(homography_path))
            st.sidebar.success("✅ Modelo Homography cargado (100% tasa éxito)")
        else:
            st.sidebar.error(f"❌ Modelo no encontrado: {homography_path}")
            st.sidebar.info("Asegúrate de tener el archivo models/homography.pt")

    elif pitch_source == "Soccana Keypoint (29 keypoints)":
        soccana_path = Path("models/soccana_keypoint/Model/weights/best.pt")
        if soccana_path.exists():
            with st.spinner("Cargando modelo Soccana_Keypoint (29 keypoints)..."):
                pitch_model = YOLO(str(soccana_path))
            st.sidebar.success("✅ Modelo Soccana cargado (~65% tasa éxito)")
        else:
            st.sidebar.error(f"❌ Modelo no encontrado")
            st.sidebar.info("Descarga con: python scripts/download_soccana_model.py")

    else:
        full_field_approx = True

# === MAIN AREA ===
# Initialize session state for stats
if 'stats' not in st.session_state:
    st.session_state.stats = None
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False

uploaded_video = st.file_uploader("📹 Arrastra un video aquí", type=["mp4", "mov", "avi"])

if uploaded_video:
    # Save input video
    input_path = INPUTS_DIR / uploaded_video.name
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(input_path, "wb") as f:
        f.write(uploaded_video.read())

    # Tabs for organization
    tabs = st.tabs(["🎬 Video", "📊 Estadísticas", "📈 Gráficos", "💾 Exportar"])

    with tabs[0]:
        col_input, col_output = st.columns(2)

        with col_input:
            st.subheader("Video Original")
            st.video(str(input_path))

        with col_output:
            if st.session_state.video_processed:
                st.subheader("Video Procesado")
                output_filename = f"processed_{uploaded_video.name}"
                target_path = OUTPUTS_DIR / output_filename
                if target_path.exists():
                    st.video(str(target_path))

                    with open(target_path, "rb") as f:
                        st.download_button(
                            "⬇️ Descargar Video",
                            f,
                            file_name=output_filename,
                            mime="video/mp4"
                        )
            else:
                st.info("👉 Haz clic en 'Procesar Video' para iniciar")

    # Process Button
    if st.button("🚀 Procesar Video", type="primary", width="stretch"):
        if player_model is None:
            st.error("❌ Debes cargar un modelo de jugadores")
        else:
            with st.spinner("Procesando video..."):
                status_placeholder = st.empty()
                progress_bar = st.progress(0)

                output_filename = f"processed_{uploaded_video.name}"
                target_path = OUTPUTS_DIR / output_filename
                OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

                try:
                    status_placeholder.text("🎯 Iniciando análisis...")
                    progress_bar.progress(10)

                    # Process video
                    process_video(
                        source_path=str(input_path),
                        target_path=str(target_path),
                        player_model=player_model,
                        ball_model=ball_model,
                        pitch_model=pitch_model,
                        conf=0.3,
                        detection_mode="players_and_ball",
                        full_field_approx=full_field_approx
                    )

                    progress_bar.progress(90)
                    status_placeholder.text("📊 Generando estadísticas...")

                    # Check if stats file exists
                    stats_path = target_path.parent / f"{target_path.stem}_stats.json"
                    if stats_path.exists():
                        with open(stats_path, 'r') as f:
                            st.session_state.stats = json.load(f)

                    progress_bar.progress(100)
                    status_placeholder.success("✅ ¡Procesamiento completado!")
                    st.session_state.video_processed = True
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    with st.expander("Ver detalles del error"):
                        st.code(traceback.format_exc())

    # Statistics Tab
    with tabs[1]:
        if st.session_state.stats:
            stats = st.session_state.stats

            st.subheader("📊 Análisis Táctico")

            # Summary metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Duración", f"{stats.get('duration_seconds', 0):.1f}s")
            with col2:
                st.metric("Frames Procesados", stats.get('total_frames', 0))
            with col3:
                fps = stats.get('total_frames', 0) / stats.get('duration_seconds', 1)
                st.metric("FPS", f"{fps:.1f}")

            st.divider()

            # Formations
            if 'formations' in stats:
                st.subheader("⚽ Formaciones Detectadas")

                col_t1, col_t2 = st.columns(2)

                with col_t1:
                    st.markdown("### 🟢 Team 1")
                    form1 = stats['formations'].get('team1', {}).get('most_common', 'N/A')
                    st.metric("Formación más común", form1, help="Basada en análisis temporal")

                with col_t2:
                    st.markdown("### 🔵 Team 2")
                    form2 = stats['formations'].get('team2', {}).get('most_common', 'N/A')
                    st.metric("Formación más común", form2, help="Basada en análisis temporal")

            st.divider()

            # Tactical Metrics
            if 'metrics' in stats:
                st.subheader("📈 Métricas Tácticas")

                metrics1 = stats['metrics'].get('team1', {})
                metrics2 = stats['metrics'].get('team2', {})

                # Comparison table
                comparison_data = {
                    'Métrica': ['Presión (m)', 'Amplitud (m)', 'Compactación (m²)'],
                    'Team 1': [
                        f"{metrics1.get('pressure_height', {}).get('mean', 0):.1f}",
                        f"{metrics1.get('offensive_width', {}).get('mean', 0):.1f}",
                        f"{metrics1.get('compactness', {}).get('mean', 0):.0f}"
                    ],
                    'Team 2': [
                        f"{metrics2.get('pressure_height', {}).get('mean', 0):.1f}",
                        f"{metrics2.get('offensive_width', {}).get('mean', 0):.1f}",
                        f"{metrics2.get('compactness', {}).get('mean', 0):.0f}"
                    ]
                }

                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, width="stretch", hide_index=True)

                # Detailed metrics
                with st.expander("Ver métricas detalladas"):
                    col_det1, col_det2 = st.columns(2)

                    with col_det1:
                        st.markdown("#### Team 1")
                        if metrics1:
                            for metric_name, values in metrics1.items():
                                if isinstance(values, dict) and 'mean' in values:
                                    st.text(f"{metric_name}:")
                                    st.text(f"  Media: {values['mean']:.2f}")
                                    st.text(f"  Min-Max: {values['min']:.2f} - {values['max']:.2f}")

                    with col_det2:
                        st.markdown("#### Team 2")
                        if metrics2:
                            for metric_name, values in metrics2.items():
                                if isinstance(values, dict) and 'mean' in values:
                                    st.text(f"{metric_name}:")
                                    st.text(f"  Media: {values['mean']:.2f}")
                                    st.text(f"  Min-Max: {values['min']:.2f} - {values['max']:.2f}")

        else:
            st.info("📊 Las estadísticas aparecerán aquí después de procesar el video")

    # Charts Tab
    with tabs[2]:
        if st.session_state.stats and 'timeline' in st.session_state.stats:
            st.subheader("📈 Evolución Temporal")

            timeline = st.session_state.stats['timeline']

            # Pressure Height Chart
            if 'team1' in timeline and 'pressure_height' in timeline['team1']:
                frames1 = timeline['team1'].get('frame_number', [])
                pressure1 = timeline['team1'].get('pressure_height', [])

                frames2 = timeline.get('team2', {}).get('frame_number', [])
                pressure2 = timeline.get('team2', {}).get('pressure_height', [])

                fig_pressure = go.Figure()
                fig_pressure.add_trace(go.Scatter(x=frames1, y=pressure1, name='Team 1', line=dict(color='green')))
                fig_pressure.add_trace(go.Scatter(x=frames2, y=pressure2, name='Team 2', line=dict(color='blue')))
                fig_pressure.update_layout(
                    title='Altura de Presión (m)',
                    xaxis_title='Frame',
                    yaxis_title='Presión (m)',
                    hovermode='x unified'
                )
                st.plotly_chart(fig_pressure, use_container_width=True)

            # Compactness Chart
            if 'team1' in timeline and 'compactness' in timeline['team1']:
                compact1 = timeline['team1'].get('compactness', [])
                compact2 = timeline.get('team2', {}).get('compactness', [])

                fig_compact = go.Figure()
                fig_compact.add_trace(go.Scatter(x=frames1, y=compact1, name='Team 1', line=dict(color='green')))
                fig_compact.add_trace(go.Scatter(x=frames2, y=compact2, name='Team 2', line=dict(color='blue')))
                fig_compact.update_layout(
                    title='Compactación (m²)',
                    xaxis_title='Frame',
                    yaxis_title='Área (m²)',
                    hovermode='x unified'
                )
                st.plotly_chart(fig_compact, use_container_width=True)

            # Width Chart
            if 'team1' in timeline and 'offensive_width' in timeline['team1']:
                width1 = timeline['team1'].get('offensive_width', [])
                width2 = timeline.get('team2', {}).get('offensive_width', [])

                fig_width = go.Figure()
                fig_width.add_trace(go.Scatter(x=frames1, y=width1, name='Team 1', line=dict(color='green')))
                fig_width.add_trace(go.Scatter(x=frames2, y=width2, name='Team 2', line=dict(color='blue')))
                fig_width.update_layout(
                    title='Amplitud Ofensiva (m)',
                    xaxis_title='Frame',
                    yaxis_title='Amplitud (m)',
                    hovermode='x unified'
                )
                st.plotly_chart(fig_width, use_container_width=True)

        else:
            st.info("📈 Los gráficos aparecerán aquí después de procesar el video con análisis táctico")

    # Export Tab
    with tabs[3]:
        if st.session_state.stats:
            st.subheader("💾 Exportar Datos")

            col_json, col_csv = st.columns(2)

            with col_json:
                st.markdown("### JSON")
                json_str = json.dumps(st.session_state.stats, indent=2)
                st.download_button(
                    label="⬇️ Descargar JSON",
                    data=json_str,
                    file_name="soccer_analytics_stats.json",
                    mime="application/json"
                )

            with col_csv:
                st.markdown("### CSV")
                if 'timeline' in st.session_state.stats:
                    timeline = st.session_state.stats['timeline']
                    if 'team1' in timeline:
                        df_export = pd.DataFrame(timeline['team1'])
                        csv = df_export.to_csv(index=False)
                        st.download_button(
                            label="⬇️ Descargar CSV (Team 1)",
                            data=csv,
                            file_name="team1_timeline.csv",
                            mime="text/csv"
                        )

        else:
            st.info("💾 Las opciones de exportación aparecerán aquí después de procesar el video")

else:
    st.info("👈 Sube un video para comenzar el análisis táctico")

# Footer
st.divider()
st.caption("Soccer Analytics AI - Sistema de Análisis Táctico Completo")
st.caption("Tracking + Formaciones + Métricas de Comportamiento Colectivo")
