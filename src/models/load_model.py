import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import os


# Modelos YOLOv8 pre-entrenados disponibles
# Para futbol, usaremos modelos YOLOv8 generales entrenados en COCO
# que funcionan bien para detección de personas y objetos deportivos
AVAILABLE_MODELS = {
    "yolov8n": "yolov8n.pt",  # Nano - más rápido
    "yolov8s": "yolov8s.pt",  # Small - equilibrado
    "yolov8m": "yolov8m.pt",  # Medium - más preciso
    "yolov8l": "yolov8l.pt",  # Large - muy preciso
    "yolov8x": "yolov8x.pt",  # Extra Large - máxima precisión
}

# IDs de Roboflow Universe para usar con inference API (alternativa)
ROBOFLOW_MODELS_INFO = {
    "player": {
        "model_id": "football-players-detection-3zvbc/10",
        "description": "Detección de jugadores y árbitros"
    },
    "ball": {
        "model_id": "soccer-ball-detection-x5dlk/3",
        "description": "Detección de pelota de fútbol"
    }
}


@st.cache_resource
def load_roboflow_model(model_type: str = "yolov8m"):
    """
    Carga modelos YOLO para detección de jugadores y objetos deportivos.

    Para esta versión, usamos modelos YOLOv8 pre-entrenados en COCO
    que incluyen clases para 'person' (jugadores) y 'sports ball' (pelota).

    Args:
        model_type: Tipo de modelo YOLO ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x')

    Returns:
        Modelo YOLO cargado

    Note:
        - Clase 0: person (jugadores, árbitros)
        - Clase 32: sports ball (pelota)
    """
    if model_type not in AVAILABLE_MODELS:
        print(f"⚠️ Modelo '{model_type}' no encontrado, usando 'yolov8m' por defecto")
        model_type = "yolov8m"

    model_name = AVAILABLE_MODELS[model_type]
    print(f"Cargando modelo {model_name}...")

    # YOLO descargará automáticamente el modelo si no existe
    return YOLO(model_name)


@st.cache_resource
def load_model(name: str):
    """Carga y cachea el modelo YOLO indicado por nombre (legacy support)."""
    return YOLO(name)