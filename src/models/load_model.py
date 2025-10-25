import streamlit as st
from ultralytics import YOLO


@st.cache_resource
def load_model(name: str):
    """Carga y cachea el modelo YOLO indicado por nombre."""
    return YOLO(name)