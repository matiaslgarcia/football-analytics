import cv2


def clip_video_simple(source_path: str, target_path: str, start_s: float, duration_s: float):
    """Extrae un segmento de video y lo guarda como MP4 sin audio."""
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se puede abrir el video fuente: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(target_path), fourcc, fps, (width, height))

    start_frame = max(0, int(start_s * fps))
    total_frames = int(duration_s * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    written = 0
    try:
        while written < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            written += 1
    finally:
        cap.release()
        out.release()