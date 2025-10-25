import cv2
import numpy as np
import supervision as sv


def process_video(source_path: str, target_path: str, model, conf: float, only_person: bool, img_size: int):
    """Procesa video completo con detección y tracking y guarda un MP4 sin audio."""
    source_path = str(source_path)
    target_path = str(target_path)

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se puede abrir el video fuente: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(target_path, fourcc, fps, (width, height))

    tracker = sv.ByteTrack()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame, conf=conf, imgsz=img_size)
            r = results[0]
            boxes = r.boxes
            xyxy = boxes.xyxy.cpu().numpy() if boxes is not None else np.empty((0, 4))
            confs = boxes.conf.cpu().numpy() if boxes is not None else np.empty((0,))
            class_ids = boxes.cls.cpu().numpy().astype(int) if boxes is not None else np.empty((0,), dtype=int)
            dets = sv.Detections(xyxy=xyxy, confidence=confs, class_id=class_ids)
            if only_person:
                mask = dets.class_id == 0
                dets = dets[mask]
            tracked = tracker.update_with_detections(dets)
            annotated = sv.BoxAnnotator().annotate(scene=frame.copy(), detections=tracked)
            out.write(annotated)
    finally:
        cap.release()
        out.release()