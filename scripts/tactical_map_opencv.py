import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


GLOBAL_FRAME_OFFSET = 0
PITCH_DIMENSIONS: Tuple[float, float] = (105.0, 68.0)


def _to_matrix_3x3(v: object) -> Optional[np.ndarray]:
    try:
        arr = np.array(v, dtype=np.float32)
        if arr.size == 9:
            return arr.reshape(3, 3)
        if arr.shape == (3, 3):
            return arr
    except Exception:
        return None
    return None


def load_calibration_map(json_path: Path) -> Dict[int, np.ndarray]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    mapping: Dict[int, np.ndarray] = {}

    def extract_h(obj: object) -> Optional[np.ndarray]:
        if isinstance(obj, (list, tuple, np.ndarray)):
            return _to_matrix_3x3(obj)
        if isinstance(obj, dict):
            for key in ("H", "homography", "matrix"):
                if key in obj:
                    m = _to_matrix_3x3(obj[key])
                    if m is not None:
                        return m
        return None

    if isinstance(data, dict):
        for k, v in data.items():
            try:
                idx = int(k)
            except Exception:
                continue
            H = extract_h(v)
            if H is not None:
                mapping[idx] = H
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "frame" in item:
                try:
                    idx = int(item["frame"])
                except Exception:
                    continue
                H = extract_h(item)
                if H is not None:
                    mapping[idx] = H
    return mapping


def get_h_for_frame(frame_idx: int, calib_map: Dict[int, np.ndarray], last_valid_h: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if frame_idx in calib_map:
        H = calib_map[frame_idx]
        return H, H
    if len(calib_map) > 0:
        keys = np.array(list(calib_map.keys()))
        nearest_key = int(keys[np.argmin(np.abs(keys - frame_idx))])
        H = calib_map[nearest_key]
        return H, last_valid_h if last_valid_h is not None else H
    return last_valid_h, last_valid_h


def _flip_matrix_x(width_m: float) -> np.ndarray:
    return np.array([[-1.0, 0.0, width_m],
                     [ 0.0, 1.0,      0.0],
                     [ 0.0, 0.0,      1.0]], dtype=np.float32)


def _flip_matrix_y(length_m: float) -> np.ndarray:
    return np.array([[ 1.0, 0.0,       0.0],
                     [ 0.0,-1.0,  length_m],
                     [ 0.0, 0.0,       1.0]], dtype=np.float32)


def choose_best_orientation(H_base: np.ndarray, pts_px: np.ndarray, field_size: Tuple[float, float], test_inverse: bool = True) -> np.ndarray:
    Wm, Lm = field_size
    candidates: List[np.ndarray] = [H_base.copy()]
    if test_inverse:
        try:
            candidates.append(np.linalg.inv(H_base))
        except Exception:
            pass

    flips = [
        np.eye(3, dtype=np.float32),
        _flip_matrix_x(Wm),
        _flip_matrix_y(Lm),
        _flip_matrix_y(Lm) @ _flip_matrix_x(Wm),
    ]

    combos: List[np.ndarray] = []
    for base in candidates:
        for F in flips:
            combos.append(F @ base)

    best_H = H_base
    best_in = -1
    for H in combos:
        ok_count = 0
        try:
            tr = cv2.perspectiveTransform(pts_px.reshape(-1, 1, 2).astype(np.float32), H)
            tr = tr.reshape(-1, 2)
            xs = tr[:, 0]
            ys = tr[:, 1]
            in_x = (xs >= 0.0) & (xs <= Wm)
            in_y = (ys >= 0.0) & (ys <= Lm)
            ok_count = int(np.sum(in_x & in_y))
        except Exception:
            ok_count = -1
        if ok_count > best_in:
            best_in = ok_count
            best_H = H
    return best_H


def transform_points_to_meters(points_px: np.ndarray, H: np.ndarray) -> np.ndarray:
    if points_px.size == 0:
        return points_px.reshape(0, 2)
    pts = points_px.reshape(-1, 1, 2).astype(np.float32)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)


def meters_to_template_pixels(points_m: np.ndarray, template_shape: Tuple[int, int], field_size: Tuple[float, float]) -> np.ndarray:
    h_px, w_px = template_shape[:2]
    Wm, Lm = field_size
    if points_m.size == 0:
        return points_m.reshape(0, 2)
    sx = w_px / Wm
    sy = h_px / Lm
    xs = points_m[:, 0] * sx
    ys = points_m[:, 1] * sy
    pts = np.column_stack([xs, ys])
    pts[:, 0] = np.clip(pts[:, 0], 0, w_px - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h_px - 1)
    return pts


def bottom_center_from_xyxy(xyxy: np.ndarray) -> np.ndarray:
    if xyxy.size == 0:
        return xyxy.reshape(0, 2)
    x1 = xyxy[:, 0]
    x2 = xyxy[:, 2]
    y2 = xyxy[:, 3]
    xc = (x1 + x2) / 2.0
    return np.column_stack([xc, y2])


def parse_clip_offset_seconds(name: str) -> Optional[float]:
    try:
        if "_clip_" in name:
            tail = name.split("_clip_")[-1]
            start_s = tail.split("-")[0]
            return float(start_s)
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser(description="Mapa táctico 2D desde clip con calibración SoccerNet")
    ap.add_argument("--video", default=str(Path("videos") / "clip_5_25.mp4"))
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--offset", type=int, default=GLOBAL_FRAME_OFFSET)
    ap.add_argument("--offset_seconds", type=float, default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    template_img = cv2.imread(args.template, cv2.IMREAD_COLOR)
    if template_img is None:
        raise RuntimeError("No se pudo abrir template")

    calib_map = load_calibration_map(Path(args.calibration))
    last_H: Optional[np.ndarray] = None

    model = YOLO(args.model)
    annotator = sv.BoxAnnotator()

    cap = cv2.VideoCapture(str(Path(args.video)))
    if not cap.isOpened():
        raise RuntimeError("No se puede abrir video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    offset_frames = int(args.offset)
    if args.offset_seconds is None:
        parsed = parse_clip_offset_seconds(Path(args.video).name)
        if parsed is not None:
            offset_frames = int(parsed * fps)
    else:
        offset_frames = int(args.offset_seconds * fps)

    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            yres = model.predict(frame, conf=args.conf, iou=0.3, imgsz=args.imgsz, max_det=100, verbose=False)
            dets = sv.Detections.from_ultralytics(yres[0])
            if dets.class_id is not None:
                mask = dets.class_id == 0
                dets = dets[mask]

            frame_annot = annotator.annotate(scene=frame.copy(), detections=dets)
            pts_bc_px = bottom_center_from_xyxy(dets.xyxy.astype(np.float32)) if len(dets) > 0 else np.zeros((0, 2), dtype=np.float32)

            original_idx = frame_count + offset_frames
            H_candidate, last_H = get_h_for_frame(original_idx, calib_map, last_H)

            pts_field_m = np.zeros((0, 2), dtype=np.float32)
            if H_candidate is not None and pts_bc_px.size > 0:
                H_use = choose_best_orientation(H_candidate, pts_bc_px, PITCH_DIMENSIONS, test_inverse=True)
                pts_field_m = transform_points_to_meters(pts_bc_px, H_use)

            map_frame = template_img.copy()
            pts_template = meters_to_template_pixels(pts_field_m, map_frame.shape, PITCH_DIMENSIONS)
            for (tx, ty) in pts_template:
                cv2.circle(map_frame, (int(tx), int(ty)), 6, (0, 0, 255), -1)
                cv2.circle(map_frame, (int(tx), int(ty)), 6, (0, 0, 0), 1)

            cv2.imshow("Video Source", frame_annot)
            cv2.imshow("2D Tactical Map", map_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

            frame_count += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
