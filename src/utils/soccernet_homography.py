import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np

def _find_candidate_files(game_dir: Path) -> list[Path]:
    patterns = [
        "**/*calib*.json",
        "**/*Calibration*.json",
        "**/*homograph*.json",
        "**/*homograph*.txt",
        "**/*homography*.json",
        "**/*homography*.txt",
    ]
    files: list[Path] = []
    for pat in patterns:
        files.extend(list(game_dir.glob(pat)))
    return files

def _extract_matrix_from_json(obj) -> Optional[np.ndarray]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k).lower()
            if key in ("h", "homography", "matrix"):
                arr = _to_matrix(v)
                if arr is not None:
                    return arr
            result = _extract_matrix_from_json(v)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for v in obj:
            result = _extract_matrix_from_json(v)
            if result is not None:
                return result
    return None

def _to_matrix(v) -> Optional[np.ndarray]:
    try:
        arr = np.array(v, dtype=float)
        if arr.size == 9:
            arr = arr.reshape(3, 3)
            return arr
        if arr.shape == (3, 3):
            return arr.astype(float)
    except Exception:
        return None
    return None

def _read_txt_matrix(path: Path) -> Optional[np.ndarray]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        nums = []
        for token in text.replace("\n", " ").split():
            try:
                nums.append(float(token))
            except Exception:
                pass
        if len(nums) >= 9:
            arr = np.array(nums[:9], dtype=float).reshape(3, 3)
            return arr
    except Exception:
        return None
    return None

def load_homography_for_video(video_path: str | Path) -> Optional[np.ndarray]:
    p = Path(video_path)
    game_dir = p.parent
    candidates = _find_candidate_files(game_dir)
    for c in candidates:
        if c.suffix.lower() == ".json":
            try:
                data = json.loads(c.read_text(encoding="utf-8", errors="ignore"))
                m = _extract_matrix_from_json(data)
                if m is not None:
                    return m.astype(np.float32)
            except Exception:
                continue
        else:
            m = _read_txt_matrix(c)
            if m is not None:
                return m.astype(np.float32)
    return None

class HomographyProvider:
    def __init__(self, entries: List[Dict[str, Any]], fallback: Optional[np.ndarray] = None, offset_s: float = 0.0):
        self.entries = entries
        self.fallback = fallback.astype(np.float32) if fallback is not None else None
        self.offset_s = float(offset_s)

    def get_for_time(self, t: float) -> Optional[np.ndarray]:
        t = t + self.offset_s
        best = None
        best_dist = float('inf')
        for e in self.entries:
            if 'start_s' in e and 'end_s' in e and e.get('H') is not None:
                if e['start_s'] <= t <= e['end_s']:
                    return e['H']
            elif 'time_s' in e and e.get('H') is not None:
                dist = abs(e['time_s'] - t)
                if dist < best_dist:
                    best_dist = dist
                    best = e['H']
        return best if best is not None else self.fallback

    def get_for_frame(self, f: int) -> Optional[np.ndarray]:
        best = None
        best_dist = float('inf')
        for e in self.entries:
            if 'start_f' in e and 'end_f' in e and e.get('H') is not None:
                if e['start_f'] <= f <= e['end_f']:
                    return e['H']
            elif 'frame' in e and e.get('H') is not None:
                dist = abs(e['frame'] - f)
                if dist < best_dist:
                    best_dist = dist
                    best = e['H']
        return best if best is not None else self.fallback

def _parse_entries_from_json(obj: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    def add(H, time_s=None, frame=None, start_s=None, end_s=None, start_f=None, end_f=None):
        m = _to_matrix(H)
        if m is not None:
            out.append({
                'H': m.astype(np.float32),
                'time_s': time_s,
                'frame': frame,
                'start_s': start_s,
                'end_s': end_s,
                'start_f': start_f,
                'end_f': end_f,
            })
    if isinstance(obj, dict):
        if 'homographies' in obj and isinstance(obj['homographies'], list):
            for it in obj['homographies']:
                if isinstance(it, dict):
                    add(
                        it.get('H') or it.get('homography') or it.get('matrix'),
                        time_s=it.get('time') or it.get('time_s') or it.get('timestamp'),
                        frame=it.get('frame')
                    )
        else:
            for k, v in obj.items():
                key = str(k).lower()
                if key in ('segments', 'intervals') and isinstance(v, list):
                    for it in v:
                        if isinstance(it, dict):
                            add(
                                it.get('H') or it.get('homography') or it.get('matrix'),
                                start_s=it.get('start_s') or it.get('start'),
                                end_s=it.get('end_s') or it.get('end')
                            )
                elif key in ('per_frame', 'frames') and isinstance(v, list):
                    for it in v:
                        if isinstance(it, dict):
                            add(
                                it.get('H') or it.get('homography') or it.get('matrix'),
                                frame=it.get('frame')
                            )
                elif key in ('h', 'homography', 'matrix'):
                    add(v)
    elif isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                add(
                    it.get('H') or it.get('homography') or it.get('matrix'),
                    time_s=it.get('time') or it.get('time_s') or it.get('timestamp'),
                    frame=it.get('frame')
                )
            else:
                add(it)
    return out

def load_homography_provider(video_path: str | Path) -> Optional[HomographyProvider]:
    p = Path(video_path)
    game_dir = p.parent
    def _clip_offset_seconds(name: str) -> float:
        try:
            if "_clip_" in name:
                tail = name.split("_clip_")[-1]
                start = tail.split("-")[0]
                return float(start)
        except Exception:
            pass
        return 0.0
    candidates = _find_candidate_files(game_dir)
    fallback = None
    entries: List[Dict[str, Any]] = []
    for c in candidates:
        if c.suffix.lower() == ".json":
            try:
                data = json.loads(c.read_text(encoding="utf-8", errors="ignore"))
                # try to get a single matrix as fallback
                single = _extract_matrix_from_json(data)
                if single is not None:
                    fallback = single
                # parse potential time/frame entries
                entries.extend(_parse_entries_from_json(data))
            except Exception:
                continue
        else:
            m = _read_txt_matrix(c)
            if m is not None:
                fallback = m
    if len(entries) == 0 and fallback is None:
        return None
    return HomographyProvider(entries, fallback, offset_s=_clip_offset_seconds(p.name))
