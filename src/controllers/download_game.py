from pathlib import Path
from typing import Dict, List
from SoccerNet.Downloader import SoccerNetDownloader

from ..utils.config import VIDEOS_DIR
from ..utils.soccernet_password import resolve_password
from .clip_video_simple import clip_video_simple


def _files_for_quality_and_half(quality: str, half_choice: str) -> List[str]:
    files = []
    if half_choice in ("1", "both"):
        files.append("1_720p.mkv" if quality == "720p" else "1_224p.mkv")
    if half_choice in ("2", "both"):
        files.append("2_720p.mkv" if quality == "720p" else "2_224p.mkv")
    return files


def download_game(
    game_path: str,
    quality: str,
    half_choice: str,
    password: str | None = None,
    local_dir: Path | str = VIDEOS_DIR,
    recortar: bool = False,
    start_s: float = 0.0,
    duration_s: float = 10.0,
) -> Dict[str, List[Path]]:
    """Descarga un juego de SoccerNet y opcionalmente recorta clips de los halves descargados.

    Retorna un dict con claves: files (archivos descargados), clips (clips generados), game_dir.
    """
    videos_dir = Path(local_dir)
    videos_dir.mkdir(exist_ok=True)

    files = _files_for_quality_and_half(quality, half_choice)
    pw = resolve_password(password)

    downloader = SoccerNetDownloader(LocalDirectory=str(videos_dir))
    downloader.password = pw

    # Estructura del path: league/season/game_id
    downloader.downloadGame(files=files, game=game_path)

    game_dir = videos_dir / Path(game_path)
    downloaded_paths = [game_dir / f for f in files]

    clips: List[Path] = []
    if recortar:
        for src in downloaded_paths:
            if not src.exists():
                continue
            base = src.stem
            clip_name = f"{base}_clip_{int(start_s)}-{int(duration_s)}.mp4"
            dst = game_dir / clip_name
            clip_video_simple(str(src), str(dst), float(start_s), float(duration_s))
            clips.append(dst)

    return {"files": downloaded_paths, "clips": clips, "game_dir": game_dir}