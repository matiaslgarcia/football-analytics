"""
Descarga un video puntual del dataset SoccerNet a la carpeta `videos/`.

Requiere:
- Password de videos tras completar el NDA de SoccerNet.
- Paquete `SoccerNet` instalado.

Uso de ejemplo (720p, ambas mitades):
  venv\Scripts\python.exe download_soccernet_video.py \
    --game "europe_uefa-champions-league/2016-2017/2017-04-18 - 21-45 Real Madrid 4 - 2 Bayern Munich" \
    --quality 720p --half both --output_dir videos --password "<TU_PASSWORD_NDA>"

Referencias:
- PyPI SoccerNet: https://pypi.org/project/SoccerNet/
- Documentación de descarga: https://github.com/SoccerNet/SoccerNet
"""

import os
from pathlib import Path
import argparse
from SoccerNet.Downloader import SoccerNetDownloader


def main():
    parser = argparse.ArgumentParser(description="Descargar un video de SoccerNet")
    parser.add_argument("--game", required=True, help="Ruta del juego: league/season/game (ver documentación de SoccerNet)")
    parser.add_argument("--quality", choices=["720p", "224p"], default="720p")
    parser.add_argument("--half", choices=["1", "2", "both"], default="both")
    parser.add_argument("--output_dir", default="videos", help="Directorio local para guardar los videos")
    parser.add_argument("--password", default=None, help="Password de videos (NDA). Alternativa: variable de entorno SOCCERNET_PASSWORD o SOCCERNET_PW")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    downloader = SoccerNetDownloader(LocalDirectory=str(out_dir))

    pw = args.password or os.getenv("SOCCERNET_PASSWORD") or os.getenv("SOCCERNET_PW")
    if pw:
        downloader.password = pw
    else:
        print("[WARN] No se proporcionó password. La descarga de videos requiere el password del NDA.")

    files = []
    if args.half in ("1", "both"):
        files.append("1_720p.mkv" if args.quality == "720p" else "1_224p.mkv")
    if args.half in ("2", "both"):
        files.append("2_720p.mkv" if args.quality == "720p" else "2_224p.mkv")

    print(f"Descargando {files} para juego: {args.game} en {out_dir}")
    downloader.downloadGame(files=files, game=args.game)
    print("Descarga completada. Busca los archivos .mkv dentro de la estructura de 'videos/'.")


if __name__ == "__main__":
    main()