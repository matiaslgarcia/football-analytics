# Proyecto: Detección y tracking de jugadores (YOLOv8 + Supervision)

Esta app de Streamlit permite:
- Subir o seleccionar videos locales de fútbol (incluye soporte para SoccerNet).
- Detectar personas con YOLOv8 y hacer tracking por ID (ByteTrack) usando Supervision.
- Procesar el video completo o sólo un segmento por tiempo (inicio/duración).
- Descargar clips tras la descarga desde SoccerNet y analizar directamente esos tramos.

## Prerrequisitos
- Python 3.10 o superior.
- Windows (PowerShell o CMD).

## Montaje paso a paso (Windows)
1. Abre PowerShell en la carpeta del proyecto.
2. Crea y activa un entorno virtual:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   - Alternativa en CMD: `venv\Scripts\activate`
3. (Opcional) Actualiza `pip`:
   ```powershell
   python -m pip install --upgrade pip
   ```
4. Instala dependencias:
   ```powershell
   pip install -r requirements.txt
   ```
5. Ejecuta la app:
   ```powershell
   streamlit run app.py
   ```
6. Abre la URL que aparece (ej. `http://localhost:8501`).

## Arquitectura modular
El código de `src/` está organizado por responsabilidad para lograr desacoplamiento, escalabilidad y claridad.

- `src/models/`
  - `load_model.py`: carga y cachea el modelo YOLOv8.
- `src/controllers/`
  - `process_video.py`: procesamiento completo con detección + tracking.
  - `process_video_segment.py`: procesamiento de segmento por tiempo.
  - `clip_video_simple.py`: recorte simple de video sin audio.
  - `download_game.py`: descarga de videos desde SoccerNet y generación opcional de clips.
- `src/utils/`
  - `config.py`: rutas base (`INPUTS_DIR`, `OUTPUTS_DIR`, `VIDEOS_DIR`) y claves de entorno de NDA.
  - `soccernet_password.py`: resolución de password (`resolve_password`).
  - `ui/`
    - `sidebar_processing_controls.py`: controles de procesamiento en barra lateral.
    - `source_selector.py`: selector de origen del video (subida o SoccerNet local).
    - `download_controls.py`: expander de descarga de SoccerNet.
- `src/tests/`
  - `test_smoke.py`: prueba mínima de importación para verificar estructura.

Cada archivo contiene únicamente funciones ligadas a su propósito. Todas las carpetas tienen `__init__.py` para permitir importaciones correctas.

## Guía rápida de imports
Ejemplo de cómo importar y usar los módulos desde `app.py`:
```python
from pathlib import Path
from src.models.load_model import load_model
from src.controllers.process_video import process_video
from src.controllers.process_video_segment import process_video_segment
from src.controllers.download_game import download_game
from src.utils.config import INPUTS_DIR, OUTPUTS_DIR, VIDEOS_DIR
from src.utils.ui.sidebar_processing_controls import sidebar_processing_controls
from src.utils.ui.source_selector import source_selector
from src.utils.ui.download_controls import download_controls

# Cargar modelo
model = load_model("yolov8n.pt")

# Procesar video completo
process_video("inputs/partido.mp4", "outputs/result_partido.mp4", model, conf=0.25, only_person=True, img_size=640)

# Procesar segmento
process_video_segment("inputs/partido.mp4", "outputs/result_seg.mp4", model, conf=0.25, only_person=True, img_size=640, start_s=30.0, duration_s=10.0)

# Descargar desde SoccerNet
result = download_game(
    game_path="europe_uefa-champions-league/2016-2017/2017-04-18 - 21-45 Real Madrid 4 - 2 Bayern Munich",
    quality="720p",
    half_choice="both",
    password=None,  # o usa variables de entorno
    local_dir=VIDEOS_DIR,
    recortar=True,
    start_s=0.0,
    duration_s=10.0,
)
```

## Estructura de carpetas
- `inputs/`: se guarda el archivo subido antes de procesarlo.
- `videos/`: videos locales y los descargados desde SoccerNet (estructura liga/temporada/partido).
- `outputs/`: resultados anotados (`result_*.mp4`).
- `yolov8n.pt`: modelo YOLOv8n (si eliges `yolov8s.pt`, se descarga automáticamente).

## Uso en la app
- Selector de origen de video: `Subir archivo` o `SoccerNet local`.
- Barra lateral:
  - `Modelo YOLOv8`, `Umbral de confianza`, `Solo personas`, `Tamaño de imagen`.
  - `Procesar solo un segmento`: define `Inicio (seg)` y `Duración (seg)` para analizar sólo ese tramo.
- Sección "Descargar video de SoccerNet":
  - Ingresa `Ruta del juego (league/season/game)`, `Calidad` (`720p`/`224p`), `Mitad` (`1`/`2`/`both`) y `Password (NDA)`.
  - Activa `Recortar tras descarga` para generar clips MP4 de X segundos por cada mitad descargada.
  - Los archivos se guardan bajo `videos/<liga>/<temporada>/<partido>/` y aparecen en el selector "SoccerNet local".

## Descarga vía consola (opcional)
También puedes usar el script CLI incluido:
```powershell
# Password del NDA (puedes pasarlo por UI o por variable)
$env:SOCCERNET_PASSWORD = "<TU_PASSWORD_NDA>"

# Ejemplo: descarga ambas mitades en 720p
venv\Scripts\python.exe download_soccernet_video.py \
  --game "europe_uefa-champions-league/2016-2017/2017-04-18 - 21-45 Real Madrid 4 - 2 Bayern Munich" \
  --quality 720p --half both --output_dir videos
```
Al finalizar, selecciona los `.mkv` o los clips `.mp4` desde "SoccerNet local".

## Notas y consejos
- Los videos oficiales de SoccerNet requieren NDA y password.
- Variables de entorno aceptadas para la SDK: `SOCCERNET_PASSWORD` o `SOCCERNET_PW`.
- El recorte simple genera MP4 sin audio. Si quieres clips con audio o concatenar mitades, se puede añadir una ruta con `ffmpeg`.
- Sin GPU, el procesamiento será en CPU y tomará más tiempo.

## Problemas comunes
- "Streamlit no se reconoce": activa el entorno virtual o reinstala dependencias.
- "No abre el venv en PowerShell": prueba `venv\Scripts\activate` desde CMD.
- Errores al descargar SoccerNet: verifica tu `Password (NDA)` y la `Ruta del juego` exacta.

## Referencias
- SoccerNet (datos/NDA): https://www.soccer-net.org/data
- SDK PyPI: https://pypi.org/project/SoccerNet/
- Supervision: https://supervision.roboflow.com/latest/how_to/track_objects/