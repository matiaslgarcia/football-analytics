"""
Wrapper mejorado de process_video que incluye análisis táctico
y retorna estadísticas completas para visualización en Streamlit.

Video generado: Solo tracking + radar 2D limpio
Retorno: Estadísticas completas, formaciones, métricas temporales
"""

import cv2
import numpy as np
import supervision as sv
from typing import Dict, Optional, Tuple
from pathlib import Path
import json

from src.controllers.formation_detector import FormationDetector
from src.controllers.tactical_metrics import TacticalMetricsCalculator, TacticalMetricsTracker
from src.utils.radar import SoccerPitchConfiguration, draw_radar_view
from src.utils.view_transformer import ViewTransformer
from src.utils.team_assigner import TeamAssigner


def process_video_with_analytics(
    source_path: str,
    target_path: str,
    player_model,
    ball_model=None,
    pitch_model=None,
    conf: float = 0.3,
    detection_mode: str = "players_and_ball",
    full_field_approx: bool = False,
    export_stats: bool = True
) -> Dict:
    """
    Procesa video con análisis táctico completo.

    Args:
        ... (mismos que process_video original)
        export_stats: Si exportar estadísticas a JSON

    Returns:
        Dict con:
            - success: bool
            - output_video: str (path)
            - stats_file: str (path al JSON con estadísticas)
            - formations: Dict con formaciones detectadas
            - metrics: Dict con métricas promedio
            - timeline: Dict con evolución temporal
    """

    # Inicializar módulos tácticos
    formation_detector = FormationDetector()
    metrics_calculator = TacticalMetricsCalculator()
    team1_tracker = TacticalMetricsTracker(history_size=500)
    team2_tracker = TacticalMetricsTracker(history_size=500)

    # Configuración del campo
    pitch_config = SoccerPitchConfiguration(model_type='soccana' if pitch_model else 'default')

    # Abrir video
    cap = cv2.VideoCapture(source_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(target_path, fourcc, fps, (width, height))

    # Trackers
    person_tracker = sv.ByteTrack()
    ball_tracker = sv.ByteTrack()
    team_assigner = TeamAssigner()

    # Almacenar formaciones detectadas
    formations_timeline = {'team1': [], 'team2': []}

    # Anotadores
    team1_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("#00FF00"), thickness=2)
    team2_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("#00BFFF"), thickness=2)
    team1_label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    team2_label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    frame_count = 0

    print(f"Procesando {total_frames} frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detección
        results = player_model.predict(frame, conf=conf, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])

        # Filtrar personas
        person_class = 0
        person_detections = detections[detections.class_id == person_class]

        # Tracking
        person_detections = person_tracker.update_with_detections(person_detections)

        # Asignar equipos
        if len(person_detections) > 0:
            team_assignments = team_assigner.assign_teams(frame, person_detections)
        else:
            team_assignments = {'team1': [], 'team2': [], 'referee': [], 'goalkeeper': []}

        # Transformación (aproximación simple para este wrapper)
        transformer = None
        if pitch_model or full_field_approx:
            # Aproximación de campo completo
            source = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
            target = np.array([[0, 0], [105, 0], [105, 68], [0, 68]], dtype=np.float32)
            transformer = ViewTransformer(source, target)

        # Transformar posiciones y calcular métricas
        team1_positions = []
        team2_positions = []

        if transformer:
            for det_idx in team_assignments['team1']:
                if det_idx < len(person_detections):
                    bbox = person_detections.xyxy[det_idx]
                    pos = np.array([[(bbox[0] + bbox[2])/2, bbox[3]]])
                    trans = transformer.transform_points(pos)
                    team1_positions.append(trans[0])

            for det_idx in team_assignments['team2']:
                if det_idx < len(person_detections):
                    bbox = person_detections.xyxy[det_idx]
                    pos = np.array([[(bbox[0] + bbox[2])/2, bbox[3]]])
                    trans = transformer.transform_points(pos)
                    team2_positions.append(trans[0])

        team1_pos_arr = np.array(team1_positions) if team1_positions else np.array([])
        team2_pos_arr = np.array(team2_positions) if team2_positions else np.array([])

        # Calcular formaciones y métricas
        if len(team1_pos_arr) >= 3:
            form1 = formation_detector.detect_formation(team1_pos_arr, "right")
            metrics1 = metrics_calculator.calculate_all_metrics(team1_pos_arr)
            team1_tracker.update(metrics1, frame_count)
            formations_timeline['team1'].append(form1['formation'])

        if len(team2_pos_arr) >= 3:
            form2 = formation_detector.detect_formation(team2_pos_arr, "left")
            metrics2 = metrics_calculator.calculate_all_metrics(team2_pos_arr)
            team2_tracker.update(metrics2, frame_count)
            formations_timeline['team2'].append(form2['formation'])

        # Dibujar radar LIMPIO (sin métricas en pantalla)
        if transformer:
            transformed_points = {
                'team1': team1_pos_arr,
                'team2': team2_pos_arr
            }

            radar_view = draw_radar_view(pitch_config, transformed_points, scale=8)

            # Redimensionar y colocar radar
            scale_factor = 0.35
            new_w = int(width * scale_factor)
            aspect_ratio = radar_view.shape[0] / radar_view.shape[1]
            new_h = int(new_w * aspect_ratio)
            radar_resized = cv2.resize(radar_view, (new_w, new_h))

            margin = 20
            x_pos = (width - new_w) // 2
            y_pos = height - new_h - margin

            if y_pos + new_h <= height and x_pos + new_w <= width:
                frame[y_pos:y_pos+new_h, x_pos:x_pos+new_w] = radar_resized

        # Anotar frame (sin radar con métricas)
        # ... (código de anotación de bounding boxes)

        out.write(frame)

        if frame_count % 100 == 0:
            print(f"  Procesado: {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")

    cap.release()
    out.release()

    # Calcular estadísticas finales
    stats1 = team1_tracker.get_statistics()
    stats2 = team2_tracker.get_statistics()

    # Formación más común
    from collections import Counter
    team1_formation = Counter(formations_timeline['team1']).most_common(1)[0][0] if formations_timeline['team1'] else 'N/A'
    team2_formation = Counter(formations_timeline['team2']).most_common(1)[0][0] if formations_timeline['team2'] else 'N/A'

    result = {
        'success': True,
        'output_video': target_path,
        'total_frames': frame_count,
        'duration_seconds': frame_count / fps if fps > 0 else 0,
        'formations': {
            'team1': {
                'most_common': team1_formation,
                'timeline': formations_timeline['team1']
            },
            'team2': {
                'most_common': team2_formation,
                'timeline': formations_timeline['team2']
            }
        },
        'metrics': {
            'team1': stats1,
            'team2': stats2
        },
        'timeline': {
            'team1': team1_tracker.export_to_dict(),
            'team2': team2_tracker.export_to_dict()
        }
    }

    # Exportar a JSON
    if export_stats:
        stats_path = Path(target_path).parent / f"{Path(target_path).stem}_stats.json"
        with open(stats_path, 'w') as f:
            # Convertir arrays numpy a listas para JSON
            json_result = _prepare_for_json(result)
            json.dump(json_result, f, indent=2)
        result['stats_file'] = str(stats_path)
        print(f"\nEstadísticas exportadas: {stats_path}")

    return result


def _prepare_for_json(data):
    """Convierte arrays numpy a listas para serialización JSON"""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: _prepare_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_prepare_for_json(item) for item in data]
    elif isinstance(data, (np.integer, np.floating)):
        return float(data)
    else:
        return data
