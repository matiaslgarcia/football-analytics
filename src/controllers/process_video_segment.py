import cv2
import numpy as np
import supervision as sv
from typing import Dict, Tuple
from collections import deque, Counter
from src.controllers.process_video import (
    extract_color_features,
    is_in_playing_field,
    is_in_goal_area,
    cluster_teams,
    classify_person_smart
)
from src.utils.view_transformer import ViewTransformer
from src.utils.radar import SoccerPitchConfiguration, draw_radar_view
from src.utils.soccernet_homography import load_homography_for_video, load_homography_provider
from ultralytics import YOLO

def process_video_segment(
    source_path: str,
    target_path: str,
    player_model,
    ball_model=None,
    pitch_model=None,
    conf: float = 0.3,
    detection_mode: str = "players_and_ball",
    img_size: int = 640,
    start_s: float = 0,
    duration_s: float = 10
):
    """
    Procesa solo un segmento del video con detección y tracking mejorado usando múltiples modelos.

    Args:
        source_path: Ruta al video de entrada
        target_path: Ruta al video de salida
        player_model: Modelo YOLO para detección de jugadores
        ball_model: Modelo YOLO para detección de pelota (opcional)
        pitch_model: Modelo YOLO para detección de campo (opcional)
        conf: Umbral de confianza para detecciones de personas
        detection_mode: Modo de detección ('players_only', 'ball_only', 'players_and_ball')
        img_size: Tamaño de imagen para inferencia
        start_s: Segundo de inicio del segmento
        duration_s: Duración del segmento en segundos
    """
    source_path = str(source_path)
    target_path = str(target_path)
    soccernet_H = load_homography_for_video(source_path)
    soccernet_provider = load_homography_provider(source_path)

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se puede abrir el video fuente: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(target_path, fourcc, fps, (width, height))

    start_frame = max(0, int(start_s * fps))
    total_frames = int(duration_s * fps)

    # Trackers modificados: Usar UN solo tracker para personas para mantener consistencia de ID
    person_tracker = sv.ByteTrack()
    ball_tracker = sv.ByteTrack()

    # Anotadores (mismos que process_video.py)
    team1_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("#00FF00"), thickness=2)
    team1_label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_color=sv.Color.BLACK, text_padding=3)

    team2_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("#00BFFF"), thickness=2)
    team2_label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_color=sv.Color.WHITE, text_padding=3)

    goalkeeper_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("#9B59B6"), thickness=2)
    goalkeeper_label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_color=sv.Color.WHITE, text_padding=3)

    referee_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("#FFD700"), thickness=2)
    referee_label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_color=sv.Color.BLACK, text_padding=3)

    ball_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("#FF0000"), thickness=3)
    ball_label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=2, text_color=sv.Color.WHITE, text_padding=4)

    # Variables para colores de equipos
    team1_colors = None
    team2_colors = None
    frame_count = 0

    # NUEVO: Sistema de tracking de asignación de equipos para consistencia
    reference_team1_color = None
    reference_team2_color = None
    clustering_initialized = False
    
    # Historial de votos para cada track_id para evitar parpadeos
    track_history = {}
    HISTORY_LEN = 30

    # NUEVO: Historial de posiciones para suavizado temporal en radar
    radar_positions_history = {}
    RADAR_SMOOTH_WINDOW = 5  # 5 frames para jugadores
    RADAR_SMOOTH_WINDOW_BALL = 3  # 3 frames para pelota (más rápido)

    # Inicializar configuración de pitch si hay modelo
    pitch_config = None
    if pitch_model or soccernet_H is not None:
        pitch_config = SoccerPitchConfiguration()

    # Ir al frame inicial
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    processed = 0
    try:
        while processed < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            annotated_frame = frame.copy()

            # --- DETECCIÓN DE JUGADORES ---
            player_results = player_model.predict(
                frame,
                conf=conf,
                iou=0.3,
                imgsz=img_size,
                max_det=100,
                verbose=False
            )
            player_detections = sv.Detections.from_ultralytics(player_results[0])
            
            # Filtrar solo clase 0 (personas)
            if player_detections.class_id is not None:
                player_detections = player_detections[player_detections.class_id == 0]

            # --- DETECCIÓN DE PELOTA MEJORADA ---
            ball_detections = sv.Detections.empty()
            if detection_mode in ["ball_only", "players_and_ball"]:
                if ball_model:
                    # Usar modelo específico de pelota con umbral más bajo
                    ball_results = ball_model.predict(
                        frame,
                        conf=max(0.1, conf * 0.3),  # Bajado de 0.15 a 0.1 (10% mínimo)
                        iou=0.4,  # Más permisivo (era 0.3)
                        imgsz=img_size,
                        verbose=False
                    )
                    ball_detections = sv.Detections.from_ultralytics(ball_results[0])

                    # Post-procesamiento: Filtrado por tamaño
                    if len(ball_detections) > 0:
                        # Calcular áreas de detecciones
                        widths = ball_detections.xyxy[:, 2] - ball_detections.xyxy[:, 0]
                        heights = ball_detections.xyxy[:, 3] - ball_detections.xyxy[:, 1]
                        areas = widths * heights

                        # Filtrar por tamaño razonable (entre 0.5% y 5% del frame)
                        max_ball_area = (width * 0.05) * (height * 0.05)
                        min_ball_area = (width * 0.005) * (height * 0.005)
                        valid_size = (areas < max_ball_area) & (areas > min_ball_area)

                        ball_detections = ball_detections[valid_size]

                        # Si hay múltiples detecciones, tomar la de mayor confianza
                        if len(ball_detections) > 1:
                            best_idx = np.argmax(ball_detections.confidence)
                            ball_detections = ball_detections[best_idx:best_idx+1]
                else:
                    # Fallback: Usar modelo de jugadores si tiene clase 32 (sports ball)
                    raw_detections = sv.Detections.from_ultralytics(player_results[0])
                    if raw_detections.class_id is not None:
                         ball_detections = raw_detections[raw_detections.class_id == 32]
                         if len(ball_detections) > 0:
                             # Aplicar mismo umbral más bajo
                             ball_conf_threshold = max(0.1, conf * 0.3)
                             ball_detections = ball_detections[ball_detections.confidence >= ball_conf_threshold]

                             # Post-procesamiento por tamaño (igual que modelo específico)
                             widths = ball_detections.xyxy[:, 2] - ball_detections.xyxy[:, 0]
                             heights = ball_detections.xyxy[:, 3] - ball_detections.xyxy[:, 1]
                             areas = widths * heights

                             max_ball_area = (width * 0.05) * (height * 0.05)
                             min_ball_area = (width * 0.005) * (height * 0.005)
                             valid_size = (areas < max_ball_area) & (areas > min_ball_area)

                             ball_detections = ball_detections[valid_size]

                             # Tomar mejor detección si hay múltiples
                             if len(ball_detections) > 1:
                                 best_idx = np.argmax(ball_detections.confidence)
                                 ball_detections = ball_detections[best_idx:best_idx+1]

            # --- MEJORA: Tracking PRIMERO ---
            tracked_persons = person_tracker.update_with_detections(player_detections)

            # Recalcular colores cada 45 frames (mejorado de 30)
            if (frame_count % 45 == 1 or team1_colors is None) and len(player_detections) > 0:
                team1_colors_new, team2_colors_new, _, _, _ = cluster_teams(
                    frame, player_detections, width, height
                )

                # NUEVO: Verificar consistencia con colores de referencia
                if not clustering_initialized:
                    # Primera inicialización
                    reference_team1_color = team1_colors_new['shirt'].copy()
                    reference_team2_color = team2_colors_new['shirt'].copy()
                    team1_colors = team1_colors_new
                    team2_colors = team2_colors_new
                    clustering_initialized = True
                else:
                    # Verificar intercambio usando suma de distancias
                    dist_t1_ref1 = np.linalg.norm(team1_colors_new['shirt'] - reference_team1_color)
                    dist_t1_ref2 = np.linalg.norm(team1_colors_new['shirt'] - reference_team2_color)
                    dist_t2_ref1 = np.linalg.norm(team2_colors_new['shirt'] - reference_team1_color)
                    dist_t2_ref2 = np.linalg.norm(team2_colors_new['shirt'] - reference_team2_color)

                    score_keep = dist_t1_ref1 + dist_t2_ref2
                    score_swap = dist_t1_ref2 + dist_t2_ref1

                    if score_swap < score_keep:
                        # INTERCAMBIAR
                        team1_colors = team2_colors_new
                        team2_colors = team1_colors_new
                    else:
                        # Mantener
                        team1_colors = team1_colors_new
                        team2_colors = team2_colors_new
                    
                    # Actualizar referencia suavemente
                    alpha = 0.1
                    reference_team1_color = (1 - alpha) * reference_team1_color + alpha * team1_colors['shirt']
                    reference_team2_color = (1 - alpha) * reference_team2_color + alpha * team2_colors['shirt']
            
            # Clasificar y Votar
            team1_mask = []
            team2_mask = []
            referee_mask = []
            goalkeeper_mask = []
            
            if len(tracked_persons) > 0 and team1_colors is not None:
                for i, (xyxy, _, _, _, tracker_id, _) in enumerate(tracked_persons):
                    # 0. Filtrar personas FUERA del campo (Entrenadores, público, etc.)
                    if not is_in_playing_field(xyxy, width, height):
                        goalkeeper_mask.append(False)
                        team1_mask.append(False)
                        team2_mask.append(False)
                        referee_mask.append(False)
                        continue

                    in_goal = is_in_goal_area(xyxy, width, height)
                    x_center = (xyxy[0] + xyxy[2]) / 2
                    is_gk = False
                    
                    # Ampliado a 12% para cubrir mejor el área chica (igual que process_video.py)
                    if in_goal and (x_center < width * 0.12 or x_center > width * 0.88):
                        is_gk = True
                        current_vote = 'goalkeeper'
                    else:
                        person_type, _ = classify_person_smart(
                            frame, xyxy, team1_colors, team2_colors, width, height
                        )
                        current_vote = person_type

                    if tracker_id not in track_history:
                        track_history[tracker_id] = deque(maxlen=HISTORY_LEN)
                    
                    track_history[tracker_id].append(current_vote)
                    
                    vote_counts = Counter(track_history[tracker_id])
                    most_common = vote_counts.most_common(1)[0][0]
                    
                    if current_vote == 'goalkeeper':
                        final_class = 'goalkeeper'
                    else:
                        final_class = most_common

                    if final_class == 'goalkeeper':
                        goalkeeper_mask.append(True)
                        team1_mask.append(False)
                        team2_mask.append(False)
                        referee_mask.append(False)
                    elif final_class == 'referee':
                        goalkeeper_mask.append(False)
                        team1_mask.append(False)
                        team2_mask.append(False)
                        referee_mask.append(True)
                    elif final_class == 'team1':
                        goalkeeper_mask.append(False)
                        team1_mask.append(True)
                        team2_mask.append(False)
                        referee_mask.append(False)
                    elif final_class == 'team2':
                        goalkeeper_mask.append(False)
                        team1_mask.append(False)
                        team2_mask.append(True)
                        referee_mask.append(False)
                    else:
                        goalkeeper_mask.append(False)
                        team1_mask.append(False)
                        team2_mask.append(False)
                        referee_mask.append(False)

            # Anotaciones
            if any(team1_mask):
                t1_dets = tracked_persons[np.array(team1_mask)]
                annotated_frame = team1_annotator.annotate(scene=annotated_frame, detections=t1_dets)
                if t1_dets.tracker_id is not None:
                    labels = [f"Team 1 #{tid}" for tid in t1_dets.tracker_id]
                    annotated_frame = team1_label_annotator.annotate(scene=annotated_frame, detections=t1_dets, labels=labels)

            if any(team2_mask):
                t2_dets = tracked_persons[np.array(team2_mask)]
                annotated_frame = team2_annotator.annotate(scene=annotated_frame, detections=t2_dets)
                if t2_dets.tracker_id is not None:
                    labels = [f"Team 2 #{tid}" for tid in t2_dets.tracker_id]
                    annotated_frame = team2_label_annotator.annotate(scene=annotated_frame, detections=t2_dets, labels=labels)

            if any(goalkeeper_mask):
                gk_dets = tracked_persons[np.array(goalkeeper_mask)]
                annotated_frame = goalkeeper_annotator.annotate(scene=annotated_frame, detections=gk_dets)
                if gk_dets.tracker_id is not None:
                    labels = [f"GK #{tid}" for tid in gk_dets.tracker_id]
                    annotated_frame = goalkeeper_label_annotator.annotate(scene=annotated_frame, detections=gk_dets, labels=labels)

            if any(referee_mask):
                ref_dets = tracked_persons[np.array(referee_mask)]
                annotated_frame = referee_annotator.annotate(scene=annotated_frame, detections=ref_dets)
                if ref_dets.tracker_id is not None:
                    labels = [f"Referee #{tid}" for tid in ref_dets.tracker_id]
                    annotated_frame = referee_label_annotator.annotate(scene=annotated_frame, detections=ref_dets, labels=labels)

            # Tracking de pelota
            tracked_ball = None # Inicializar
            if len(ball_detections) > 0:
                tracked_ball = ball_tracker.update_with_detections(ball_detections)
                annotated_frame = ball_annotator.annotate(scene=annotated_frame, detections=tracked_ball)
                ball_labels = ["BALL"] * len(tracked_ball)
                annotated_frame = ball_label_annotator.annotate(scene=annotated_frame, detections=tracked_ball, labels=ball_labels)

            # --- RADAR CON SUAVIZADO TEMPORAL ---
            if pitch_config:
                try:
                    transformer = None
                    if soccernet_provider is not None:
                        m = soccernet_provider.get_for_time((start_frame + processed) / fps)
                        if m is not None:
                            transformer = ViewTransformer.from_matrix(m)
                    elif soccernet_H is not None:
                        transformer = ViewTransformer.from_matrix(soccernet_H)
                    if transformer is None and pitch_model:
                        pitch_results = pitch_model(frame, verbose=False, conf=0.3)[0]
                        if pitch_results.keypoints is not None and len(pitch_results.keypoints) > 0:
                            keypoints_xy = pitch_results.keypoints.xy.cpu().numpy()[0]
                            keypoints_conf = pitch_results.keypoints.conf.cpu().numpy()[0]

                            valid_kp_mask = keypoints_conf > 0.5
                            valid_keypoints = keypoints_xy[valid_kp_mask]
                            valid_indices = np.where(valid_kp_mask)[0]

                            if len(valid_keypoints) >= 4:
                                target_points = pitch_config.get_keypoints_from_ids(valid_indices)
                                transformer = ViewTransformer(valid_keypoints, target_points)

                    points_to_transform = {}

                    def get_bottom_center(dets):
                        return np.column_stack([
                            (dets.xyxy[:, 0] + dets.xyxy[:, 2]) / 2,
                            dets.xyxy[:, 3]
                        ])

                    def smooth_positions(tracker_ids, raw_positions, window_size):
                        """Suaviza posiciones usando promedio móvil por tracker_id"""
                        if len(tracker_ids) == 0 or len(raw_positions) == 0:
                            return raw_positions

                        smoothed = []
                        for tid, pos in zip(tracker_ids, raw_positions):
                            key = f"{tid}"
                            if key not in radar_positions_history:
                                radar_positions_history[key] = deque(maxlen=window_size)

                            radar_positions_history[key].append(pos)
                            avg_pos = np.mean(list(radar_positions_history[key]), axis=0)
                            smoothed.append(avg_pos)

                        return np.array(smoothed)

                    # Transformar y suavizar Team 1
                    if any(team1_mask):
                        t1_dets = tracked_persons[np.array(team1_mask)]
                        t1_raw = transformer.transform_points(get_bottom_center(t1_dets))
                        if t1_dets.tracker_id is not None:
                            points_to_transform['team1'] = smooth_positions(
                                t1_dets.tracker_id, t1_raw, RADAR_SMOOTH_WINDOW
                            )
                        else:
                            points_to_transform['team1'] = t1_raw

                    # Transformar y suavizar Team 2
                    if any(team2_mask):
                        t2_dets = tracked_persons[np.array(team2_mask)]
                        t2_raw = transformer.transform_points(get_bottom_center(t2_dets))
                        if t2_dets.tracker_id is not None:
                            points_to_transform['team2'] = smooth_positions(
                                t2_dets.tracker_id, t2_raw, RADAR_SMOOTH_WINDOW
                            )
                        else:
                            points_to_transform['team2'] = t2_raw

                    # Referee (sin suavizado, se mueven menos)
                    if any(referee_mask):
                        points_to_transform['referee'] = transformer.transform_points(
                            get_bottom_center(tracked_persons[np.array(referee_mask)])
                        )

                    # Goalkeeper (sin suavizado, se mueven menos)
                    if any(goalkeeper_mask):
                        points_to_transform['goalkeeper'] = transformer.transform_points(
                            get_bottom_center(tracked_persons[np.array(goalkeeper_mask)])
                        )

                    # Ball con ventana más corta (movimiento rápido)
                    if tracked_ball is not None and len(tracked_ball) > 0:
                        ball_raw = transformer.transform_points(get_bottom_center(tracked_ball))
                        if tracked_ball.tracker_id is not None:
                            points_to_transform['ball'] = smooth_positions(
                                tracked_ball.tracker_id, ball_raw, RADAR_SMOOTH_WINDOW_BALL
                            )
                        else:
                            points_to_transform['ball'] = ball_raw

                    radar_view = draw_radar_view(pitch_config, points_to_transform)
                    
                    # Overlay PIP
                    scale_factor = 0.3 # 30% del ancho
                    new_w = int(width * scale_factor)
                    aspect_ratio = radar_view.shape[0] / radar_view.shape[1]
                    new_h = int(new_w * aspect_ratio)
                    
                    radar_resized = cv2.resize(radar_view, (new_w, new_h))
                    
                    # Posición: Abajo al centro
                    offset_x = (width - new_w) // 2
                    offset_y = height - new_h - 20
                    
                    if offset_y > 0:
                        annotated_frame[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = radar_resized
                except Exception as e:
                    print(f"Error en Radar: {e}")

            out.write(annotated_frame)
            processed += 1

    finally:
        cap.release()
        out.release()
