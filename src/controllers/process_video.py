import cv2
import numpy as np
import supervision as sv
from typing import Dict, Tuple, List
from sklearn.cluster import KMeans
from collections import deque, Counter
from src.utils.view_transformer import ViewTransformer
from src.utils.radar import SoccerPitchConfiguration, draw_radar_view
from ultralytics import YOLO


def extract_color_features(frame: np.ndarray, bbox: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extrae características de color de una persona, separando camiseta y pantalón.

    Returns:
        Dict con:
        - 'shirt': [H_mean, S_mean, V_mean] para camiseta
        - 'pants': [H_mean, S_mean, V_mean] para pantalón
        - 'combined': [H_shirt, S_shirt, V_shirt, H_pants, S_pants, V_pants]
        - 'color_variance': diferencia entre color de camiseta y pantalón
    """
    x1, y1, x2, y2 = map(int, bbox)
    height = y2 - y1
    width = x2 - x1

    # ROI para camiseta (torso superior)
    shirt_center_x = x1 + width // 2
    shirt_center_y = y1 + int(height * 0.30)
    shirt_roi_width = int(width * 0.5)
    shirt_roi_height = int(height * 0.25)

    shirt_x1 = max(0, shirt_center_x - shirt_roi_width // 2)
    shirt_y1 = max(0, shirt_center_y - shirt_roi_height // 2)
    shirt_x2 = min(frame.shape[1], shirt_center_x + shirt_roi_width // 2)
    shirt_y2 = min(frame.shape[0], shirt_center_y + shirt_roi_height // 2)

    # ROI para pantalón (piernas)
    pants_center_x = x1 + width // 2
    pants_center_y = y1 + int(height * 0.70)
    pants_roi_width = int(width * 0.5)
    pants_roi_height = int(height * 0.20)

    pants_x1 = max(0, pants_center_x - pants_roi_width // 2)
    pants_y1 = max(0, pants_center_y - pants_roi_height // 2)
    pants_x2 = min(frame.shape[1], pants_center_x + pants_roi_width // 2)
    pants_y2 = min(frame.shape[0], pants_center_y + pants_roi_height // 2)

    # Extraer colores de camiseta
    shirt_roi = frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
    if shirt_roi.size == 0:
        shirt_color = np.array([0, 0, 0])
    else:
        shirt_hsv = cv2.cvtColor(shirt_roi, cv2.COLOR_BGR2HSV)
        shirt_color = np.array([
            np.mean(shirt_hsv[:, :, 0]),
            np.mean(shirt_hsv[:, :, 1]),
            np.mean(shirt_hsv[:, :, 2])
        ])

    # Extraer colores de pantalón
    pants_roi = frame[pants_y1:pants_y2, pants_x1:pants_x2]
    if pants_roi.size == 0:
        pants_color = np.array([0, 0, 0])
    else:
        pants_hsv = cv2.cvtColor(pants_roi, cv2.COLOR_BGR2HSV)
        pants_color = np.array([
            np.mean(pants_hsv[:, :, 0]),
            np.mean(pants_hsv[:, :, 1]),
            np.mean(pants_hsv[:, :, 2])
        ])

    # Calcular varianza de color (diferencia entre camiseta y pantalón)
    color_variance = np.linalg.norm(shirt_color - pants_color)

    return {
        'shirt': shirt_color,
        'pants': pants_color,
        'combined': np.concatenate([shirt_color, pants_color]),
        'color_variance': color_variance
    }


def is_in_playing_field(bbox: np.ndarray, frame_width: int, frame_height: int) -> bool:
    """
    Determina si una persona está dentro del área de juego visible.
    Filtra personas en las bandas (banquillos, línea de banda) y áreas no jugables.

    Args:
        bbox: [x1, y1, x2, y2] - Coordenadas del bounding box
        frame_width: Ancho del frame
        frame_height: Alto del frame

    Returns:
        True si la persona está en el área de juego
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    bbox_height = y2 - y1

    # 1. Filtrar zona superior (marcadores, estadísticas, público alto)
    # Solo considerar personas en la mitad/tercio inferior de la imagen
    if center_y < frame_height * 0.15:  # Top 15% del frame (ajustado de 20%)
        return False

    # 2. Filtrar zona muy inferior (publicidad, borde inferior)
    if center_y > frame_height * 0.95:  # Bottom 5% del frame
        return False

    # 3. Filtrar personas muy pequeñas (probablemente en el fondo o público)
    min_height = frame_height * 0.05  # Al menos 5% de la altura (ajustado de 6%)
    if bbox_height < min_height:
        return False

    # 4. Filtrar bandas laterales (banquillos, línea de banda, entrenadores)
    # Área de juego principal: 8% - 92% horizontal (ajustado de 5-95 para excluir DTs)
    if center_x < frame_width * 0.08 or center_x > frame_width * 0.92:
        # Si está en los extremos laterales Y no está muy abajo (córner)
        # Los entrenadores suelen estar en los costados y no en la línea de fondo
        if center_y < frame_height * 0.85:
            return False

    return True


def is_in_goal_area(bbox: np.ndarray, frame_width: int, frame_height: int) -> bool:
    """
    Determina si una persona está en el área de gol (mejorado).
    Área más estricta: 10% de los extremos con verificación de posición vertical.

    Args:
        bbox: [x1, y1, x2, y2] - Coordenadas del bounding box
        frame_width: Ancho del frame
        frame_height: Alto del frame

    Returns:
        True si la persona está en área de gol
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    bbox_height = y2 - y1

    # Área de gol más estricta: 10% de cada lado del campo
    left_goal_area = frame_width * 0.10
    right_goal_area = frame_width * 0.90

    # Verificaciones mejoradas:
    # 1. Debe estar en la zona del campo (no en marcadores o cielo)
    in_field_area = center_y > frame_height * 0.25 and center_y < frame_height * 0.95

    # 2. Debe tener tamaño razonable (no ser una detección muy pequeña en el fondo)
    min_height = frame_height * 0.08  # Al menos 8% de la altura del frame
    is_reasonable_size = bbox_height > min_height

    # 3. En los extremos del campo
    is_in_goal_zone = center_x < left_goal_area or center_x > right_goal_area

    return is_in_goal_zone and in_field_area and is_reasonable_size


def cluster_teams(frame: np.ndarray, detections: sv.Detections, frame_width: int, frame_height: int) -> Tuple[Dict, Dict, List[int], List[int], List[int]]:
    """
    Agrupa jugadores en 2 equipos y detecta árbitros usando K-means clustering mejorado.

    Estrategia mejorada (v3.1):
    1. Filtrar personas FUERA del área de juego (banquillos, director técnico)
    2. Excluir porteros (por posición en área de gol)
    3. Detectar árbitros por score (color típico + varianza) - EXCLUYE ROJO
    4. K-means con 2 clusters solo sobre jugadores (no árbitros ni porteros)
    5. Usar características de camiseta para clustering

    Returns:
        team1_colors, team2_colors, team1_indices, team2_indices, referee_indices
        donde team_colors es un Dict con 'shirt' y 'pants'
    """
    if len(detections) < 4:
        mid = len(detections) // 2
        team1_indices = list(range(mid))
        team2_indices = list(range(mid, len(detections)))
        default_team1 = {'shirt': np.array([90, 100, 100]), 'pants': np.array([0, 50, 50])}
        default_team2 = {'shirt': np.array([0, 100, 100]), 'pants': np.array([0, 50, 50])}
        return default_team1, default_team2, team1_indices, team2_indices, []

    # Paso 1: Filtrar personas FUERA del área de juego (banquillos, bancas)
    in_field_indices = []
    out_of_field_indices = []
    for idx, bbox in enumerate(detections.xyxy):
        if is_in_playing_field(bbox, frame_width, frame_height):
            in_field_indices.append(idx)
        else:
            out_of_field_indices.append(idx)

    # Paso 2: Identificar porteros por posición (solo de los que están en campo)
    goalkeeper_indices_set = set()
    for idx in in_field_indices:
        bbox = detections.xyxy[idx]
        if is_in_goal_area(bbox, frame_width, frame_height):
            x_center = (bbox[0] + bbox[2]) / 2
            if x_center < frame_width * 0.05 or x_center > frame_width * 0.95:
                goalkeeper_indices_set.add(idx)

    # Paso 3: Extraer características de color de personas en campo (NO-porteros)
    color_data = []
    non_gk_indices = []

    for idx in in_field_indices:
        if idx not in goalkeeper_indices_set:
            bbox = detections.xyxy[idx]
            features = extract_color_features(frame, bbox)
            color_data.append({
                'idx': idx,
                'features': features,
                'shirt': features['shirt'],
                'pants': features['pants'],
                'variance': features['color_variance']
            })
            non_gk_indices.append(idx)

    if len(color_data) < 3:
        mid = len(non_gk_indices) // 2
        team1_indices = non_gk_indices[:mid]
        team2_indices = non_gk_indices[mid:]
        default_team1 = {'shirt': np.array([90, 100, 100]), 'pants': np.array([0, 50, 50])}
        default_team2 = {'shirt': np.array([0, 100, 100]), 'pants': np.array([0, 50, 50])}
        return default_team1, default_team2, team1_indices, team2_indices, []

    # Paso 3: Detectar árbitros por alta varianza de color (algoritmo mejorado v3.2)
    # Los árbitros suelen tener colores muy diferentes entre camiseta y pantalón
    variances = np.array([d['variance'] for d in color_data])

    # Usar percentil más estricto para evitar falsos positivos
    if len(variances) > 5:
        variance_threshold = np.percentile(variances, 80)  # Top 20% de varianza (antes 70%)
    else:
        variance_threshold = np.median(variances) + np.std(variances) * 0.7

    # También considerar si el color es muy diferente del promedio general
    all_shirt_colors = np.array([d['shirt'] for d in color_data])
    mean_shirt_color = np.mean(all_shirt_colors, axis=0)
    std_shirt_color = np.std(all_shirt_colors, axis=0)

    # Análisis HSV específico para árbitros
    # Los árbitros suelen usar negro, amarillo neón, o verde brillante
    referee_candidates = []
    player_candidates = []

    for data in color_data:
        shirt_hsv = data['shirt']
        pants_hsv = data['pants']
        is_high_variance = data['variance'] > max(variance_threshold, 45)  # Aumentado de 35 a 45
        shirt_distance_from_mean = np.linalg.norm(data['shirt'] - mean_shirt_color)

        # Detectar colores típicos de árbitros en HSV (MEJORADO v3.2)
        h, s, v = shirt_hsv
        h_pants, s_pants, v_pants = pants_hsv

        # Colores de camiseta típicos de árbitros
        is_black_shirt = v < 70 and s < 70  # Negro/gris oscuro
        is_yellow_shirt = 20 < h < 35 and s > 100 and v > 130  # Amarillo brillante
        is_bright_green_shirt = 40 < h < 75 and s > 110 and v > 110  # Verde lima

        # EXCLUIR BLANCO - Los equipos usan blanco, árbitros rara vez
        is_white_shirt = v > 180 and s < 50  # Blanco (V alto, S bajo)

        # EXCLUIR ROJO - Los equipos pueden usar rojo
        is_red_shirt = (h < 10 or h > 170) and s > 80 and v > 80

        # Solo considerar colores de árbitro si NO es rojo NI blanco
        is_ref_color = (is_black_shirt or is_yellow_shirt or is_bright_green_shirt) and not is_red_shirt and not is_white_shirt

        # Detectar pantalón negro (típico de árbitros)
        is_black_pants = v_pants < 80 and s_pants < 80

        # Calcular score de árbitro (MÁS ESTRICTO)
        referee_score = 0

        # Varianza alta es fuerte indicador (camiseta diferente de pantalón)
        if is_high_variance:
            referee_score += 3  # Aumentado de 2 a 3

        # Color típico de árbitro (amarillo/verde/negro) en camiseta
        if is_ref_color:
            referee_score += 3  # Aumentado de 2 a 3

        # Pantalón negro + camiseta de color → típico árbitro
        if is_black_pants and (is_yellow_shirt or is_bright_green_shirt):
            referee_score += 2  # Bonus por combinación típica

        # Muy diferente del promedio del grupo
        if shirt_distance_from_mean > 75:  # Aumentado de 65 a 75
            referee_score += 1
        if shirt_distance_from_mean > 95:  # Aumentado de 85 a 95
            referee_score += 1

        # Penalización por colores de jugador
        if is_white_shirt:
            referee_score -= 3  # Penalizar blanco fuertemente

        # Score mínimo más alto para clasificar como árbitro
        # Antes: >= 3, Ahora: >= 5
        if referee_score >= 5:
            referee_candidates.append(data)
        else:
            player_candidates.append(data)

    # Limitar árbitros a máximo 4 personas (usualmente 1-3 en cámara)
    if len(referee_candidates) > 4:
        # Ordenar por score combinado (varianza + distancia)
        referee_candidates.sort(
            key=lambda x: x['variance'] + np.linalg.norm(x['shirt'] - mean_shirt_color),
            reverse=True
        )
        extra_players = referee_candidates[4:]
        referee_candidates = referee_candidates[:4]
        player_candidates.extend(extra_players)

    # Si no hay suficientes jugadores, reclasificar
    if len(player_candidates) < 4:
        player_candidates.extend(referee_candidates)
        referee_candidates = []

    referee_indices = [d['idx'] for d in referee_candidates]

    # Paso 4: Clustering solo sobre jugadores usando color de camiseta
    if len(player_candidates) < 2:
        default_team1 = {'shirt': np.array([90, 100, 100]), 'pants': np.array([0, 50, 50])}
        default_team2 = {'shirt': np.array([0, 100, 100]), 'pants': np.array([0, 50, 50])}
        return default_team1, default_team2, [], [], referee_indices

    player_shirt_colors = np.array([d['shirt'] for d in player_candidates])

    # K-means con 2 clusters (los 2 equipos)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(player_shirt_colors)

    # Separar jugadores por equipo
    team1_data = [player_candidates[i] for i in range(len(player_candidates)) if labels[i] == 0]
    team2_data = [player_candidates[i] for i in range(len(player_candidates)) if labels[i] == 1]

    team1_indices = [d['idx'] for d in team1_data]
    team2_indices = [d['idx'] for d in team2_data]

    # Calcular colores representativos de cada equipo (camiseta y pantalón por separado)
    team1_shirt_color = kmeans.cluster_centers_[0]
    team2_shirt_color = kmeans.cluster_centers_[1]

    # Calcular color promedio de pantalón para cada equipo
    team1_pants_colors = np.array([d['pants'] for d in team1_data]) if team1_data else np.array([[0, 50, 50]])
    team2_pants_colors = np.array([d['pants'] for d in team2_data]) if team2_data else np.array([[0, 50, 50]])

    team1_pants_color = np.mean(team1_pants_colors, axis=0)
    team2_pants_color = np.mean(team2_pants_colors, axis=0)

    team1_colors = {'shirt': team1_shirt_color, 'pants': team1_pants_color}
    team2_colors = {'shirt': team2_shirt_color, 'pants': team2_pants_color}

    return team1_colors, team2_colors, team1_indices, team2_indices, referee_indices


def classify_person_smart(
    frame: np.ndarray,
    bbox: np.ndarray,
    team1_colors: Dict,
    team2_colors: Dict,
    frame_width: int,
    frame_height: int
) -> Tuple[str, int]:
    """
    Clasifica una persona como jugador de equipo 1, equipo 2, árbitro o portero.

    Mejoras v3.1:
    - Excluye ROJO de colores típicos de árbitros
    - Score mejorado para clasificación de árbitros
    - Mejor distinción entre equipos con camisetas coloridas

    Args:
        team1_colors: Dict con 'shirt' y 'pants' del equipo 1
        team2_colors: Dict con 'shirt' y 'pants' del equipo 2

    Returns:
        ('team1'|'team2'|'referee'|'goalkeeper', team_number)
    """
    features = extract_color_features(frame, bbox)
    person_shirt = features['shirt']
    person_pants = features['pants']
    color_variance = features['color_variance']

    in_goal_area = is_in_goal_area(bbox, frame_width, frame_height)

    # Si está en área de gol cerca de los extremos, probablemente es portero
    # Ampliado a 12% para cubrir mejor el área chica
    if in_goal_area:
        x_center = (bbox[0] + bbox[2]) / 2
        if x_center < frame_width * 0.12:
            return ('goalkeeper', 1)
        elif x_center > frame_width * 0.88:
            return ('goalkeeper', 2)

    # Calcular distancias a colores de equipos (usando camiseta principalmente)
    dist_team1_shirt = np.linalg.norm(person_shirt - team1_colors['shirt'])
    dist_team2_shirt = np.linalg.norm(person_shirt - team2_colors['shirt'])

    # Detectar árbitros con criterio mejorado v3.3:
    # 1. Alta varianza de color (camiseta diferente de pantalón)
    # 2. Color muy diferente de ambos equipos
    # 3. Patrones de color típicos de árbitros
    # 4. EXCLUIR blanco y rojo explícitamente (con umbrales relajados para rojo)
    min_team_dist = min(dist_team1_shirt, dist_team2_shirt)

    # Umbrales más estrictos
    high_variance_threshold = 50  # Aumentado de 45 a 50
    outlier_threshold = 75  # Aumentado de 70 a 75

    # Detectar colores típicos de árbitros (MEJORADO v3.2)
    h, s, v = person_shirt
    h_pants, s_pants, v_pants = person_pants

    # Colores de camiseta
    is_black_shirt = v < 70 and s < 70
    is_yellow_shirt = 20 < h < 35 and s > 100 and v > 130
    is_bright_green_shirt = 40 < h < 75 and s > 110 and v > 110

    # EXCLUIR BLANCO - Equipos usan blanco frecuentemente
    is_white_shirt = v > 180 and s < 50

    # EXCLUIR ROJO - Los equipos pueden usar rojo
    # Rango de Hue para rojo: 0-10 y 160-180
    # Relajamos S y V a > 50 para capturar rojos oscuros o desaturados
    is_red_shirt = (h < 15 or h > 165) and s > 50 and v > 50

    # Solo considerar colores de árbitro si NO es rojo NI blanco
    is_ref_color = (is_black_shirt or is_yellow_shirt or is_bright_green_shirt) and not is_red_shirt and not is_white_shirt

    # Detectar pantalón negro (típico de árbitros)
    is_black_pants = v_pants < 80 and s_pants < 80

    # Calcular score de árbitro (MÁS ESTRICTO)
    referee_score = 0

    # Varianza alta
    if color_variance > high_variance_threshold:
        referee_score += 3  # Aumentado de 2 a 3

    # Color típico de árbitro
    if is_ref_color:
        referee_score += 3  # Aumentado de 2 a 3

    # Combinación típica: camiseta de color + pantalón negro
    if is_black_pants and (is_yellow_shirt or is_bright_green_shirt):
        referee_score += 2

    # Muy diferente de ambos equipos
    if min_team_dist > outlier_threshold:
        referee_score += 2
    if min_team_dist > 95:  # Aumentado de 85 a 95
        referee_score += 1

    # Penalización por colores de jugador
    if is_white_shirt:
        referee_score -= 3  # Penalizar blanco fuertemente

    # Score mínimo más alto: >= 6 (antes era >= 4)
    is_likely_referee = referee_score >= 6

    if is_likely_referee:
        # Doble verificación: si es rojo, NO es árbitro (prioridad a equipo rojo)
        if is_red_shirt:
            # Si parece árbitro pero es rojo, forzar clasificación por distancia
            if dist_team1_shirt < dist_team2_shirt:
                return ('team1', 1)
            else:
                return ('team2', 2)
        return ('referee', 0)

    # Clasificar en equipos basándose en color de camiseta
    if dist_team1_shirt < dist_team2_shirt:
        return ('team1', 1)
    else:
        return ('team2', 2)


def process_video(
    source_path: str,
    target_path: str,
    player_model,
    ball_model=None,
    pitch_model=None,
    conf: float = 0.3,
    detection_mode: str = "players_and_ball",
    img_size: int = 640,
    full_field_approx: bool = False
):
    """
    Procesa video completo con detección y tracking mejorado usando múltiples modelos.

    Args:
        source_path: Ruta al video de entrada
        target_path: Ruta al video de salida
        player_model: Modelo YOLO para detección de jugadores
        ball_model: Modelo YOLO para detección de pelota (opcional)
        pitch_model: Modelo YOLO para detección de campo (opcional)
        conf: Umbral de confianza para detecciones de personas
        detection_mode: Modo de detección ('players_only', 'ball_only', 'players_and_ball')
        img_size: Tamaño de imagen para inferencia
        full_field_approx: Si True, asume que la imagen completa es el campo (experimental)
    """
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

    # Trackers
    person_tracker = sv.ByteTrack()
    ball_tracker = sv.ByteTrack()

    # Configurar modelo de pitch si se proporciona O si se usa aproximación
    pitch_config = None
    if pitch_model or full_field_approx:
        pitch_config = SoccerPitchConfiguration()

    # Anotadores
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

    # Variables de estado
    team1_colors = None
    team2_colors = None
    frame_count = 0
    reference_team1_color = None
    reference_team2_color = None
    clustering_initialized = False
    track_history = {}
    HISTORY_LEN = 30

    try:
        while True:
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
            # Asumimos que el modelo de jugadores es COCO (clase 0) o Custom (probablemente clase 0)
            # Si es custom con solo 1 clase, class_id será 0.
            # Si es COCO, filtramos class_id == 0.
            if player_detections.class_id is not None:
                player_detections = player_detections[player_detections.class_id == 0]
            
            # --- DETECCIÓN DE PELOTA ---
            ball_detections = sv.Detections.empty()
            if detection_mode in ["ball_only", "players_and_ball"]:
                if ball_model:
                    # Usar modelo específico de pelota
                    ball_results = ball_model.predict(
                        frame,
                        conf=max(0.15, conf * 0.4), # Umbral más bajo para pelota
                        iou=0.3,
                        imgsz=img_size,
                        verbose=False
                    )
                    ball_detections = sv.Detections.from_ultralytics(ball_results[0])
                    # No filtramos clase aquí, asumimos que el modelo de pelota detecta la pelota
                else:
                    # Fallback: Usar modelo de jugadores si tiene clase 32 (sports ball)
                    # Necesitamos correr predict otra vez? No, ya corrimos.
                    # Pero arriba filtramos class_id == 0.
                    # Recuperemos del raw result si es necesario.
                    raw_detections = sv.Detections.from_ultralytics(player_results[0])
                    if raw_detections.class_id is not None:
                         ball_detections = raw_detections[raw_detections.class_id == 32]
                         # Filtrar por confianza más baja si es necesario
                         if len(ball_detections) > 0:
                             ball_conf_threshold = max(0.15, conf * 0.4)
                             ball_detections = ball_detections[ball_detections.confidence >= ball_conf_threshold]

            # --- TRACKING DE JUGADORES ---
            tracked_persons = person_tracker.update_with_detections(player_detections)
            
            # Actualizar modelos de color (Clustering)
            if (frame_count % 45 == 1 or team1_colors is None) and len(player_detections) > 0:
                team1_colors_new, team2_colors_new, _, _, _ = cluster_teams(
                    frame, player_detections, width, height
                )

                if not clustering_initialized:
                    reference_team1_color = team1_colors_new['shirt'].copy()
                    reference_team2_color = team2_colors_new['shirt'].copy()
                    team1_colors = team1_colors_new
                    team2_colors = team2_colors_new
                    clustering_initialized = True
                else:
                    # Verificar intercambio
                    dist_t1_ref1 = np.linalg.norm(team1_colors_new['shirt'] - reference_team1_color)
                    dist_t1_ref2 = np.linalg.norm(team1_colors_new['shirt'] - reference_team2_color)
                    dist_t2_ref1 = np.linalg.norm(team2_colors_new['shirt'] - reference_team1_color)
                    dist_t2_ref2 = np.linalg.norm(team2_colors_new['shirt'] - reference_team2_color)

                    score_keep = dist_t1_ref1 + dist_t2_ref2
                    score_swap = dist_t1_ref2 + dist_t2_ref1

                    if score_swap < score_keep:
                        team1_colors = team2_colors_new
                        team2_colors = team1_colors_new
                    else:
                        team1_colors = team1_colors_new
                        team2_colors = team2_colors_new
                    
                    alpha = 0.1
                    reference_team1_color = (1 - alpha) * reference_team1_color + alpha * team1_colors['shirt']
                    reference_team2_color = (1 - alpha) * reference_team2_color + alpha * team2_colors['shirt']

            # Clasificar cada persona rastreada
            team1_mask = []
            team2_mask = []
            referee_mask = []
            goalkeeper_mask = []
            
            if len(tracked_persons) > 0 and team1_colors is not None:
                for i, (xyxy, _, _, _, tracker_id, _) in enumerate(tracked_persons):
                    if not is_in_playing_field(xyxy, width, height):
                        goalkeeper_mask.append(False)
                        team1_mask.append(False)
                        team2_mask.append(False)
                        referee_mask.append(False)
                        continue

                    in_goal = is_in_goal_area(xyxy, width, height)
                    x_center = (xyxy[0] + xyxy[2]) / 2
                    
                    if in_goal and (x_center < width * 0.05 or x_center > width * 0.95):
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
            
            # Anotar
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

            # --- TRACKING DE PELOTA ---
            tracked_ball = None
            if len(ball_detections) > 0:
                tracked_ball = ball_tracker.update_with_detections(ball_detections)
                annotated_frame = ball_annotator.annotate(scene=annotated_frame, detections=tracked_ball)
                ball_labels = ["BALL"] * len(tracked_ball)
                annotated_frame = ball_label_annotator.annotate(scene=annotated_frame, detections=tracked_ball, labels=ball_labels)

            # --- RADAR ---
            if pitch_config:  # Si hay configuración de pitch (ya sea por modelo o approx)
                transformer = None
                
                # Caso A: Modelo de Pitch disponible
                if pitch_model:
                    try:
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
                    except Exception as e:
                        print(f"Error en inferencia de pitch: {e}")

                # Caso B: Aproximación de Campo Completo (Experimental)
                elif full_field_approx:
                    # Asumimos que los 4 bordes de la imagen son los 4 bordes del campo
                    # Keypoints estándar (0: TL, 1: TR, 2: BR, 3: BL)
                    source_points = np.array([
                        [0, 0],         # Top-Left
                        [width, 0],     # Top-Right
                        [width, height],# Bottom-Right
                        [0, height]     # Bottom-Left
                    ])
                    target_points = np.array([
                        pitch_config.keypoints_map[0],
                        pitch_config.keypoints_map[1],
                        pitch_config.keypoints_map[2],
                        pitch_config.keypoints_map[3]
                    ])
                    transformer = ViewTransformer(source_points, target_points)

                # Si tenemos un transformer válido, proyectamos y dibujamos
                if transformer:
                    try:
                        points_to_transform = {}
                        
                        def get_bottom_center(dets):
                            return np.column_stack([
                                (dets.xyxy[:, 0] + dets.xyxy[:, 2]) / 2,
                                dets.xyxy[:, 3]
                            ])

                        if any(team1_mask):
                            points_to_transform['team1'] = transformer.transform_points(
                                get_bottom_center(tracked_persons[np.array(team1_mask)])
                            )
                        if any(team2_mask):
                            points_to_transform['team2'] = transformer.transform_points(
                                get_bottom_center(tracked_persons[np.array(team2_mask)])
                            )
                        if any(referee_mask):
                            points_to_transform['referee'] = transformer.transform_points(
                                get_bottom_center(tracked_persons[np.array(referee_mask)])
                            )
                        if any(goalkeeper_mask):
                            points_to_transform['goalkeeper'] = transformer.transform_points(
                                get_bottom_center(tracked_persons[np.array(goalkeeper_mask)])
                            )
                        if tracked_ball is not None and len(tracked_ball) > 0:
                            points_to_transform['ball'] = transformer.transform_points(
                                get_bottom_center(tracked_ball)
                            )
                            
                        radar_view = draw_radar_view(pitch_config, points_to_transform)
                        
                        scale_factor = 0.3
                        new_w = int(width * scale_factor)
                        aspect_ratio = radar_view.shape[0] / radar_view.shape[1]
                        new_h = int(new_w * aspect_ratio)
                        
                        radar_resized = cv2.resize(radar_view, (new_w, new_h))
                        
                        offset_x = (width - new_w) // 2
                        offset_y = height - new_h - 20
                        
                        if offset_y > 0:
                            annotated_frame[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = radar_resized
                    except Exception as e:
                        print(f"Error dibujando radar: {e}")

            out.write(annotated_frame)

    finally:
        cap.release()
        out.release()
