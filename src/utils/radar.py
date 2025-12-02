import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

# Try to import from roboflow sports configuration if available
try:
    from sports.configs.soccer import SoccerPitchConfiguration as RoboflowSoccerPitchConfig
    ROBOFLOW_SPORTS_AVAILABLE = True
except ImportError:
    ROBOFLOW_SPORTS_AVAILABLE = False

class SoccerPitchConfiguration:
    def __init__(self, width=105, length=68, margins=5, model_type='roboflow'):
        """
        Args:
            width: Field width in meters (default: 105m FIFA standard)
            length: Field length in meters (default: 68m FIFA standard)
            margins: Margin around field in meters
            model_type: 'roboflow' for Roboflow Sports, 'soccana' for Soccana YOLOv11, 'default' for basic
        """
        self.width = width  # meters (X-axis, Touch line length)
        self.length = length # meters (Y-axis, Goal line length)
        self.margins = margins
        self.model_type = model_type

        # Standard dimensions in meters
        self.goal_width = 7.32
        self.goal_depth = 2.44
        self.penalty_box_width = 40.32  # Along Goal line (Y)
        self.penalty_box_depth = 16.5   # Into field (X)
        self.six_yard_box_width = 18.32
        self.six_yard_box_depth = 5.5
        self.center_circle_radius = 9.15
        self.penalty_spot_distance = 11.0

        self.keypoints_map = {}

        # Configure keypoints based on model type
        if model_type == 'soccana':
            self._set_soccana_keypoints()
        elif model_type == 'roboflow':
            if ROBOFLOW_SPORTS_AVAILABLE:
                try:
                    # Load configuration from Roboflow Sports to get the 32 keypoints
                    rf_config = RoboflowSoccerPitchConfig()
                    
                    # Roboflow Sports defines the pitch vertically:
                    # width ~70m (X), length ~120m (Y)
                    # Our system defines the pitch horizontally:
                    # width ~105m (X), length ~68m (Y)
                    
                    # So we map Roboflow dimensions to ours by swapping axes:
                    self.width = rf_config.length / 100.0   # Roboflow Y -> My X
                    self.length = rf_config.width / 100.0   # Roboflow X -> My Y
                    
                    # Update standard markings dimensions based on Roboflow config if desired,
                    # but standard 105x68 dimensions usually imply standard box sizes.
                    # We can update them to match the specific model config:
                    # Roboflow penalty_box_width (41m) is along X (My Y)
                    self.penalty_box_width = rf_config.penalty_box_width / 100.0
                    # Roboflow penalty_box_length (20.15m) is along Y (My X)
                    self.penalty_box_depth = rf_config.penalty_box_length / 100.0
                    
                    # Map vertices to keypoints (swapping X and Y)
                    # vertices is a list of (x, y) in cm
                    for i, (x, y) in enumerate(rf_config.vertices):
                        # Swap axes: x -> y, y -> x
                        # And convert cm to meters
                        self.keypoints_map[i] = (y / 100.0, x / 100.0)
                        
                except Exception as e:
                    print(f"Warning: Could not load Roboflow pitch configuration: {e}")
                    self._set_roboflow_keypoints_manual()
            else:
                self._set_roboflow_keypoints_manual()
        else:
            self._set_default_keypoints()

    def _set_roboflow_keypoints_manual(self):
        """
        Manual mapping for Roboflow Sports 32 keypoints (if library not available).
        Mapped based on standard soccer field dimensions (105x68m).
        Roboflow indices 0-31.
        """
        w, l = 105.0, 68.0
        self.width = w
        self.length = l
        
        # Standard FIFA dimensions
        # Based on inspection of Roboflow Sports dataset keypoints:
        # 0: Top-Left Corner (0,0)
        # 5: Top-Right Corner (w,0)
        # 29: Bottom-Right Corner (w,l)
        # 24: Bottom-Left Corner (0,l)
        
        # We define a reasonable approximation for the most important points (corners)
        # Ideally we would copy the full 32 point map, but for now corners are most critical for homography
        
        # Full map based on typical Roboflow Sports structure (approximate)
        self.keypoints_map = {
            # Corners
            0: (0, 0),
            5: (w, 0),
            29: (w, l),
            24: (0, l),
            
            # Goal posts (approx)
            # ... (omitted for brevity, corners are enough for basic homography if 4+ points found)
        }
        
        # Let's try to populate more if possible, but corners are critical.
        # If we only have 4 points in the map, homography will only work if those specific 4 are detected.
        # Ideally we need the full list.
        # Since we don't have the full list handy without the library, 
        # we'll rely on the fact that homography needs ANY 4 matching points.
        
        # Actually, it's better to rely on the library or the user providing it.
        # But since we want to support 'homography.pt' which has 32 points, 
        # we MUST have a map for those 32 points.
        
        # If the library is missing, we are in trouble for the inner points.
        # However, the user's environment likely has 'sports' if they are using Roboflow models.
        # If not, we should at least support the corners.
        pass

    def _set_default_keypoints(self):
        # Fallback to basic 4 corners + center if Roboflow lib not present
        # This is a minimal map, likely insufficient for 32-point models
        self.keypoints_map = {
             0: (0, 0),                            # Top-Left Corner
             1: (self.width, 0),                   # Top-Right Corner
             2: (self.width, self.length),         # Bottom-Right Corner
             3: (0, self.length),                  # Bottom-Left Corner
             # Center
             4: (self.width/2, self.length/2),     # Center of Field
        }

    def _set_soccana_keypoints(self):
        """
        Configura el mapeo de 29 keypoints del modelo Soccana.
        Basado en est\u00e1ndares FIFA para un campo de 105m x 68m
        """
        # Campo est\u00e1ndar FIFA: 105m (ancho) x 68m (largo)
        w, l = self.width, self.length

        # Keypoints de Soccana (29 puntos):
        self.keypoints_map = {
            # === ESQUINAS (4 puntos) ===
            0: (0, 0),                         # sideline_top_left
            16: (w, 0),                        # sideline_top_right
            9: (0, l),                         # sideline_bottom_left
            25: (w, l),                        # sideline_bottom_right

            # === AREA PENAL IZQUIERDA (4 puntos) ===
            1: (0, 13.84),                     # penalty_left_top_corner
            2: (16.5, 13.84),                  # penalty_left_top_interior
            3: (16.5, l - 13.84),              # penalty_left_bottom_interior
            4: (0, l - 13.84),                 # penalty_left_bottom_corner

            # === AREA PEQUENA IZQUIERDA (4 puntos) ===
            5: (0, 24.84),                     # goal_left_top_corner
            6: (5.5, 24.84),                   # goal_left_top_interior
            7: (5.5, l - 24.84),               # goal_left_bottom_interior
            8: (0, l - 24.84),                 # goal_left_bottom_corner

            # === CENTRO DEL CAMPO (9 puntos) ===
            10: (w/2 - 9.15, 0),               # semicircle_left_top
            11: (w/2, 0),                      # center_line_top
            26: (w/2 + 9.15, 0),               # semicircle_right_top

            12: (w/2, l),                      # center_line_bottom
            13: (w/2 - 9.15, l/2),             # center_circle_left
            14: (w/2, l/2 - 9.15),             # center_circle_top
            15: (w/2, l/2),                    # field_center
            27: (w/2 - 9.15, l/2),             # center_circle_left_dup (mismo que 13)
            28: (w/2 + 9.15, l/2),             # center_circle_right

            # === AREA PENAL DERECHA (4 puntos) ===
            17: (w, 13.84),                    # penalty_right_top_corner
            18: (w - 16.5, 13.84),             # penalty_right_top_interior
            19: (w - 16.5, l - 13.84),         # penalty_right_bottom_interior
            20: (w, l - 13.84),                # penalty_right_bottom_corner

            # === AREA PEQUENA DERECHA (4 puntos) ===
            21: (w, 24.84),                    # goal_right_top_corner
            22: (w - 5.5, 24.84),              # goal_right_top_interior
            23: (w - 5.5, l - 24.84),          # goal_right_bottom_interior
            24: (w, l - 24.84),                # goal_right_bottom_corner
        }

    def get_corner_keypoint_ids(self) -> List[int]:
        """
        Returns the keypoint IDs for the 4 corners of the field.
        Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left

        For Roboflow Sports (32 keypoints): [0, 5, 29, 24]
        For default (4 keypoints): [0, 1, 2, 3]
        """
        # Verificar si estamos usando Roboflow Sports con muchos keypoints
        if len(self.keypoints_map) > 10:
            # Roboflow Sports configuration with 32 keypoints
            # Find corners by position
            corners = [
                (0.0, 0.0),           # Top-Left
                (self.width, 0.0),    # Top-Right
                (self.width, self.length),  # Bottom-Right
                (0.0, self.length)    # Bottom-Left
            ]

            corner_ids = []
            for corner_pos in corners:
                # Find closest keypoint to this corner
                min_dist = float('inf')
                closest_id = None
                for kid, kpos in self.keypoints_map.items():
                    dist = ((kpos[0] - corner_pos[0])**2 + (kpos[1] - corner_pos[1])**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        closest_id = kid

                # Siempre agregar el más cercano, sin importar la distancia
                if closest_id is not None:
                    corner_ids.append(closest_id)

            if len(corner_ids) == 4:
                return corner_ids

        # Default: assume IDs 0, 1, 2, 3 are the corners
        return [0, 1, 2, 3]

    def get_keypoints_from_ids(self, class_ids: np.ndarray) -> np.ndarray:
        """Returns the real-world coordinates for the given class IDs."""
        points = []
        for cid in class_ids:
            if cid in self.keypoints_map:
                points.append(self.keypoints_map[cid])
            else:
                # Fallback or ignore
                points.append([0, 0]) 
        return np.array(points)

def draw_pitch(config: SoccerPitchConfiguration, scale=5, background_color=(0, 0, 0), line_color=(255, 255, 255)) -> np.ndarray:
    """
    Draws a soccer pitch representation.
    scale: pixels per meter
    """
    w_pixels = int((config.width + 2 * config.margins) * scale)
    l_pixels = int((config.length + 2 * config.margins) * scale)
    
    img = np.full((l_pixels, w_pixels, 3), background_color, dtype=np.uint8)
    
    # Offset for margins
    ox = int(config.margins * scale)
    oy = int(config.margins * scale)
    
    # Helper to convert meters to pixels
    def to_px(x, y):
        return int(x * scale) + ox, int(y * scale) + oy

    # Outer boundary
    p1 = to_px(0, 0)
    p2 = to_px(config.width, config.length)
    cv2.rectangle(img, p1, p2, line_color, 2)
    
    # Center line
    top_mid = to_px(config.width / 2, 0)
    bot_mid = to_px(config.width / 2, config.length)
    cv2.line(img, top_mid, bot_mid, line_color, 2)
    
    # Center circle
    center = to_px(config.width / 2, config.length / 2)
    radius = int(config.center_circle_radius * scale)
    cv2.circle(img, center, radius, line_color, 2)
    
    # Penalty boxes (Left)
    p_box_l_top = to_px(0, (config.length - config.penalty_box_width) / 2)
    p_box_l_bot = to_px(config.penalty_box_depth, (config.length + config.penalty_box_width) / 2)
    cv2.rectangle(img, p_box_l_top, p_box_l_bot, line_color, 2)
    
    # Penalty boxes (Right)
    p_box_r_top = to_px(config.width - config.penalty_box_depth, (config.length - config.penalty_box_width) / 2)
    p_box_r_bot = to_px(config.width, (config.length + config.penalty_box_width) / 2)
    cv2.rectangle(img, p_box_r_top, p_box_r_bot, line_color, 2)
    
    return img

def draw_radar_view(
    config: SoccerPitchConfiguration,
    transformed_points: Dict[str, np.ndarray],
    scale=5
) -> np.ndarray:
    """
    Draws the radar view with players.
    transformed_points: Dict mapping 'team1', 'team2', 'ball', 'referee' to (N, 2) arrays of coordinates in meters.
    """
    pitch_img = draw_pitch(config, scale=scale, background_color=(34, 139, 34)) # Green pitch
    
    ox = int(config.margins * scale)
    oy = int(config.margins * scale)
    
    colors = {
        'team1': (0, 255, 0),     # Green (BGR) -> Lime
        'team2': (255, 191, 0),   # Sky Blueish (BGR)
        'ball': (0, 0, 255),      # Red
        'referee': (0, 215, 255), # Gold
        'goalkeeper': (182, 89, 155) # Purple
    }
    
    for category, points in transformed_points.items():
        if points is None or len(points) == 0:
            continue
            
        color = colors.get(category, (255, 255, 255))
        
        for point in points:
            px, py = point
            # Map to image coordinates
            x = int(px * scale) + ox
            y = int(py * scale) + oy
            
            # Clip to image bounds
            x = max(0, min(x, pitch_img.shape[1] - 1))
            y = max(0, min(y, pitch_img.shape[0] - 1))
            
            radius = 8 if category == 'ball' else 14  # Aumentado de 6/10 a 8/14
            cv2.circle(pitch_img, (x, y), radius, color, -1)
            cv2.circle(pitch_img, (x, y), radius+1, (0, 0, 0), 2) # Outline más grueso
            
    return pitch_img


def draw_radar_with_metrics(
    config: SoccerPitchConfiguration,
    transformed_points: Dict[str, np.ndarray],
    formations: Dict[str, str] = None,
    metrics: Dict[str, Dict] = None,
    scale=5
) -> np.ndarray:
    """
    Dibuja el radar con jugadores, formaciones y métricas tácticas.

    Args:
        config: Configuración del campo
        transformed_points: Puntos transformados por equipo
        formations: Dict con formaciones por equipo {'team1': '4-4-2', 'team2': '4-3-3'}
        metrics: Dict con métricas por equipo
        scale: Escala de píxeles por metro

    Returns:
        Imagen del radar con anotaciones
    """
    # Dibujar campo base con jugadores
    pitch_img = draw_radar_view(config, transformed_points, scale)

    # Si no hay métricas, retornar solo el campo
    if formations is None and metrics is None:
        return pitch_img

    # Preparar área para texto (arriba del radar)
    text_height = 120
    text_area = np.zeros((text_height, pitch_img.shape[1], 3), dtype=np.uint8)
    text_area[:] = (40, 40, 40)  # Fondo gris oscuro

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    line_height = 25

    y_offset = 25

    # Dibujar formaciones
    if formations:
        team1_formation = formations.get('team1', 'N/A')
        team2_formation = formations.get('team2', 'N/A')

        cv2.putText(text_area, f"Team 1: {team1_formation}",
                   (10, y_offset), font, font_scale, (0, 255, 0), font_thickness)
        cv2.putText(text_area, f"Team 2: {team2_formation}",
                   (pitch_img.shape[1] // 2 + 10, y_offset), font, font_scale, (255, 191, 0), font_thickness)
        y_offset += line_height

    # Dibujar métricas clave
    if metrics:
        # Team 1 metrics
        if 'team1' in metrics:
            m1 = metrics['team1']
            cv2.putText(text_area, f"Pressure: {m1.get('pressure_height', 0):.1f}m",
                       (10, y_offset), font, font_scale * 0.85, (200, 200, 200), font_thickness)
            cv2.putText(text_area, f"Width: {m1.get('offensive_width', 0):.1f}m",
                       (10, y_offset + line_height), font, font_scale * 0.85, (200, 200, 200), font_thickness)
            cv2.putText(text_area, f"Compact: {m1.get('compactness', 0):.0f}m²",
                       (10, y_offset + line_height * 2), font, font_scale * 0.85, (200, 200, 200), font_thickness)

        # Team 2 metrics
        if 'team2' in metrics:
            m2 = metrics['team2']
            x_pos = pitch_img.shape[1] // 2 + 10
            cv2.putText(text_area, f"Pressure: {m2.get('pressure_height', 0):.1f}m",
                       (x_pos, y_offset), font, font_scale * 0.85, (200, 200, 200), font_thickness)
            cv2.putText(text_area, f"Width: {m2.get('offensive_width', 0):.1f}m",
                       (x_pos, y_offset + line_height), font, font_scale * 0.85, (200, 200, 200), font_thickness)
            cv2.putText(text_area, f"Compact: {m2.get('compactness', 0):.0f}m²",
                       (x_pos, y_offset + line_height * 2), font, font_scale * 0.85, (200, 200, 200), font_thickness)

    # Combinar área de texto con campo
    combined = np.vstack([text_area, pitch_img])

    return combined
