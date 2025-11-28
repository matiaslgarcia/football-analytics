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
    def __init__(self, width=105, length=68, margins=5):
        self.width = width  # meters (X-axis, Touch line length)
        self.length = length # meters (Y-axis, Goal line length)
        self.margins = margins
        
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
                self._set_default_keypoints()
        else:
            self._set_default_keypoints()

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
            
            radius = 6 if category == 'ball' else 10
            cv2.circle(pitch_img, (x, y), radius, color, -1)
            cv2.circle(pitch_img, (x, y), radius, (0, 0, 0), 1) # Outline
            
    return pitch_img
