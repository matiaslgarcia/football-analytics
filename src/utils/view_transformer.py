import numpy as np
import cv2

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        """
        Initialize ViewTransformer with source and target points.
        
        Args:
            source: Array of shape (N, 2) containing source points (pixels).
            target: Array of shape (N, 2) containing target points (meters/world coordinates).
        """
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        # Usar RANSAC para robustez si hay suficientes puntos
        if len(source) >= 4:
            self.m, _ = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
        else:
            self.m = None

    def transform_points(self, points: np.ndarray, flip_x: bool = False) -> np.ndarray:
        """
        Transform points from source perspective to target perspective.

        Args:
            points: Array of shape (N, 2) containing points to transform.
            flip_x: If True, inverts X coordinates (105 - x) to match broadcast view.

        Returns:
            Transformed points of shape (N, 2).
        """
        if self.m is None or points is None or points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        transformed_points = transformed_points.reshape(-1, 2)

        # Invertir eje X para que coincida con la vista de broadcast
        # En broadcast: equipo de la izquierda ataca hacia la derecha
        # Sin inversión: el radar muestra el campo al revés
        if flip_x:
            transformed_points[:, 0] = 105.0 - transformed_points[:, 0]

        return transformed_points
