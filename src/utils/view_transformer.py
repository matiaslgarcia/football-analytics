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

    @classmethod
    def from_matrix(cls, H: np.ndarray):
        obj = cls.__new__(cls)
        obj.m = H.astype(np.float32)
        return obj

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points from source perspective to target perspective.
        
        Args:
            points: Array of shape (N, 2) containing points to transform.
            
        Returns:
            Transformed points of shape (N, 2).
        """
        if self.m is None or points is None or points.size == 0:
            return points
        
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)
