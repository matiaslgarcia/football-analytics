"""
Tactical Metrics Calculator - Paso 3
Calcula métricas de comportamiento colectivo para análisis táctico de fútbol.

Métricas implementadas:
- Compactación del equipo (área ocupada)
- Altura de presión (posición promedio)
- Amplitud ofensiva (dispersión horizontal)
- Centroide del equipo
- Stretch Index (elongación)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial import ConvexHull
from collections import deque


class TacticalMetricsCalculator:
    """
    Calcula métricas tácticas para equipos de fútbol basándose en posiciones.
    """

    def __init__(self, field_width: float = 105.0, field_length: float = 68.0):
        """
        Args:
            field_width: Ancho del campo en metros (105m FIFA standard)
            field_length: Largo del campo en metros (68m FIFA standard)
        """
        self.field_width = field_width
        self.field_length = field_length

    def calculate_all_metrics(self, positions: np.ndarray) -> Dict:
        """
        Calcula todas las métricas para un conjunto de posiciones.

        Args:
            positions: Array (N, 2) con posiciones [x, y] en metros

        Returns:
            Dict con todas las métricas calculadas
        """
        if len(positions) < 3:
            return self._empty_metrics()

        return {
            'compactness': self.calculate_compactness(positions),
            'pressure_height': self.calculate_pressure_height(positions),
            'offensive_width': self.calculate_offensive_width(positions),
            'centroid': self.calculate_centroid(positions),
            'stretch_index': self.calculate_stretch_index(positions),
            'defensive_depth': self.calculate_defensive_depth(positions),
            'num_players': len(positions)
        }

    def calculate_compactness(self, positions: np.ndarray) -> float:
        """
        Calcula la compactación del equipo (área ocupada).

        Compactación = Área del polígono convexo que contiene a los jugadores

        Menor área = Mayor compactación (equipo más junto)
        Mayor área = Menor compactación (equipo más disperso)

        Returns:
            Área en metros cuadrados
        """
        if len(positions) < 3:
            return 0.0

        try:
            hull = ConvexHull(positions)
            area = hull.volume  # En 2D, 'volume' es el área
            return float(area)
        except Exception:
            # Si los puntos son colineales, calcular área del rectángulo
            x_range = positions[:, 0].max() - positions[:, 0].min()
            y_range = positions[:, 1].max() - positions[:, 1].min()
            return x_range * y_range

    def calculate_pressure_height(self, positions: np.ndarray) -> float:
        """
        Calcula la altura de presión del equipo.

        Presión = Posición X promedio de los jugadores

        Valores altos = Presión alta (cerca del arco rival)
        Valores bajos = Presión baja (cerca del arco propio)

        Returns:
            Posición X promedio en metros (0-105)
        """
        if len(positions) == 0:
            return 0.0

        return float(np.mean(positions[:, 0]))

    def calculate_offensive_width(self, positions: np.ndarray) -> float:
        """
        Calcula la amplitud ofensiva (dispersión horizontal).

        Amplitud = Rango en el eje Y (ancho del campo ocupado)

        Mayor valor = Mayor amplitud (equipo estirado horizontalmente)
        Menor valor = Menor amplitud (equipo concentrado al centro)

        Returns:
            Amplitud en metros (0-68)
        """
        if len(positions) == 0:
            return 0.0

        y_range = positions[:, 1].max() - positions[:, 1].min()
        return float(y_range)

    def calculate_centroid(self, positions: np.ndarray) -> Tuple[float, float]:
        """
        Calcula el centroide (centro geométrico) del equipo.

        Returns:
            Tupla (x, y) con el centroide en metros
        """
        if len(positions) == 0:
            return (0.0, 0.0)

        centroid_x = float(np.mean(positions[:, 0]))
        centroid_y = float(np.mean(positions[:, 1]))
        return (centroid_x, centroid_y)

    def calculate_stretch_index(self, positions: np.ndarray) -> float:
        """
        Calcula el índice de elongación del equipo.

        Stretch Index = Desviación estándar de posiciones X / Desviación estándar de posiciones Y

        > 1.0 = Equipo más estirado verticalmente (en profundidad)
        < 1.0 = Equipo más estirado horizontalmente (en amplitud)
        ~ 1.0 = Equipo equilibrado

        Returns:
            Ratio de elongación
        """
        if len(positions) < 2:
            return 1.0

        std_x = np.std(positions[:, 0])
        std_y = np.std(positions[:, 1])

        if std_y < 0.1:  # Evitar división por cero
            return 10.0 if std_x > 0.1 else 1.0

        return float(std_x / std_y)

    def calculate_defensive_depth(self, positions: np.ndarray) -> float:
        """
        Calcula la profundidad defensiva (distancia entre jugador más adelantado y más retrasado).

        Returns:
            Profundidad en metros
        """
        if len(positions) == 0:
            return 0.0

        x_range = positions[:, 0].max() - positions[:, 0].min()
        return float(x_range)

    def calculate_defensive_block_compactness(self, positions: np.ndarray) -> float:
        """
        Calcula la compactación del bloque defensivo (jugadores más retrasados).

        Returns:
            Área del 50% de jugadores más retrasados
        """
        if len(positions) < 4:
            return self.calculate_compactness(positions)

        # Tomar mitad de jugadores más retrasados
        sorted_by_x = positions[np.argsort(positions[:, 0])]
        defensive_half = sorted_by_x[:len(positions)//2]

        return self.calculate_compactness(defensive_half)

    def _empty_metrics(self) -> Dict:
        """Retorna métricas vacías cuando no hay suficientes jugadores."""
        return {
            'compactness': 0.0,
            'pressure_height': 0.0,
            'offensive_width': 0.0,
            'centroid': (0.0, 0.0),
            'stretch_index': 1.0,
            'defensive_depth': 0.0,
            'num_players': 0
        }


class TacticalMetricsTracker:
    """
    Rastrea métricas tácticas a lo largo del tiempo y calcula tendencias.
    """

    def __init__(self, history_size: int = 300):
        """
        Args:
            history_size: Número de frames a mantener en historia (default: 300 = ~10 seg a 30fps)
        """
        self.history_size = history_size
        self.metrics_history = {
            'compactness': deque(maxlen=history_size),
            'pressure_height': deque(maxlen=history_size),
            'offensive_width': deque(maxlen=history_size),
            'centroid_x': deque(maxlen=history_size),
            'centroid_y': deque(maxlen=history_size),
            'stretch_index': deque(maxlen=history_size),
            'defensive_depth': deque(maxlen=history_size),
            'frame_number': deque(maxlen=history_size)
        }

    def update(self, metrics: Dict, frame_number: int):
        """
        Actualiza el historial con nuevas métricas.

        Args:
            metrics: Dict con métricas del frame actual
            frame_number: Número del frame
        """
        self.metrics_history['compactness'].append(metrics['compactness'])
        self.metrics_history['pressure_height'].append(metrics['pressure_height'])
        self.metrics_history['offensive_width'].append(metrics['offensive_width'])
        self.metrics_history['centroid_x'].append(metrics['centroid'][0])
        self.metrics_history['centroid_y'].append(metrics['centroid'][1])
        self.metrics_history['stretch_index'].append(metrics['stretch_index'])
        self.metrics_history['defensive_depth'].append(metrics['defensive_depth'])
        self.metrics_history['frame_number'].append(frame_number)

    def get_statistics(self) -> Dict:
        """
        Calcula estadísticas sobre las métricas históricas.

        Returns:
            Dict con media, std, min, max para cada métrica
        """
        stats = {}

        for metric_name, values in self.metrics_history.items():
            if metric_name == 'frame_number':
                continue

            if len(values) == 0:
                stats[metric_name] = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
                continue

            values_array = np.array(list(values))
            stats[metric_name] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'current': float(values_array[-1]) if len(values_array) > 0 else 0.0
            }

        return stats

    def get_trend(self, metric_name: str, window: int = 30) -> str:
        """
        Determina la tendencia de una métrica (creciente, decreciente, estable).

        Args:
            metric_name: Nombre de la métrica
            window: Ventana para calcular tendencia

        Returns:
            "increasing", "decreasing", "stable"
        """
        if metric_name not in self.metrics_history:
            return "unknown"

        values = list(self.metrics_history[metric_name])

        if len(values) < window:
            return "insufficient_data"

        recent_values = values[-window:]
        first_half = np.mean(recent_values[:window//2])
        second_half = np.mean(recent_values[window//2:])

        change_pct = abs((second_half - first_half) / first_half) * 100 if first_half != 0 else 0

        if change_pct < 5:  # Menos del 5% de cambio
            return "stable"
        elif second_half > first_half:
            return "increasing"
        else:
            return "decreasing"

    def export_to_dict(self) -> Dict:
        """
        Exporta todo el historial a un diccionario (para guardar en JSON/CSV).

        Returns:
            Dict con arrays de métricas por frame
        """
        return {
            metric_name: list(values)
            for metric_name, values in self.metrics_history.items()
        }

    def export_to_arrays(self) -> Dict[str, np.ndarray]:
        """
        Exporta historial como arrays de NumPy (para gráficos).

        Returns:
            Dict con arrays de NumPy por métrica
        """
        return {
            metric_name: np.array(list(values))
            for metric_name, values in self.metrics_history.items()
        }
