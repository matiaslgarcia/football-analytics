"""
Formation Detector - Paso 2
Detecta y clasifica formaciones tácticas de equipos de fútbol
basándose en posiciones proyectadas en el campo (homografía).

Sistema adaptativo que funciona con jugadores parcialmente visibles.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
from sklearn.cluster import KMeans


class FormationDetector:
    """
    Detecta formaciones tácticas analizando posiciones de jugadores en el campo.

    Formaciones soportadas:
    - 4-4-2: 4 defensores, 4 mediocampistas, 2 delanteros
    - 4-3-3: 4 defensores, 3 mediocampistas, 3 delanteros
    - 3-5-2: 3 defensores, 5 mediocampistas, 2 delanteros
    - 4-5-1: 4 defensores, 5 mediocampistas, 1 delantero
    - 5-3-2: 5 defensores, 3 mediocampistas, 2 delanteros
    """

    def __init__(self, field_width: float = 105.0, field_length: float = 68.0):
        """
        Args:
            field_width: Ancho del campo en metros (default: 105m FIFA)
            field_length: Largo del campo en metros (default: 68m FIFA)
        """
        self.field_width = field_width
        self.field_length = field_length

        # Umbrales para clasificar posiciones (en % del campo)
        self.defense_threshold = 0.30   # 30% desde el fondo
        self.attack_threshold = 0.70    # 70% desde el fondo

    def detect_formation(
        self,
        positions: np.ndarray,
        team_attacking_direction: str = "right"
    ) -> Dict:
        """
        Detecta la formación táctica de un equipo.

        Args:
            positions: Array (N, 2) con posiciones [x, y] en metros
            team_attacking_direction: "right" o "left" (hacia dónde ataca el equipo)

        Returns:
            Dict con:
                - formation: str (ej: "4-4-2")
                - lines: Dict con jugadores por línea
                - confidence: float (0-1)
                - players_per_line: [defensores, mediocampistas, delanteros]
        """
        if len(positions) < 4:
            return {
                'formation': 'Unknown',
                'lines': {'defense': [], 'midfield': [], 'attack': []},
                'confidence': 0.0,
                'players_per_line': [0, 0, 0],
                'total_players': len(positions)
            }

        # Normalizar posiciones según dirección de ataque
        if team_attacking_direction == "left":
            # Invertir coordenadas X (atacan hacia la izquierda)
            norm_positions = positions.copy()
            norm_positions[:, 0] = self.field_width - norm_positions[:, 0]
        else:
            norm_positions = positions.copy()

        # Clasificar jugadores en líneas basándose en posición X
        defense_line = []
        midfield_line = []
        attack_line = []

        defense_x = self.field_width * self.defense_threshold
        attack_x = self.field_width * self.attack_threshold

        for i, (x, y) in enumerate(norm_positions):
            if x < defense_x:
                defense_line.append(i)
            elif x > attack_x:
                attack_line.append(i)
            else:
                midfield_line.append(i)

        # Contar jugadores por línea
        num_defense = len(defense_line)
        num_midfield = len(midfield_line)
        num_attack = len(attack_line)

        # Determinar formación
        formation_str, confidence = self._classify_formation(
            num_defense, num_midfield, num_attack, len(positions)
        )

        return {
            'formation': formation_str,
            'lines': {
                'defense': defense_line,
                'midfield': midfield_line,
                'attack': attack_line
            },
            'confidence': confidence,
            'players_per_line': [num_defense, num_midfield, num_attack],
            'total_players': len(positions),
            'attacking_direction': team_attacking_direction
        }

    def _classify_formation(
        self,
        defense: int,
        midfield: int,
        attack: int,
        total: int
    ) -> Tuple[str, float]:
        """
        Clasifica la formación basándose en distribución de jugadores.

        Returns:
            (formation_name, confidence)
        """
        # Si tenemos menos de 6 jugadores, es difícil determinar formación
        if total < 6:
            return f"Partial ({defense}-{midfield}-{attack})", 0.3

        # Formaciones conocidas (sin contar portero)
        known_formations = {
            (4, 4, 2): "4-4-2",
            (4, 3, 3): "4-3-3",
            (3, 5, 2): "3-5-2",
            (4, 5, 1): "4-5-1",
            (5, 3, 2): "5-3-2",
            (3, 4, 3): "3-4-3",
            (5, 4, 1): "5-4-1",
            (4, 2, 4): "4-2-4",
        }

        pattern = (defense, midfield, attack)

        # Coincidencia exacta
        if pattern in known_formations:
            # Confianza alta si tenemos 10-11 jugadores
            confidence = 0.9 if total >= 10 else 0.7
            return known_formations[pattern], confidence

        # Buscar formación más cercana
        if total >= 8:
            # Permitir variación de ±1 jugador
            best_match = None
            min_distance = float('inf')

            for known_pattern, name in known_formations.items():
                distance = sum(abs(a - b) for a, b in zip(pattern, known_pattern))
                if distance < min_distance:
                    min_distance = distance
                    best_match = name

            if min_distance <= 2:  # Variación aceptable
                confidence = 0.6 - (min_distance * 0.1)
                return f"{best_match} (approx)", confidence

        # Formación no reconocida
        return f"{defense}-{midfield}-{attack}", 0.4

    def detect_formation_over_time(
        self,
        positions_history: List[np.ndarray],
        team_direction: str = "right",
        window_size: int = 30
    ) -> Dict:
        """
        Detecta formación usando ventana temporal (más robusto).

        Args:
            positions_history: Lista de arrays de posiciones por frame
            team_direction: Dirección de ataque
            window_size: Número de frames a considerar

        Returns:
            Formación más común en la ventana con confianza mejorada
        """
        if not positions_history:
            return {
                'formation': 'Unknown',
                'confidence': 0.0,
                'samples': 0
            }

        # Analizar últimos N frames
        recent_frames = positions_history[-window_size:]
        formations = []

        for positions in recent_frames:
            if len(positions) >= 4:
                result = self.detect_formation(positions, team_direction)
                formations.append(result['formation'])

        if not formations:
            return {
                'formation': 'Unknown',
                'confidence': 0.0,
                'samples': 0
            }

        # Formación más común
        formation_counts = Counter(formations)
        most_common_formation, count = formation_counts.most_common(1)[0]

        # Confianza basada en consistencia temporal
        consistency = count / len(formations)
        confidence = min(consistency * 1.2, 1.0)  # Boost por consistencia

        return {
            'formation': most_common_formation,
            'confidence': confidence,
            'samples': len(formations),
            'distribution': dict(formation_counts)
        }

    def get_line_positions(
        self,
        positions: np.ndarray,
        lines: Dict[str, List[int]]
    ) -> Dict[str, np.ndarray]:
        """
        Obtiene posiciones de jugadores agrupadas por línea.

        Returns:
            Dict con arrays de posiciones por línea
        """
        return {
            line_name: positions[indices] if len(indices) > 0 else np.array([])
            for line_name, indices in lines.items()
        }

    def get_defensive_line_height(self, positions: np.ndarray) -> float:
        """
        Calcula la altura promedio de la línea defensiva.

        Returns:
            Posición X promedio de los defensores (en metros)
        """
        result = self.detect_formation(positions)
        defense_indices = result['lines']['defense']

        if len(defense_indices) == 0:
            return 0.0

        defense_positions = positions[defense_indices]
        return np.mean(defense_positions[:, 0])

    def get_offensive_line_height(self, positions: np.ndarray) -> float:
        """
        Calcula la altura promedio de la línea ofensiva.

        Returns:
            Posición X promedio de los delanteros (en metros)
        """
        result = self.detect_formation(positions)
        attack_indices = result['lines']['attack']

        if len(attack_indices) == 0:
            return self.field_width

        attack_positions = positions[attack_indices]
        return np.mean(attack_positions[:, 0])
