import os
from typing import Optional
from .config import SOCCERNET_PASSWORD_KEYS


def resolve_password(explicit_password: Optional[str]) -> str:
    """Obtiene el password para SoccerNet desde argumento o variables de entorno."""
    if explicit_password:
        return explicit_password
    for key in SOCCERNET_PASSWORD_KEYS:
        val = os.getenv(key)
        if val:
            return val
    raise ValueError("No se encontró password NDA de SoccerNet. Configure variable de entorno.")