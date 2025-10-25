from pathlib import Path

# Directorios base del proyecto
INPUTS_DIR = Path("inputs")
OUTPUTS_DIR = Path("outputs")
VIDEOS_DIR = Path("videos")

# Garantizar que existan
for d in (INPUTS_DIR, OUTPUTS_DIR, VIDEOS_DIR):
    d.mkdir(exist_ok=True)

# Claves posibles para password NDA de SoccerNet
SOCCERNET_PASSWORD_KEYS = ("SOCCERNET_PASSWORD", "SOCCERNET_PW")