def test_imports():
    # Prueba básica de importación de módulos reorganizados
    from src.models.load_model import load_model  # noqa: F401
    from src.controllers.process_video import process_video  # noqa: F401
    from src.controllers.process_video_segment import process_video_segment  # noqa: F401
    from src.controllers.clip_video_simple import clip_video_simple  # noqa: F401
    from src.controllers.download_game import download_game  # noqa: F401
    from src.utils.config import INPUTS_DIR, OUTPUTS_DIR, VIDEOS_DIR  # noqa: F401
    from src.utils.soccernet_password import resolve_password  # noqa: F401
    from src.utils.ui.sidebar_processing_controls import sidebar_processing_controls  # noqa: F401
    from src.utils.ui.source_selector import source_selector  # noqa: F401
    from src.utils.ui.download_controls import download_controls  # noqa: F401

    assert True