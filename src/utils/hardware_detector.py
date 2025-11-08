"""
Detector de hardware disponible (GPU/CPU) para optimizar el procesamiento
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class HardwareDetector:
    """Detecta y configura el hardware disponible para procesamiento"""

    def __init__(self):
        self._device_info = None
        self._detect_hardware()

    def _detect_hardware(self):
        """Detecta el hardware disponible"""
        self._device_info = {
            'has_cuda': False,
            'cuda_available': False,
            'device': 'cpu',
            'device_name': 'CPU',
            'gpu_name': None,
            'gpu_memory': None,
            'onnx_providers': ['CPUExecutionProvider']
        }

        # Detectar PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                self._device_info['has_cuda'] = True
                self._device_info['cuda_available'] = True
                self._device_info['device'] = 'cuda'
                self._device_info['device_name'] = 'GPU (CUDA)'

                # Obtener info de la GPU
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB

                self._device_info['gpu_name'] = gpu_name
                self._device_info['gpu_memory'] = f"{gpu_memory:.1f} GB"

                logger.info(f"✅ GPU detectada: {gpu_name} ({gpu_memory:.1f} GB)")

                # Configurar providers de ONNX Runtime para GPU
                self._device_info['onnx_providers'] = [
                    'CUDAExecutionProvider',
                    'CPUExecutionProvider'  # Fallback
                ]
            else:
                logger.info("ℹ️ CUDA no disponible, usando CPU")

        except ImportError:
            logger.warning("⚠️ PyTorch no instalado, no se puede detectar CUDA")
        except Exception as e:
            logger.warning(f"⚠️ Error al detectar GPU: {e}")

    @property
    def device(self) -> str:
        """Retorna el device a usar ('cpu' o 'cuda')"""
        return self._device_info['device']

    @property
    def device_name(self) -> str:
        """Retorna el nombre legible del device"""
        return self._device_info['device_name']

    @property
    def has_gpu(self) -> bool:
        """Retorna True si hay GPU disponible"""
        return self._device_info['cuda_available']

    @property
    def gpu_info(self) -> Dict[str, Any]:
        """Retorna información de la GPU (si está disponible)"""
        if self.has_gpu:
            return {
                'name': self._device_info['gpu_name'],
                'memory': self._device_info['gpu_memory']
            }
        return None

    @property
    def onnx_providers(self) -> list:
        """Retorna los providers de ONNX Runtime en orden de preferencia"""
        return self._device_info['onnx_providers']

    def get_info_dict(self) -> Dict[str, Any]:
        """Retorna toda la información del hardware"""
        return self._device_info.copy()

    def print_info(self):
        """Imprime información del hardware detectado"""
        print("=" * 60)
        print("🖥️  INFORMACIÓN DE HARDWARE")
        print("=" * 60)
        print(f"Device: {self.device_name}")
        if self.has_gpu:
            print(f"GPU: {self._device_info['gpu_name']}")
            print(f"VRAM: {self._device_info['gpu_memory']}")
            print(f"ONNX Providers: {', '.join(self.onnx_providers)}")
        else:
            print("GPU: No disponible")
        print("=" * 60)


# Instancia global singleton
_hardware_detector = None

def get_hardware_detector() -> HardwareDetector:
    """Retorna la instancia singleton del detector de hardware"""
    global _hardware_detector
    if _hardware_detector is None:
        _hardware_detector = HardwareDetector()
    return _hardware_detector


# Funciones de conveniencia
def get_device() -> str:
    """Retorna 'cpu' o 'cuda' según disponibilidad"""
    return get_hardware_detector().device


def has_gpu() -> bool:
    """Retorna True si hay GPU disponible"""
    return get_hardware_detector().has_gpu


def get_onnx_providers() -> list:
    """Retorna los providers de ONNX Runtime"""
    return get_hardware_detector().onnx_providers
