"""
Utils module for CTR prediction
"""

from .config import DataConfig, ModelConfig, TrainingConfig
from .device import get_device, print_device_info

__all__ = [
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "get_device",
    "print_device_info",
]
