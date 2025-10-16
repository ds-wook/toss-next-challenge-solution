"""
Device utilities
"""

import torch


def get_device(device: str = None) -> str:
    """
    Get available device

    Args:
        device: Preferred device ('auto', 'cuda', or 'cpu')

    Returns:
        Available device string
    """
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    return device


def print_device_info(device: str):
    """
    Print device information

    Args:
        device: Device string
    """
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

        # Memory info
        if torch.cuda.is_available():
            print(
                f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
            )
            print(
                f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB"
            )
