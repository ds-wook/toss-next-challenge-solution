"""
캐싱 유틸리티 함수들
"""

from pathlib import Path
import hashlib


def get_cache_path(cache_dir: str, cache_key: str, extension: str = "pkl") -> str:
    """캐시 파일 경로 생성"""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / f"{cache_key}.{extension}")


def generate_cache_key(*args, **kwargs) -> str:
    """인자들을 기반으로 캐시 키 생성"""
    key_str = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_str.encode()).hexdigest()
