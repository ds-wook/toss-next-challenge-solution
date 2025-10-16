"""
스트리밍 방식 데이터 처리 유틸리티
메모리 사용량을 최소화하여 대용량 데이터 처리
"""

import pickle
from pathlib import Path
from typing import List, Iterator, Callable
import pandas as pd
import numpy as np
from collections import Counter


class StreamingSequenceProcessor:
    """스트리밍 방식 시퀀스 처리기"""

    def __init__(self, temp_dir: str = "/tmp", chunk_size: int = 5000):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.chunk_size = chunk_size
        self.temp_files = []

    def process_sequences_streaming(
        self,
        seq_series: pd.Series,
        process_func: Callable[[List[str]], List[List[int]]] = None,
    ) -> Iterator[List[List[int]]]:
        """
        스트리밍 방식으로 시퀀스 처리

        Args:
            seq_series: 시퀀스 시리즈
            process_func: 각 청크를 처리할 함수

        Yields:
            처리된 시퀀스 청크
        """
        if process_func is None:
            process_func = self._default_sequence_parser

        total_chunks = (len(seq_series) + self.chunk_size - 1) // self.chunk_size
        print(f"스트리밍 처리 시작: {len(seq_series)}개 시퀀스, {total_chunks}개 청크")

        for i in range(0, len(seq_series), self.chunk_size):
            chunk = seq_series.iloc[i : i + self.chunk_size]
            chunk_num = i // self.chunk_size + 1

            print(
                f"  청크 {chunk_num}/{total_chunks} 처리 중... ({len(chunk)}개 시퀀스)"
            )

            # 청크 처리
            processed_chunk = process_func(chunk.tolist())

            # 임시 파일에 저장
            temp_file = self.temp_dir / f"seq_chunk_{chunk_num}.pkl"
            with open(temp_file, "wb") as f:
                pickle.dump(processed_chunk, f)
            self.temp_files.append(temp_file)

            yield processed_chunk

    def _default_sequence_parser(self, seq_strings: List[str]) -> List[List[int]]:
        """기본 시퀀스 파서"""
        sequences = []
        for seq_str in seq_strings:
            if pd.isna(seq_str) or str(seq_str) == "nan":
                sequences.append([])
            else:
                try:
                    seq = [int(x) for x in str(seq_str).split(",") if x.strip()]
                    sequences.append(seq)
                except (ValueError, TypeError):
                    sequences.append([])
        return sequences

    def load_processed_sequences(self) -> List[List[int]]:
        """처리된 시퀀스들을 메모리로 로드"""
        all_sequences = []

        print("임시 파일에서 시퀀스 로드 중...")
        for i, temp_file in enumerate(self.temp_files):
            if temp_file.exists():
                with open(temp_file, "rb") as f:
                    chunk_sequences = pickle.load(f)
                    all_sequences.extend(chunk_sequences)
                print(f"  청크 {i + 1}/{len(self.temp_files)} 로드 완료")

        print(f"총 {len(all_sequences)}개 시퀀스 로드 완료")
        return all_sequences

    def cleanup_temp_files(self):
        """임시 파일들 정리"""
        for temp_file in self.temp_files:
            if temp_file.exists():
                temp_file.unlink()
        self.temp_files.clear()
        print("임시 파일 정리 완료")


class StreamingFeatureProcessor:
    """스트리밍 방식 피처 처리기"""

    def __init__(self, temp_dir: str = "/tmp", chunk_size: int = 2000):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.chunk_size = chunk_size
        self.temp_files = []

    def process_features_streaming(
        self,
        sequences: List[List[int]],
        process_func: Callable[[List[List[int]]], pd.DataFrame] = None,
    ) -> Iterator[pd.DataFrame]:
        """
        스트리밍 방식으로 피처 처리

        Args:
            sequences: 시퀀스 리스트
            process_func: 각 청크를 처리할 함수

        Yields:
            처리된 피처 DataFrame 청크
        """
        if process_func is None:
            process_func = self._default_feature_extractor

        total_chunks = (len(sequences) + self.chunk_size - 1) // self.chunk_size
        print(
            f"스트리밍 피처 처리 시작: {len(sequences)}개 시퀀스, {total_chunks}개 청크"
        )

        for i in range(0, len(sequences), self.chunk_size):
            chunk = sequences[i : i + self.chunk_size]
            chunk_num = i // self.chunk_size + 1

            print(
                f"  피처 청크 {chunk_num}/{total_chunks} 처리 중... ({len(chunk)}개 시퀀스)"
            )

            # 청크 처리
            processed_chunk = process_func(chunk)

            # 임시 파일에 저장
            temp_file = self.temp_dir / f"feature_chunk_{chunk_num}.parquet"
            processed_chunk.to_parquet(temp_file, index=False)
            self.temp_files.append(temp_file)

            yield processed_chunk

    def _default_feature_extractor(self, sequences: List[List[int]]) -> pd.DataFrame:
        """기본 피처 추출기"""
        features = []

        for seq in sequences:
            if not seq:
                features.append(
                    {
                        "seq_len": 0,
                        "seq_unique": 0,
                        "seq_diversity": 0.0,
                        "seq_entropy": 0.0,
                        "seq_compression": 0.0,
                        "recent_items": [0] * 5,
                        "item_freq_avg": 0.0,
                        "item_freq_std": 0.0,
                    }
                )
                continue

            # 기본 통계
            seq_len = len(seq)
            unique_items = len(set(seq))
            diversity = unique_items / seq_len if seq_len > 0 else 0

            # 엔트로피 계산
            item_counts = Counter(seq)
            entropy = 0.0
            for count in item_counts.values():
                if count > 0:
                    p = count / seq_len
                    entropy -= p * np.log2(p)

            # 압축률
            compression = seq_len / unique_items if unique_items > 0 else 0

            # 최근 아이템들
            recent_items = seq[-5:] if len(seq) >= 5 else seq + [0] * (5 - len(seq))

            features.append(
                {
                    "seq_len": seq_len,
                    "seq_unique": unique_items,
                    "seq_diversity": diversity,
                    "seq_entropy": entropy,
                    "seq_compression": compression,
                    "recent_items": recent_items,
                    "item_freq_avg": 0.0,  # 어휘 사전이 없으므로 0
                    "item_freq_std": 0.0,
                }
            )

        return pd.DataFrame(features)

    def load_processed_features(self) -> pd.DataFrame:
        """처리된 피처들을 메모리로 로드"""
        all_features = []

        print("임시 파일에서 피처 로드 중...")
        for i, temp_file in enumerate(self.temp_files):
            if temp_file.exists():
                chunk_features = pd.read_parquet(temp_file)
                all_features.append(chunk_features)
                print(f"  피처 청크 {i + 1}/{len(self.temp_files)} 로드 완료")

        if all_features:
            result_df = pd.concat(all_features, ignore_index=True)
            print(f"총 {len(result_df)}개 피처 로드 완료")
            return result_df
        else:
            return pd.DataFrame()

    def cleanup_temp_files(self):
        """임시 파일들 정리"""
        for temp_file in self.temp_files:
            if temp_file.exists():
                temp_file.unlink()
        self.temp_files.clear()
        print("임시 파일 정리 완료")


def process_large_dataset_streaming(
    df: pd.DataFrame,
    seq_col: str = "seq",
    chunk_size: int = 5000,
    temp_dir: str = "/tmp",
) -> tuple:
    """
    대용량 데이터셋을 스트리밍 방식으로 처리

    Returns:
        (sequences, features_df)
    """
    print("=== 스트리밍 방식 대용량 데이터 처리 ===")

    # 시퀀스 처리
    seq_processor = StreamingSequenceProcessor(temp_dir, chunk_size)
    sequences = []

    for chunk in seq_processor.process_sequences_streaming(df[seq_col]):
        sequences.extend(chunk)

    print(f"시퀀스 처리 완료: {len(sequences)}개")

    # 피처 처리
    feature_processor = StreamingFeatureProcessor(temp_dir, chunk_size // 2)
    features_df = feature_processor.load_processed_features()

    # 임시 파일 정리
    seq_processor.cleanup_temp_files()
    feature_processor.cleanup_temp_files()

    return sequences, features_df
