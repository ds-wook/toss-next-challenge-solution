"""
BST 모델을 위한 피처 엔지니어링
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter
import pickle
from pathlib import Path
import time
import os

from utils.caching import (
    get_cache_path,
    generate_cache_key,
)
from utils.memory_utils import (
    log_memory_usage,
    force_garbage_collection,
    optimize_dataframe_memory,
    safe_divide_chunks,
)


class BSTFeatureProcessor:
    """BST 모델을 위한 피처 처리 클래스"""

    def __init__(
        self, max_seq_len: int = 20, min_item_freq: int = 5, cache_dir: str = "cache"
    ):
        self.max_seq_len = max_seq_len
        self.min_item_freq = min_item_freq
        self.cache_dir = cache_dir
        self.item_vocab = {}
        self.item_freq = {}
        self.feature_stats = {}

    def build_item_vocabulary(
        self, sequences: List[List[int]], force_refresh: bool = False
    ) -> Dict[int, int]:
        """아이템 어휘 사전 구축 (캐싱 지원)"""
        # 캐시 키 생성
        cache_key = generate_cache_key(
            "item_vocab",
            self.min_item_freq,
            len(sequences),
            sum(len(seq) for seq in sequences[:100]),  # 샘플링으로 키 생성
        )

        cache_path = get_cache_path(self.cache_dir, cache_key, "pkl")

        # 캐시에서 로드 시도
        if not force_refresh and Path(cache_path).exists():
            print(f"캐시에서 어휘 사전 로드: {cache_path}")
            with open(cache_path, "rb") as f:
                vocab_data = pickle.load(f)
                self.item_vocab = vocab_data["item_vocab"]
                self.item_freq = vocab_data["item_freq"]
                print(
                    f"어휘 크기: {len(self.item_vocab)} (원본 아이템: {len(self.item_freq)})"
                )
                return self.item_vocab

        print("아이템 어휘 사전 구축 중...")
        start_time = time.time()

        # 모든 아이템 수집
        all_items = []
        for seq in sequences:
            all_items.extend(seq)

        # 아이템 빈도 계산
        item_counts = Counter(all_items)
        self.item_freq = dict(item_counts)

        # 최소 빈도 이상인 아이템만 선택
        filtered_items = {
            item: count
            for item, count in item_counts.items()
            if count >= self.min_item_freq
        }

        # 어휘 사전 생성 (0은 패딩용)
        vocab = {0: 0}  # 패딩 토큰
        for i, item in enumerate(sorted(filtered_items.keys()), 1):
            vocab[item] = i

        self.item_vocab = vocab

        # 캐시에 저장
        vocab_data = {
            "item_vocab": self.item_vocab,
            "item_freq": self.item_freq,
            "min_item_freq": self.min_item_freq,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(vocab_data, f)

        build_time = time.time() - start_time
        print(f"어휘 크기: {len(vocab)} (원본 아이템: {len(item_counts)})")
        print(f"최소 빈도: {self.min_item_freq}")
        print(f"어휘 구축 시간: {build_time:.2f}초")
        print(f"캐시 저장 완료: {cache_path}")

        return vocab

    def _parse_sequences_memory_efficient(
        self, seq_series: pd.Series
    ) -> List[List[int]]:
        """메모리 효율적인 시퀀스 파싱"""
        print(f"시퀀스 파싱 중... (총 {len(seq_series)}개)")

        # 청크 크기 계산 (더 작게 설정)
        chunk_size = safe_divide_chunks(len(seq_series), max_chunk_size=10000)
        print(f"청크 크기: {chunk_size}")

        # 중간 저장을 위한 임시 파일들
        temp_files = []

        for i in range(0, len(seq_series), chunk_size):
            chunk = seq_series.iloc[i : i + chunk_size]
            chunk_sequences = []

            for seq_str in chunk:
                if pd.isna(seq_str) or str(seq_str) == "nan":
                    chunk_sequences.append([])
                else:
                    try:
                        seq = [int(x) for x in str(seq_str).split(",") if x.strip()]
                        chunk_sequences.append(seq)
                    except (ValueError, TypeError):
                        chunk_sequences.append([])

            # 청크를 임시 파일에 저장
            temp_file = f"/tmp/seq_chunk_{i // chunk_size}.pkl"
            with open(temp_file, "wb") as f:
                pickle.dump(chunk_sequences, f)
            temp_files.append(temp_file)

            # 메모리 정리
            del chunk_sequences
            force_garbage_collection()

            if (i // chunk_size + 1) % 10 == 0:
                print(
                    f"  진행률: {i + len(chunk)}/{len(seq_series)} ({(i + len(chunk)) / len(seq_series) * 100:.1f}%)"
                )
                log_memory_usage(None, f"청크 {i // chunk_size + 1}")

        # 임시 파일들을 그대로 유지하고 파일 경로만 반환
        print(f"시퀀스 파싱 완료: {len(temp_files)}개 청크 파일 생성")
        return temp_files

    def process_sequences(self, sequences_or_files) -> List[List[int]]:
        """시퀀스 처리 (어휘 사전 적용) - 임시 파일 지원"""
        print("시퀀스 처리 중...")

        # 임시 파일 리스트인지 확인
        if (
            isinstance(sequences_or_files, list)
            and len(sequences_or_files) > 0
            and isinstance(sequences_or_files[0], str)
        ):
            return self._process_sequences_from_files(sequences_or_files)
        else:
            return self._process_sequences_in_memory(sequences_or_files)

    def _process_sequences_in_memory(
        self, sequences: List[List[int]]
    ) -> List[List[int]]:
        """메모리에서 시퀀스 처리"""
        processed_sequences = []
        unknown_count = 0
        total_items = 0

        for seq in sequences:
            processed_seq = []
            for item in seq:
                total_items += 1
                if item in self.item_vocab:
                    processed_seq.append(self.item_vocab[item])
                else:
                    processed_seq.append(0)  # 패딩 토큰으로 대체
                    unknown_count += 1

            processed_sequences.append(processed_seq)

        print(f"알 수 없는 아이템 비율: {unknown_count / total_items * 100:.2f}%")

        return processed_sequences

    def _process_sequences_from_files(self, temp_files: List[str]) -> List[List[int]]:
        """임시 파일에서 시퀀스 처리"""
        processed_sequences = []
        unknown_count = 0
        total_items = 0

        print(f"임시 파일에서 시퀀스 처리 중... ({len(temp_files)}개 파일)")

        for i, temp_file in enumerate(temp_files):
            if i % 10 == 0:
                print(f"  파일 {i + 1}/{len(temp_files)} 처리 중...")

            with open(temp_file, "rb") as f:
                chunk_sequences = pickle.load(f)

            for seq in chunk_sequences:
                processed_seq = []
                for item in seq:
                    total_items += 1
                    if item in self.item_vocab:
                        processed_seq.append(self.item_vocab[item])
                    else:
                        processed_seq.append(0)  # 패딩 토큰으로 대체
                        unknown_count += 1

                processed_sequences.append(processed_seq)

            # 메모리 정리
            del chunk_sequences
            force_garbage_collection()

            # 임시 파일 삭제
            os.remove(temp_file)

        print(f"알 수 없는 아이템 비율: {unknown_count / total_items * 100:.2f}%")
        print(f"시퀀스 처리 완료: {len(processed_sequences)}개")

        return processed_sequences

    def extract_sequence_features(
        self, sequences: List[List[int]], force_refresh: bool = False
    ) -> pd.DataFrame:
        """시퀀스 기반 피처 추출 (캐싱 지원)"""
        # 캐시 키 생성
        cache_key = generate_cache_key(
            "seq_features",
            len(sequences),
            self.max_seq_len,
            sum(len(seq) for seq in sequences[:100]),  # 샘플링으로 키 생성
        )

        cache_path = get_cache_path(self.cache_dir, cache_key, "parquet")

        # 캐시에서 로드 시도
        if not force_refresh and Path(cache_path).exists():
            print(f"캐시에서 시퀀스 피처 로드: {cache_path}")
            return pd.read_parquet(cache_path)

        print("시퀀스 피처 추출 중...")
        print(f"입력 시퀀스 수: {len(sequences)}")
        start_time = time.time()

        features = []

        for i, seq in enumerate(sequences):
            if not seq:
                features.append(
                    {
                        "seq_len": 0,
                        "seq_unique": 0,
                        "seq_diversity": 0.0,
                        "seq_entropy": 0.0,
                        "seq_compression": 0.0,
                        "recent_items": [0] * 5,  # 최근 5개 아이템
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

            # 아이템 빈도 통계
            item_freqs = [self.item_freq.get(item, 0) for item in seq]
            freq_avg = np.mean(item_freqs) if item_freqs else 0
            freq_std = np.std(item_freqs) if len(item_freqs) > 1 else 0

            features.append(
                {
                    "seq_len": seq_len,
                    "seq_unique": unique_items,
                    "seq_diversity": diversity,
                    "seq_entropy": entropy,
                    "seq_compression": compression,
                    "recent_items": recent_items,
                    "item_freq_avg": freq_avg,
                    "item_freq_std": freq_std,
                }
            )

            # 디버깅: 처음 몇 개만 출력
            if i < 5:
                print(f"  시퀀스 {i}: 길이={seq_len}, 고유={unique_items}")

        result_df = pd.DataFrame(features)

        # 캐시에 저장
        result_df.to_parquet(cache_path, index=False)

        extract_time = time.time() - start_time
        print(f"시퀀스 피처 결과 크기: {result_df.shape}")
        print(f"시퀀스 피처 추출 시간: {extract_time:.2f}초")
        print(f"캐시 저장 완료: {cache_path}")

        return result_df

    def extract_sequence_features_memory_efficient(
        self, sequences_or_files, force_refresh: bool = False
    ) -> pd.DataFrame:
        """메모리 효율적인 시퀀스 피처 추출 - 임시 파일 지원"""
        # 임시 파일 리스트인지 확인
        if (
            isinstance(sequences_or_files, list)
            and len(sequences_or_files) > 0
            and isinstance(sequences_or_files[0], str)
        ):
            return self._extract_features_from_files(sequences_or_files, force_refresh)
        else:
            return self._extract_features_in_memory(sequences_or_files, force_refresh)

    def _extract_features_in_memory(
        self, sequences: List[List[int]], force_refresh: bool = False
    ) -> pd.DataFrame:
        """메모리에서 피처 추출"""
        # 캐시 키 생성
        cache_key = generate_cache_key(
            "seq_features_mem_eff",
            len(sequences),
            self.max_seq_len,
            sum(len(seq) for seq in sequences[:100]),  # 샘플링으로 키 생성
        )

        cache_path = get_cache_path(self.cache_dir, cache_key, "parquet")

        # 캐시에서 로드 시도
        if not force_refresh and Path(cache_path).exists():
            print(f"캐시에서 시퀀스 피처 로드: {cache_path}")
            return pd.read_parquet(cache_path)

        print("메모리에서 시퀀스 피처 추출 중...")
        print(f"입력 시퀀스 수: {len(sequences)}")
        start_time = time.time()

        # 청크 크기 계산 (더 작게 설정)
        chunk_size = safe_divide_chunks(len(sequences), max_chunk_size=5000)
        print(f"청크 크기: {chunk_size}")

        all_features = []

        for i in range(0, len(sequences), chunk_size):
            chunk_sequences = sequences[i : i + chunk_size]
            chunk_features = self._extract_chunk_features(chunk_sequences)
            all_features.extend(chunk_features)

            # 메모리 정리
            del chunk_features
            force_garbage_collection()

            if (i // chunk_size + 1) % 10 == 0:
                print(
                    f"  진행률: {i + len(chunk_sequences)}/{len(sequences)} ({(i + len(chunk_sequences)) / len(sequences) * 100:.1f}%)"
                )
                log_memory_usage(None, f"피처 청크 {i // chunk_size + 1}")

        result_df = pd.DataFrame(all_features)

        # 캐시에 저장
        result_df.to_parquet(cache_path, index=False)

        extract_time = time.time() - start_time
        print(f"시퀀스 피처 결과 크기: {result_df.shape}")
        print(f"시퀀스 피처 추출 시간: {extract_time:.2f}초")
        print(f"캐시 저장 완료: {cache_path}")

        return result_df

    def _extract_features_from_files(
        self, temp_files: List[str], force_refresh: bool = False
    ) -> pd.DataFrame:
        """임시 파일에서 피처 추출"""
        print("임시 파일에서 시퀀스 피처 추출 중...")
        print(f"입력 파일 수: {len(temp_files)}")
        start_time = time.time()

        all_features = []

        for i, temp_file in enumerate(temp_files):
            if i % 10 == 0:
                print(f"  파일 {i + 1}/{len(temp_files)} 처리 중...")

            with open(temp_file, "rb") as f:
                chunk_sequences = pickle.load(f)

            chunk_features = self._extract_chunk_features(chunk_sequences)
            all_features.extend(chunk_features)

            # 메모리 정리
            del chunk_sequences, chunk_features
            force_garbage_collection()

            if i % 50 == 0:
                log_memory_usage(None, f"피처 파일 {i + 1}")

        result_df = pd.DataFrame(all_features)

        extract_time = time.time() - start_time
        print(f"시퀀스 피처 결과 크기: {result_df.shape}")
        print(f"시퀀스 피처 추출 시간: {extract_time:.2f}초")

        return result_df

    def _extract_chunk_features(self, chunk_sequences: List[List[int]]) -> List[dict]:
        """청크 시퀀스에서 피처 추출"""
        chunk_features = []

        for seq in chunk_sequences:
            if not seq:
                chunk_features.append(
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

            # 아이템 빈도 통계
            item_freqs = [self.item_freq.get(item, 0) for item in seq]
            freq_avg = np.mean(item_freqs) if item_freqs else 0
            freq_std = np.std(item_freqs) if len(item_freqs) > 1 else 0

            chunk_features.append(
                {
                    "seq_len": seq_len,
                    "seq_unique": unique_items,
                    "seq_diversity": diversity,
                    "seq_entropy": entropy,
                    "seq_compression": compression,
                    "recent_items": recent_items,
                    "item_freq_avg": freq_avg,
                    "item_freq_std": freq_std,
                }
            )

        return chunk_features

    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """시간적 피처 추출"""
        print("시간적 피처 추출 중...")

        temporal_features = df.copy()

        # 시간대별 피처 (데이터 타입 확인)
        if "hour" in df.columns:
            hour_values = pd.to_numeric(df["hour"], errors="coerce").fillna(0)
            temporal_features["hour_sin"] = np.sin(2 * np.pi * hour_values / 24)
            temporal_features["hour_cos"] = np.cos(2 * np.pi * hour_values / 24)

        # 요일별 피처 (데이터 타입 확인)
        if "day_of_week" in df.columns:
            day_values = pd.to_numeric(df["day_of_week"], errors="coerce").fillna(0)
            temporal_features["day_sin"] = np.sin(2 * np.pi * day_values / 7)
            temporal_features["day_cos"] = np.cos(2 * np.pi * day_values / 7)

        # 주말 여부
        if "day_of_week" in df.columns:
            day_values = pd.to_numeric(df["day_of_week"], errors="coerce").fillna(0)
            temporal_features["is_weekend"] = (day_values >= 5).astype(int)

        # 시간대 그룹
        if "hour" in df.columns:
            hour_values = pd.to_numeric(df["hour"], errors="coerce").fillna(0)
            temporal_features["hour_group"] = pd.cut(
                hour_values,
                bins=[0, 6, 12, 18, 24],
                labels=["night", "morning", "afternoon", "evening"],
                include_lowest=True,
            )

        # 원-핫 인코딩
        if "hour_group" in temporal_features.columns:
            hour_group_dummies = pd.get_dummies(
                temporal_features["hour_group"], prefix="hour_group"
            )
            temporal_features = pd.concat(
                [temporal_features, hour_group_dummies], axis=1
            )

        return temporal_features

    def extract_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """사용자 피처 추출"""
        print("사용자 피처 추출 중...")

        user_features = df.copy()

        # 성별 원-핫 인코딩
        gender_dummies = pd.get_dummies(df["gender"], prefix="gender")
        user_features = pd.concat([user_features, gender_dummies], axis=1)

        # 연령대 원-핫 인코딩
        age_dummies = pd.get_dummies(df["age_group"], prefix="age")
        user_features = pd.concat([user_features, age_dummies], axis=1)

        return user_features

    def extract_ad_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """광고 피처 추출"""
        print("광고 피처 추출 중...")

        ad_features = df.copy()

        # 광고 계층 피처들
        ad_hierarchy_cols = ["inventory_id", "l_feat_11", "l_feat_12", "l_feat_14"]

        for col in ad_hierarchy_cols:
            if col in df.columns:
                # 고유값 수가 적으면 원-핫 인코딩
                unique_count = df[col].nunique()
                if unique_count <= 100:  # 임계값 조정 가능
                    dummies = pd.get_dummies(df[col], prefix=col)
                    ad_features = pd.concat([ad_features, dummies], axis=1)
                else:
                    # 고유값이 많으면 라벨 인코딩
                    from sklearn.preprocessing import LabelEncoder

                    le = LabelEncoder()
                    ad_features[f"{col}_encoded"] = le.fit_transform(
                        df[col].astype(str)
                    )

        return ad_features

    def extract_content_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """콘텐츠 피처 추출"""
        print("콘텐츠 피처 추출 중...")

        content_features = df.copy()

        # 정보영역 피처들
        info_areas = ["feat_a_", "feat_b_", "feat_c_", "feat_d_", "feat_e_"]

        for area in info_areas:
            area_cols = [col for col in df.columns if col.startswith(area)]
            if area_cols:
                # 영역별 통계 피처
                content_features[f"{area}sum"] = df[area_cols].sum(axis=1)
                content_features[f"{area}mean"] = df[area_cols].mean(axis=1)
                content_features[f"{area}std"] = df[area_cols].std(axis=1)
                content_features[f"{area}max"] = df[area_cols].max(axis=1)
                content_features[f"{area}min"] = df[area_cols].min(axis=1)

        # 히스토리 피처들
        history_areas = ["history_a_", "history_b_"]

        for area in history_areas:
            area_cols = [col for col in df.columns if col.startswith(area)]
            if area_cols:
                # 영역별 통계 피처
                content_features[f"{area}sum"] = df[area_cols].sum(axis=1)
                content_features[f"{area}mean"] = df[area_cols].mean(axis=1)
                content_features[f"{area}std"] = df[area_cols].std(axis=1)

        return content_features

    def _extract_features_memory_efficient(self, df: pd.DataFrame) -> pd.DataFrame:
        """메모리 효율적인 피처 추출"""
        print("메모리 효율적인 피처 추출 중...")
        print(f"원본 데이터 크기: {df.shape}")

        # DataFrame 메모리 최적화
        df_processed = optimize_dataframe_memory(df.copy())
        log_memory_usage(None, "DataFrame 메모리 최적화 후")

        # 시간적 피처
        if "hour" in df.columns and "day_of_week" in df.columns:
            df_processed = self.extract_temporal_features(df_processed)
            print(f"시간적 피처 처리 후: {df_processed.shape}")
            log_memory_usage(None, "시간적 피처 처리 후")

        # 사용자 피처
        if "gender" in df.columns and "age_group" in df.columns:
            df_processed = self.extract_user_features(df_processed)
            print(f"사용자 피처 처리 후: {df_processed.shape}")
            log_memory_usage(None, "사용자 피처 처리 후")

        # 광고 피처
        ad_hierarchy_cols = ["inventory_id", "l_feat_11", "l_feat_12", "l_feat_14"]
        if any(col in df.columns for col in ad_hierarchy_cols):
            df_processed = self.extract_ad_features(df_processed)
            print(f"광고 피처 처리 후: {df_processed.shape}")
            log_memory_usage(None, "광고 피처 처리 후")

        # 콘텐츠 피처
        df_processed = self.extract_content_features(df_processed)
        print(f"콘텐츠 피처 처리 후: {df_processed.shape}")
        log_memory_usage(None, "콘텐츠 피처 처리 후")

        return df_processed

    def process_data(
        self,
        df: pd.DataFrame,
        seq_col: str = "seq",
        target_col: Optional[str] = None,
        is_training: bool = True,
        force_refresh: bool = False,
    ) -> Tuple[List[List[int]], np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """
        전체 데이터 처리 파이프라인 (캐싱 지원)

        Returns:
            sequences: 처리된 시퀀스 리스트
            other_features: 기타 피처 배열
            targets: 타겟 배열 (훈련 시에만)
            feature_info: 피처 정보 딕셔너리
        """
        # 캐시 키 생성
        cache_key = generate_cache_key(
            "process_data",
            is_training,
            len(df),
            df.shape[1],
            self.max_seq_len,
            self.min_item_freq,
            seq_col,
            target_col,
        )

        # 전체 결과 캐시 확인
        full_cache_path = get_cache_path(self.cache_dir, f"{cache_key}_full", "pkl")
        if not force_refresh and Path(full_cache_path).exists():
            print(f"캐시에서 전체 처리 결과 로드: {full_cache_path}")
            with open(full_cache_path, "rb") as f:
                cached_result = pickle.load(f)
                # 어휘 사전도 복원
                self.item_vocab = cached_result["feature_info"]["item_vocab"]
                self.item_freq = cached_result["feature_info"]["item_freq"]
                return (
                    cached_result["sequences"],
                    cached_result["other_features"],
                    cached_result["targets"],
                    cached_result["feature_info"],
                )

        print("=== BST 데이터 처리 시작 ===")
        start_time = time.time()
        log_memory_usage(None, "처리 시작")

        # 시퀀스 파싱 (메모리 효율적)
        print("시퀀스 파싱 중...")
        sequences = self._parse_sequences_memory_efficient(df[seq_col])
        log_memory_usage(None, "시퀀스 파싱 완료")

        # 어휘 사전 구축 (훈련 시에만)
        if is_training:
            self.build_item_vocabulary(sequences, force_refresh=force_refresh)

        # 시퀀스 처리
        processed_sequences = self.process_sequences(sequences)

        # 시퀀스 피처 추출 (메모리 효율적)
        seq_features = self.extract_sequence_features_memory_efficient(
            processed_sequences, force_refresh=force_refresh
        )

        # 기타 피처 추출 (메모리 효율적)
        df_processed = self._extract_features_memory_efficient(df)
        log_memory_usage(None, "기타 피처 추출 완료")

        # 시퀀스 피처와 결합
        print(f"시퀀스 피처 크기: {seq_features.shape}")

        # 인덱스 리셋하여 concat 문제 해결
        df_processed = df_processed.reset_index(drop=True)
        seq_features = seq_features.reset_index(drop=True)

        df_final = pd.concat([df_processed, seq_features], axis=1)
        print(f"최종 데이터 크기: {df_final.shape}")

        # 기타 피처 선택 (시퀀스 관련 피처 제외)
        exclude_cols = [seq_col, target_col] + [
            col for col in df_final.columns if col.startswith("recent_items")
        ]
        other_feature_cols = [
            col for col in df_final.columns if col not in exclude_cols
        ]

        # 수치형 피처만 선택
        numeric_cols = (
            df_final[other_feature_cols].select_dtypes(include=[np.number]).columns
        )
        other_features = df_final[numeric_cols].fillna(0).values

        # 타겟 추출
        targets = None
        if target_col and target_col in df.columns:
            targets = df[target_col].values

        # 피처 정보 저장
        feature_info = {
            "vocab_size": len(self.item_vocab),
            "other_feature_dim": other_features.shape[1],
            "feature_names": list(numeric_cols),
            "item_vocab": self.item_vocab,
            "item_freq": self.item_freq,
        }

        # 전체 결과 캐시에 저장
        result = {
            "sequences": processed_sequences,
            "other_features": other_features,
            "targets": targets,
            "feature_info": feature_info,
        }
        with open(full_cache_path, "wb") as f:
            pickle.dump(result, f)

        process_time = time.time() - start_time
        print("처리 완료:")
        print(f"  시퀀스 수: {len(processed_sequences)}")
        print(f"  어휘 크기: {len(self.item_vocab)}")
        print(f"  기타 피처 차원: {other_features.shape[1]}")
        print(f"  기타 피처 샘플 수: {other_features.shape[0]}")
        print(f"  타겟 샘플 수: {len(targets) if targets is not None else 'None'}")
        print(f"전체 처리 시간: {process_time:.2f}초")
        print(f"캐시 저장 완료: {full_cache_path}")

        return processed_sequences, other_features, targets, feature_info

    def save_processor(self, filepath: str):
        """프로세서 저장"""
        processor_data = {
            "max_seq_len": self.max_seq_len,
            "min_item_freq": self.min_item_freq,
            "item_vocab": self.item_vocab,
            "item_freq": self.item_freq,
            "feature_stats": self.feature_stats,
        }

        with open(filepath, "wb") as f:
            pickle.dump(processor_data, f)

        print(f"프로세서 저장 완료: {filepath}")

    def load_processor(self, filepath: str):
        """프로세서 로드"""
        with open(filepath, "rb") as f:
            processor_data = pickle.load(f)

        self.max_seq_len = processor_data["max_seq_len"]
        self.min_item_freq = processor_data["min_item_freq"]
        self.item_vocab = processor_data["item_vocab"]
        self.item_freq = processor_data["item_freq"]
        self.feature_stats = processor_data["feature_stats"]

        print(f"프로세서 로드 완료: {filepath}")


def create_bst_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seq_col: str = "seq",
    target_col: str = "clicked",
    max_seq_len: int = 20,
    min_item_freq: int = 5,
    processor_save_path: Optional[str] = None,
    cache_dir: str = "cache",
    force_refresh: bool = False,
) -> Tuple[
    List[List[int]], np.ndarray, np.ndarray, List[List[int]], np.ndarray, Dict[str, Any]
]:
    """
    BST 모델을 위한 데이터셋 생성 (캐싱 지원)

    Returns:
        train_sequences: 훈련 시퀀스
        train_other_features: 훈련 기타 피처
        train_targets: 훈련 타겟
        test_sequences: 테스트 시퀀스
        test_other_features: 테스트 기타 피처
        feature_info: 피처 정보
    """
    print("=== BST 데이터셋 생성 ===")

    # 전체 데이터셋 캐시 확인
    dataset_cache_key = generate_cache_key(
        "bst_datasets",
        len(train_df),
        len(test_df),
        max_seq_len,
        min_item_freq,
        seq_col,
        target_col,
    )

    dataset_cache_path = get_cache_path(
        cache_dir, f"{dataset_cache_key}_datasets", "pkl"
    )
    if not force_refresh and Path(dataset_cache_path).exists():
        print(f"캐시에서 데이터셋 로드: {dataset_cache_path}")
        with open(dataset_cache_path, "rb") as f:
            cached_datasets = pickle.load(f)
            return (
                cached_datasets["train_sequences"],
                cached_datasets["train_other_features"],
                cached_datasets["train_targets"],
                cached_datasets["test_sequences"],
                cached_datasets["test_other_features"],
                cached_datasets["feature_info"],
            )

    # 피처 프로세서 생성
    processor = BSTFeatureProcessor(
        max_seq_len=max_seq_len, min_item_freq=min_item_freq, cache_dir=cache_dir
    )

    # 훈련 데이터 처리
    print("\n훈련 데이터 처리 중...")
    train_sequences, train_other_features, train_targets, feature_info = (
        processor.process_data(
            train_df,
            seq_col=seq_col,
            target_col=target_col,
            is_training=True,
            force_refresh=force_refresh,
        )
    )

    # 프로세서 저장
    if processor_save_path:
        processor.save_processor(processor_save_path)

    # 테스트 데이터 처리
    print("\n테스트 데이터 처리 중...")
    test_sequences, test_other_features, _, _ = processor.process_data(
        test_df,
        seq_col=seq_col,
        target_col=None,
        is_training=False,
        force_refresh=force_refresh,
    )

    # 전체 데이터셋 캐시에 저장
    datasets_result = {
        "train_sequences": train_sequences,
        "train_other_features": train_other_features,
        "train_targets": train_targets,
        "test_sequences": test_sequences,
        "test_other_features": test_other_features,
        "feature_info": feature_info,
    }
    with open(dataset_cache_path, "wb") as f:
        pickle.dump(datasets_result, f)

    print("\n데이터셋 생성 완료:")
    print(f"  훈련 샘플: {len(train_sequences)}")
    print(f"  테스트 샘플: {len(test_sequences)}")
    print(f"  어휘 크기: {feature_info['vocab_size']}")
    print(f"  기타 피처 차원: {feature_info['other_feature_dim']}")
    print(f"캐시 저장 완료: {dataset_cache_path}")

    # 데이터 길이 일치 확인
    assert len(train_sequences) == len(train_other_features) == len(train_targets), (
        f"훈련 데이터 길이 불일치: sequences={len(train_sequences)}, other_features={len(train_other_features)}, targets={len(train_targets)}"
    )
    assert len(test_sequences) == len(test_other_features), (
        f"테스트 데이터 길이 불일치: sequences={len(test_sequences)}, other_features={len(test_other_features)}"
    )

    return (
        train_sequences,
        train_other_features,
        train_targets,
        test_sequences,
        test_other_features,
        feature_info,
    )
