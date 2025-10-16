"""
transformer data preprocessing that prevents data leakage
"""

import numpy as np
from typing import List, Dict, Tuple, Any
import pandas as pd
import yaml
from pathlib import Path


def preprocess_numerical_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_cols: List[str],
    denoise_steps: Dict[str, float] = None,
    preprocessing_config: Dict[str, Any] = None,
    logger=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Preprocess numerical features with improved stability"""
    from sklearn.preprocessing import RobustScaler
    import numpy as np

    scalers = {}

    if denoise_steps is None:
        denoise_steps = {}

    if preprocessing_config is None:
        preprocessing_config = {}

    # Default preprocessing settings
    use_quantile_clipping = preprocessing_config.get("use_quantile_clipping", True)
    quantile_lower = preprocessing_config.get("quantile_lower", 0.01)
    quantile_upper = preprocessing_config.get("quantile_upper", 0.99)
    use_log_transform = preprocessing_config.get("use_log_transform", False)
    log_transform_cols = preprocessing_config.get("log_transform_cols", [])

    for col in num_cols:
        if logger:
            logger.info(f"Processing numerical feature: {col}")

        # Check if column is actually numeric
        if not pd.api.types.is_numeric_dtype(train_df[col]):
            if logger:
                logger.warning(
                    f"Skipping non-numeric column {col} (type: {train_df[col].dtype})"
                )
            continue

        # Step 1: Quantile-based clipping for stability
        if use_quantile_clipping:
            lower_bound = train_df[col].quantile(quantile_lower)
            upper_bound = train_df[col].quantile(quantile_upper)

            train_df[col] = train_df[col].clip(lower_bound, upper_bound)
            test_df[col] = test_df[col].clip(lower_bound, upper_bound)

            if logger:
                logger.info(
                    f"{col} quantile clipping: [{lower_bound:.4f}, {upper_bound:.4f}]"
                )

        # Step 2: Log transformation for skewed features
        if use_log_transform and col in log_transform_cols:
            # Ensure non-negative values for log transformation
            min_val = min(train_df[col].min(), test_df[col].min())
            if min_val <= 0:
                offset = abs(min_val) + 1
                train_df[col] = train_df[col] + offset
                test_df[col] = test_df[col] + offset
                if logger:
                    logger.info(f"{col} log transform with offset: {offset}")

            train_df[col] = np.log1p(train_df[col])
            test_df[col] = np.log1p(test_df[col])

            if logger:
                logger.info(f"{col} log1p transformation applied")

        # Step 3: IQR-based outlier removal (if specified)
        if col in denoise_steps:
            threshold = denoise_steps[col]
            if logger:
                logger.info(f"{col} IQR outlier removal (threshold: {threshold})")

            Q1 = train_df[col].quantile(0.25)
            Q3 = train_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            train_df[col] = train_df[col].clip(lower_bound, upper_bound)
            test_df[col] = test_df[col].clip(lower_bound, upper_bound)

        # Step 4: Robust scaling
        scaler = RobustScaler()
        train_df[col] = scaler.fit_transform(train_df[[col]]).flatten()
        test_df[col] = scaler.transform(test_df[[col]]).flatten()
        scalers[col] = scaler

        if logger:
            logger.info(
                f"{col} RobustScaler applied - mean: {scaler.center_[0]:.4f}, scale: {scaler.scale_[0]:.4f}"
            )

    return train_df, test_df, scalers


def encode_categoricals(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cat_cols: List[str], logger=None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Encode categorical features with data leakage prevention"""
    from sklearn.preprocessing import LabelEncoder
    import numpy as np

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()

        # CRITICAL: Only fit on training data to prevent data leakage
        train_values = train_df[col].astype(str).fillna("UNK")
        le.fit(train_values)

        # Transform training data
        train_df[col] = le.transform(train_values)

        # Transform test data - unknown categories become UNK
        test_values = test_df[col].astype(str).fillna("UNK")
        # Map unknown categories to UNK
        test_values = test_values.apply(lambda x: x if x in le.classes_ else "UNK")

        # Handle UNK in test data by creating a new encoder with UNK
        if "UNK" not in le.classes_:
            # Create a new encoder that includes UNK
            le_with_unk = LabelEncoder()
            all_classes = np.append(le.classes_, "UNK")
            le_with_unk.fit(all_classes)
            test_df[col] = le_with_unk.transform(test_values)
            encoders[col] = le_with_unk
        else:
            test_df[col] = le.transform(test_values)
            encoders[col] = le

        # Log statistics
        final_encoder = encoders[col]
        unique_train = len(le.classes_)
        unique_test = len(test_df[col].unique())
        unk_count = (
            (test_df[col] == final_encoder.transform(["UNK"])[0]).sum()
            if "UNK" in final_encoder.classes_
            else 0
        )

        if logger:
            logger.info(
                f"{col} encoding: {unique_train} unique categories, {unique_test} test unique, {unk_count} UNK in test"
            )
        else:
            print(
                f"{col} encoding: {unique_train} unique categories, {unique_test} test unique, {unk_count} UNK in test"
            )

    return train_df, test_df, encoders


def analyze_sequence_lengths(
    df: pd.DataFrame, seq_col: str, logger=None
) -> Dict[str, float]:
    """Analyze sequence length distribution"""
    seq_lengths = []
    for seq_str in df[seq_col].astype(str):
        if seq_str and seq_str != "nan":
            seq_len = len(seq_str.split(","))
            seq_lengths.append(seq_len)
        else:
            seq_lengths.append(0)

    seq_lengths = np.array(seq_lengths)

    stats = {
        "mean": seq_lengths.mean(),
        "median": np.median(seq_lengths),
        "p85": np.percentile(seq_lengths, 85),
        "p95": np.percentile(seq_lengths, 95),
        "p99": np.percentile(seq_lengths, 99),
        "max": seq_lengths.max(),
    }

    if logger:
        logger.info("시퀀스 길이 통계:")
        logger.info(f"  - 평균: {stats['mean']:.2f}")
        logger.info(f"  - 중간값: {stats['median']:.2f}")
        logger.info(f"  - 85%ile: {stats['p85']:.2f}")
        logger.info(f"  - 95%ile: {stats['p95']:.2f}")
        logger.info(f"  - 99%ile: {stats['p99']:.2f}")
        logger.info(f"  - 최대값: {stats['max']}")
    else:
        print("시퀀스 길이 통계:")
        print(f"  - 평균: {stats['mean']:.2f}")
        print(f"  - 중간값: {stats['median']:.2f}")
        print(f"  - 85%ile: {stats['p85']:.2f}")
        print(f"  - 95%ile: {stats['p95']:.2f}")
        print(f"  - 99%ile: {stats['p99']:.2f}")
        print(f"  - 최대값: {stats['max']}")

    return stats


def preprocess_transformer_data(
    train: pd.DataFrame, test: pd.DataFrame, config: Dict[str, Any], logger=None
) -> Dict[str, Any]:
    """
     preprocessing pipeline that does NOT apply sampling here.
    Sampling will be applied within each CV fold to prevent data leakage.
    """
    # Load dataset config
    config_path = Path(config["data"]["config_path"])
    with open(config_path, "r", encoding="utf-8") as f:
        dataset_config = yaml.safe_load(f)

    # Feature configuration
    target_col = dataset_config.get("target", "clicked")
    seq_col = "seq"
    id_col = dataset_config.get("id", "ID")

    # IMPORTANT: DO NOT apply negative sampling here!
    # This was the source of data leakage in the original implementation
    if config["data"].get("use_sampling", False):
        if logger:
            logger.warning(
                "⚠️  Sampling configuration detected but NOT applied here to prevent data leakage"
            )
            logger.warning(
                "⚠️  Sampling will be applied within each CV fold during training"
            )
        else:
            print(
                "⚠️  Sampling configuration detected but NOT applied here to prevent data leakage"
            )
            print("⚠️  Sampling will be applied within each CV fold during training")

    # Use existing get_feature_columns function
    from data.loader import get_feature_columns

    exclude_cols = [target_col, seq_col, id_col]
    feature_cols = get_feature_columns(train, exclude_cols)

    # Apply drop_features from dataset config
    drop_features = dataset_config.get("drop_features", [])
    original_feature_count = len(feature_cols)
    feature_cols = [col for col in feature_cols if col not in drop_features]
    dropped_count = original_feature_count - len(feature_cols)

    if logger:
        logger.info(f"원본 피처 수: {original_feature_count}")
        logger.info(f"제거된 피처 수: {dropped_count}")
        logger.info(f"최종 사용할 피처 수: {len(feature_cols)}")
        if dropped_count > 0:
            logger.info(f"제거된 피처: {drop_features}")
    else:
        print(f"원본 피처 수: {original_feature_count}")
        print(f"제거된 피처 수: {dropped_count}")
        print(f"최종 사용할 피처 수: {len(feature_cols)}")
        if dropped_count > 0:
            print(f"제거된 피처: {drop_features}")

    # Categorical features
    categorical_features = dataset_config.get("cat_features", [])
    cat_cols = [col for col in categorical_features if col in feature_cols]

    # Numerical features
    num_cols = [c for c in feature_cols if c not in cat_cols]

    if logger:
        logger.info(f"Num features: {len(num_cols)} | Cat features: {len(cat_cols)}")
        logger.info(f"Categorical features: {cat_cols}")
    else:
        print(f"Num features: {len(num_cols)} | Cat features: {len(cat_cols)}")
        print(f"Categorical features: {cat_cols}")

    # Preprocess numerical features
    if logger:
        logger.info("수치형 피처 전처리 시작")
    else:
        print("수치형 피처 전처리 시작")
    denoise_steps = dataset_config.get("denoise_steps", {})
    preprocessing_config = config["data"].get("preprocessing", {})
    train, test, num_scalers = preprocess_numerical_features(
        train, test, num_cols, denoise_steps, preprocessing_config, logger
    )

    # Encode categorical features
    if logger:
        logger.info("범주형 피처 인코딩 시작")
    else:
        print("범주형 피처 인코딩 시작")
    train, test, cat_encoders = encode_categoricals(train, test, cat_cols, logger)

    # Analyze sequence lengths
    if logger:
        logger.info("시퀀스 길이 분포 분석 중...")
    else:
        print("시퀀스 길이 분포 분석 중...")
    seq_stats = analyze_sequence_lengths(train, seq_col, logger)

    # Set maximum sequence length
    max_seq_length = min(int(seq_stats["p85"]), config.get("max_seq_length", 512))
    if logger:
        logger.info(f"시퀀스 최대 길이 제한: {max_seq_length}")
    else:
        print(f"시퀀스 최대 길이 제한: {max_seq_length}")

    return {
        "train": train,  # Original train data WITHOUT sampling
        "test": test,
        "target_col": target_col,
        "seq_col": seq_col,
        "id_col": id_col,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_encoders": cat_encoders,
        "max_seq_length": max_seq_length,
        # Store sampling config for use in CV folds
        "sampling_config": {
            "use_sampling": config["data"].get("use_sampling", False),
            "sampling_ratio": config["data"].get("sampling_ratio", 0.5),
            "seed": config["data"].get("seed", 42),
        },
    }
