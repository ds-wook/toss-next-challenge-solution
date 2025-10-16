import numpy as np
import polars as pl
from category_encoders import CatBoostEncoder
from sklearn.model_selection import StratifiedGroupKFold


def kfold_target_encode(
    train_df: pl.DataFrame,
    cat_col: str,
    target_col: str,
    n_splits: int = 7,
    seed: int = 42,
) -> tuple[pl.DataFrame, dict]:
    """
    Train 데이터에 대해 KFold target encoding을 수행합니다.
    - 각 fold의 train으로만 평균을 계산해서 validation에 적용
    - leakage 방지

    Returns:
        tuple: (encoded_train_df, global_mapping)
    """
    kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # numpy로 인덱싱
    y = train_df[target_col].to_numpy()
    cat_values = train_df[cat_col].to_numpy()

    encoded = np.zeros(len(train_df), dtype=float)

    for train_idx, val_idx in kf.split(train_df, y, groups=train_df["day_of_week"]):
        # train fold에서 mean 계산
        train_part = pl.DataFrame(
            {cat_col: cat_values[train_idx], target_col: y[train_idx]}
        )
        mapping = train_part.group_by(cat_col).agg(
            pl.col(target_col).mean().alias("te_value")
        )

        # val fold에 join
        val_part = pl.DataFrame({cat_col: cat_values[val_idx]})
        val_encoded = (
            val_part.join(mapping, on=cat_col, how="left").with_columns(
                pl.col("te_value").fill_null(train_part[target_col].mean())
            )
        )["te_value"].to_numpy()

        encoded[val_idx] = val_encoded

    # 전체 train 데이터로 global mapping 계산 (test 데이터용)
    global_mapping = (
        train_df.group_by(cat_col)
        .agg(pl.col(target_col).mean().alias("te_value"))
        .to_pandas()
        .set_index(cat_col)["te_value"]
        .to_dict()
    )

    global_mean = train_df[target_col].mean()

    # 최종 컬럼 붙이기
    encoded_train_df = train_df.with_columns(pl.Series(f"{cat_col}_kfold_te", encoded))

    return encoded_train_df, {"mapping": global_mapping, "global_mean": global_mean}


def apply_target_encoding(
    df: pl.DataFrame, cat_col: str, mapping_info: dict
) -> pl.DataFrame:
    """
    Test 데이터에 target encoding을 적용합니다.

    Args:
        df: Test DataFrame
        cat_col: Target encoding할 컬럼명
        mapping_info: Train에서 계산된 mapping 정보

    Returns:
        DataFrame with target encoded column
    """
    mapping = mapping_info["mapping"]
    global_mean = mapping_info["global_mean"]

    # mapping 적용
    encoded_values = df[cat_col].map_elements(
        lambda x: mapping.get(x, global_mean), return_dtype=pl.Float64
    )

    return df.with_columns(pl.Series(f"{cat_col}_kfold_te", encoded_values))


def kfold_catboost_encode(
    train_df: pl.DataFrame,
    cat_col: list[str],
    target_col: str,
    n_splits: int = 7,
    seed: int = 42,
) -> tuple[pl.DataFrame, dict]:
    """
    CatBoostEncoder를 활용한 KFold Target Encoding
    - leakage 방지: 각 fold의 train에서만 encoding 학습 후 val에 적용
    """

    # numpy 기반으로 변환
    X = train_df[cat_col].to_pandas()
    y = train_df[target_col].to_numpy()
    groups = train_df["day_of_week"].to_numpy()

    # StratifiedGroupKFold 정의
    kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    encoded = np.zeros((len(train_df), len(cat_col)), dtype=float)

    for train_idx, val_idx in kf.split(X, y, groups=groups):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val = X.iloc[val_idx]

        # CatBoostEncoder 학습
        encoder = CatBoostEncoder(cols=cat_col, random_state=seed)
        encoder.fit(X_train, y_train)

        # validation fold transform
        encoded[val_idx] = encoder.transform(X_val[cat_col]).to_numpy()

    # 최종 컬럼 붙이기
    encoded_train_df = train_df.with_columns(
        [pl.Series(f"{col}_cb_te", encoded[:, i]) for i, col in enumerate(cat_col)]
    )

    # 전체 데이터로 encoder 학습 → test용
    final_encoder = CatBoostEncoder(cols=cat_col, random_state=seed)
    final_encoder.fit(X, y)

    return encoded_train_df, {"encoder": final_encoder, "col": cat_col}


def apply_catboost_encoding(df: pl.DataFrame, mapping_info: dict) -> pl.DataFrame:
    """
    CatBoostEncoder 기반 target encoding을 Test 데이터에 적용
    """
    encoder = mapping_info["encoder"]
    cat_col = mapping_info["col"]

    encoded_values = encoder.transform(df[cat_col].to_pandas())[cat_col].to_numpy()

    return df.with_columns(
        [
            pl.Series(f"{col}_cb_te", encoded_values[:, i])
            for i, col in enumerate(cat_col)
        ]
    )
