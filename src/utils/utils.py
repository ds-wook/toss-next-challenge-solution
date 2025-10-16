from typing import Union

import numpy as np
import polars as pl


def reduce_mem_usage(df: pl.DataFrame, verbose: bool = True) -> pl.DataFrame:
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    Optimized for Polars DataFrame.
    """
    # 수치형 타입 정의
    numeric_types = {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }

    start_mem = df.estimated_size() / 1024**2
    optimized_columns = []

    for col in df.columns:
        col_type = df[col].dtype

        # 수치형 컬럼만 최적화
        if col_type in numeric_types:
            # 컬럼의 최솟값과 최댓값 계산
            c_min = df[col].min()
            c_max = df[col].max()

            # 정수형 최적화
            if col_type in {
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
            }:
                new_type = _optimize_int_type(c_min, c_max, col_type)
            # 실수형 최적화
            elif col_type in {pl.Float32, pl.Float64}:
                new_type = _optimize_float_type(c_min, c_max)
            else:
                new_type = col_type

            # 타입이 변경된 경우에만 컬럼 추가
            if new_type != col_type:
                optimized_columns.append(pl.col(col).cast(new_type))
            else:
                optimized_columns.append(pl.col(col))
        else:
            # 수치형이 아닌 컬럼은 그대로 유지
            optimized_columns.append(pl.col(col))

    # 최적화된 컬럼으로 DataFrame 재생성
    if optimized_columns:
        df = df.with_columns(optimized_columns)

    end_mem = df.estimated_size() / 1024**2

    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0
        print(
            f"Mem. usage decreased to {end_mem:5.2f} Mb "
            + f"({reduction:.1f}% reduction)"
        )

    return df


def _optimize_int_type(
    min_val: Union[int, float], max_val: Union[int, float], current_type: pl.DataType
) -> pl.DataType:
    """정수형 타입 최적화"""
    # None 값이 있는 경우 처리
    if min_val is None or max_val is None:
        return current_type

    # 부호 있는 정수형 최적화
    if current_type in {pl.Int8, pl.Int16, pl.Int32, pl.Int64}:
        if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
            return pl.Int8
        elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
            return pl.Int16
        elif min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
            return pl.Int32
        else:
            return pl.Int64

    # 부호 없는 정수형 최적화
    elif current_type in {pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}:
        if min_val >= 0 and max_val <= np.iinfo(np.uint8).max:
            return pl.UInt8
        elif min_val >= 0 and max_val <= np.iinfo(np.uint16).max:
            return pl.UInt16
        elif min_val >= 0 and max_val <= np.iinfo(np.uint32).max:
            return pl.UInt32
        else:
            return pl.UInt64

    return current_type


def _optimize_float_type(
    min_val: Union[int, float], max_val: Union[int, float]
) -> pl.DataType:
    """실수형 타입 최적화"""
    # None 값이 있는 경우 처리
    if min_val is None or max_val is None:
        return pl.Float32

    # Float16은 정밀도가 낮아서 일반적으로 권장하지 않음
    # Float32와 Float64만 사용
    if min_val >= np.finfo(np.float32).min and max_val <= np.finfo(np.float32).max:
        return pl.Float32
    else:
        return pl.Float64


def get_memory_usage(df: pl.DataFrame) -> dict[str, float]:
    """DataFrame의 메모리 사용량 정보 반환"""
    total_size = df.estimated_size() / 1024**2

    # 컬럼별 메모리 사용량
    column_sizes = {}
    for col in df.columns:
        col_size = df.select(pl.col(col)).estimated_size() / 1024**2
        column_sizes[col] = col_size

    return {
        "total_size_mb": total_size,
        "column_sizes": column_sizes,
        "row_count": len(df),
        "column_count": len(df.columns),
    }


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
