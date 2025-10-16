import os
import pickle

import numpy as np
import polars as pl
import scipy.sparse as sp
from easydict import EasyDict
from omegaconf import DictConfig
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer

from data.base import BaseDataLoader


class FMDataLoader(BaseDataLoader):
    def __init__(self, cfg: DictConfig | EasyDict):
        super().__init__(cfg)
        self.fitted_params = {}
        self.is_fitted = False
        self.cfg.data.all_features = []

        # drop features in numerical and categorical features
        self.cfg.data.num_features = [
            col
            for col in self.cfg.data.num_features + self.cfg.data.engineered_features
            if col not in self.cfg.data.drop_features
        ]
        self.cfg.data.cat_features = [
            col
            for col in self.cfg.data.cat_features
            if col not in self.cfg.data.drop_features
        ]

    def preprocess_train_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fit preprocessing parameters on training data and transform"""
        # select only necessary columns
        columns_to_select = (
            self.cfg.data.cat_features
            + self.cfg.data.num_features
            + [self.cfg.data.target]
        )
        df = df.select([col for col in columns_to_select if col in df.columns])

        # Fit and transform
        df = self._fit_fill_missing_categorical_features(df)
        df = self._fit_fill_missing_numerical_features(df)
        df = self._fit_encode_categorical_variables_with_ordinal(df)
        df = self._fit_gaussrank_scale_numerical_features(df)

        self.is_fitted = True

        # save train preprocessing parameters for future test inference
        pickle.dump(
            self.fitted_params,
            open(os.path.join(self.cfg.data.result_path, "fitted_params.pkl"), "wb"),
        )

        return df

    def preprocess_test_data(
        self, df: pl.DataFrame, is_validation: bool = False
    ) -> pl.DataFrame:
        """Apply fitted preprocessing to new data (test set)"""
        # if directly loading trained model, preprocessing params need to be loaded
        if not self.is_fitted:
            self.fitted_params = pickle.load(
                open(os.path.join(self.cfg.data.result_path, "fitted_params.pkl"), "rb")
            )
            self.is_fitted = True
            self.cfg.data.all_features = (
                self.cfg.data.num_features
                + self.fitted_params.get("encoded_cat_features", [])
            )
            print("Loaded fitted preprocessing parameters for test data.")

        # select only necessary columns (excluding target for test data)
        columns_to_select = self.cfg.data.cat_features + self.cfg.data.num_features
        if is_validation:
            columns_to_select.append(self.cfg.data.target)
        else:
            columns_to_select.append(self.cfg.data.id)

        df = df.select([col for col in columns_to_select if col in df.columns])

        # Transform using fitted parameters
        df = self._transform_fill_missing_categorical_features(df)
        df = self._transform_fill_missing_numerical_features(df)
        df = self._transform_encode_categorical_variables_with_ordinal(df)
        df = self._transform_gaussrank_scale_numerical_features(df)

        return df

    def _to_sparse_matrix_batched(
        self, df: pl.DataFrame, one_hot_columns: list, batch_size: int = 100000
    ):
        """Convert one-hot encoded DataFrame to sparse matrix in batches to avoid OOM"""
        n_rows = df.height
        n_cols = len(one_hot_columns)

        # Initialize list to store sparse matrices
        sparse_matrices = []

        # Process in batches
        for start_idx in range(0, n_rows, batch_size):
            end_idx = min(start_idx + batch_size, n_rows)

            # Get batch slice
            batch_df = df.slice(start_idx, end_idx - start_idx)

            # Select only one-hot columns
            batch_cat = batch_df.select(one_hot_columns)

            # Convert to numpy and then to sparse
            batch_numpy = batch_cat.to_numpy()
            batch_sparse = sp.csr_matrix(batch_numpy)

            sparse_matrices.append(batch_sparse)

        # Vertically stack all sparse matrices
        if sparse_matrices:
            final_sparse_matrix = sp.vstack(sparse_matrices)
            return final_sparse_matrix
        else:
            # Return empty sparse matrix if no data
            return sp.csr_matrix((n_rows, n_cols))

    def _fit_encode_categorical_variables_with_ordinal(
        self, df: pl.DataFrame
    ) -> pl.DataFrame:
        """Fit and encode categorical variables with OrdinalEncoder"""
        if not self.cfg.data.cat_features:
            return df

        print("Fitting OrdinalEncoder for categorical features...")

        # Initialize encoders dictionary
        categorical_encoders = {}

        # store number of unique categories for each categorical feature
        self.categorical_field_dims = []

        for col in self.cfg.data.cat_features:
            if col in df.columns:
                # Create OrdinalEncoder
                encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int32
                )

                # Fit encoder
                col_data = df[col].to_numpy().reshape(-1, 1)
                encoder.fit(col_data)
                categorical_encoders[col] = encoder
                field_dims = len(encoder.categories_[0])
                self.categorical_field_dims.append(field_dims)

                print(f"  {col}: {field_dims} unique categories")

        # Store fitted encoders
        self.fitted_params["categorical_encoders"] = categorical_encoders

        # Apply encoding
        for col in self.cfg.data.cat_features:
            if col in df.columns and col in categorical_encoders:
                encoder = categorical_encoders[col]
                col_data = df[col].to_numpy().reshape(-1, 1)
                encoded_values = encoder.transform(col_data).flatten()

                df = df.with_columns(
                    pl.Series(name=col, values=encoded_values).cast(pl.Int32)
                )

        # Update all_features list
        self.cfg.data.all_features = (
            self.cfg.data.num_features + self.cfg.data.cat_features
        )

        return df

    def _fit_fill_missing_categorical_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fit and fill missing values in categorical features with 'NA' string"""
        if not self.cfg.data.cat_features:
            return df

        # Store the fill value for categorical features
        self.fitted_params["cat_fill_value"] = "NA"

        # Fill null values with "NA" for all categorical features
        fill_expressions = [
            pl.col(col).fill_null(self.fitted_params["cat_fill_value"])
            for col in self.cfg.data.cat_features
            if col in df.columns
        ]

        if fill_expressions:
            df = df.with_columns(fill_expressions)

        return df

    def _fit_fill_missing_numerical_features(
        self, df: pl.DataFrame, fill_with_mean: bool = True
    ) -> pl.DataFrame:
        """Fit and fill missing values in numerical features with column mean"""
        if not self.cfg.data.num_features:
            return df

        # Calculate and store means for numerical features
        means = {}
        for col in self.cfg.data.num_features:
            if col in df.columns:
                means[col] = df[col].mean() if fill_with_mean else 0.0
        self.fitted_params["num_fill_value"] = means

        # Fill null values with mean for all numerical features
        fill_expressions = [
            pl.col(col).fill_null(self.fitted_params["num_fill_value"][col])
            for col in self.cfg.data.num_features
            if col in df.columns and col in self.fitted_params["num_fill_value"]
        ]

        if fill_expressions:
            df = df.with_columns(fill_expressions)

        return df

    def _fit_gaussrank_scale_numerical_features(self, df: pl.DataFrame) -> pl.DataFrame:
        # Initialize encoders dictionary
        gaussrank_scalers = {}

        for col in self.cfg.data.num_features:
            if col in df.columns:
                scaler = QuantileTransformer(
                    n_quantiles=100, output_distribution="normal"
                )
                scaled_values = scaler.fit_transform(
                    df[col].to_numpy().reshape(-1, 1)
                ).flatten()
                gaussrank_scalers[col] = scaler
                df = df.with_columns(
                    pl.Series(name=col, values=scaled_values).cast(pl.Float32)
                )

        # Store fitted scalers
        self.fitted_params["gaussrank_scalers"] = gaussrank_scalers

        return df

    def _transform_encode_categorical_variables_with_ordinal(
        self, df: pl.DataFrame
    ) -> pl.DataFrame:
        """Transform categorical variables using fitted OrdinalEncoder"""
        if (
            not self.cfg.data.cat_features
            or "categorical_encoders" not in self.fitted_params
        ):
            return df

        print("Transforming categorical features with fitted OrdinalEncoder...")

        categorical_encoders = self.fitted_params["categorical_encoders"]

        # Apply encoding using fitted encoders
        for col in self.cfg.data.cat_features:
            if col in df.columns and col in categorical_encoders:
                encoder = categorical_encoders[col]
                encoded_values = encoder.transform(
                    df[col].to_numpy().reshape(-1, 1)
                ).flatten()
                df = df.with_columns(
                    pl.Series(name=col, values=encoded_values).cast(pl.Int32)
                )
            elif col in df.columns:
                print(f"Warning: No fitted encoder found for column {col}")
                # Fill with unknown value if no encoder
                df = df.with_columns(pl.col(col).fill_null(-1).cast(pl.Int32))

        return df

    def _transform_fill_missing_categorical_features(
        self, df: pl.DataFrame
    ) -> pl.DataFrame:
        """Transform: fill missing values in categorical features using fitted parameters"""
        if not self.cfg.data.cat_features or "cat_fill_value" not in self.fitted_params:
            return df

        # Fill null values with fitted fill value
        fill_expressions = [
            pl.col(col).fill_null(self.fitted_params["cat_fill_value"])
            for col in self.cfg.data.cat_features
            if col in df.columns
        ]

        if fill_expressions:
            df = df.with_columns(fill_expressions)

        return df

    def _transform_fill_missing_numerical_features(
        self, df: pl.DataFrame
    ) -> pl.DataFrame:
        """Transform: fill missing values in numerical features using fitted means"""
        if not self.cfg.data.num_features or "num_fill_value" not in self.fitted_params:
            return df

        # Fill null values with fitted means
        fill_expressions = [
            pl.col(col).fill_null(self.fitted_params["num_fill_value"][col])
            for col in self.cfg.data.num_features
            if col in df.columns and col in self.fitted_params["num_fill_value"]
        ]

        if fill_expressions:
            df = df.with_columns(fill_expressions)

        return df

    def _transform_gaussrank_scale_numerical_features(
        self, df: pl.DataFrame
    ) -> pl.DataFrame:
        if (
            not self.cfg.data.num_features
            or "gaussrank_scalers" not in self.fitted_params
        ):
            return df

        print("Transforming numerical features with fitted GaussRank scalers...")

        gaussrank_scalers = self.fitted_params["gaussrank_scalers"]

        # Apply scaling using fitted scalers
        for col in self.cfg.data.num_features:
            if col in df.columns and col in gaussrank_scalers:
                scaler = gaussrank_scalers[col]
                scaled_values = scaler.transform(
                    df[col].to_numpy().reshape(-1, 1)
                ).flatten()
                df = df.with_columns(
                    pl.Series(name=col, values=scaled_values).cast(pl.Float32)
                )

        return df
