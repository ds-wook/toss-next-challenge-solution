import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from features.tcn_encoder import TCNEncoder
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(
        description="Train TCN History Encoder with K-Fold CV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["train", "encode", "train_kfold"],
        required=True,
        help="Mode: train TCN encoder, encode data, or train with K-Fold CV",
    )

    # Data paths
    parser.add_argument(
        "--train_data",
        type=str,
        default="input/toss-next-challenge/train_preprocessed.parquet",
        help="Path to training data",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default=None,  # Will split from train data automatically
        help="Path to validation data (if None, will split from train data)",
    )
    parser.add_argument(
        "--val_split_ratio",
        type=float,
        default=0.2,
        help="Validation split ratio when splitting from train data",
    )
    parser.add_argument(
        "--stratify_column",
        type=str,
        default=None,
        help="Column to use for stratification (if None, uses improved random split)",
    )
    parser.add_argument(
        "--stratify_by_target",
        action="store_true",
        default=True,
        help="Whether to stratify by target variable when using random split",
    )
    parser.add_argument(
        "--no_stratify_by_target",
        action="store_false",
        dest="stratify_by_target",
        help="Disable stratify by target variable",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="input/toss-next-challenge/test_preprocessed.parquet",
        help="Path to test data",
    )

    # K-Fold CV parameters
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of folds for K-Fold cross-validation",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for K-Fold split (must match DCN+MHA training)",
    )

    # Training parameters
    parser.add_argument(
        "--training_mode",
        choices=["reconstruction", "supervised"],
        default="reconstruction",
        help="Training mode: reconstruction (self-supervised) or supervised (CTR prediction)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=32,
        help="Hidden dimension for TCN encoder output",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size for training"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )

    # Model architecture
    parser.add_argument(
        "--num_channels",
        type=int,
        nargs="+",
        default=[16, 32, 32, 32],
        help="Number of channels for each TCN layer",
    )
    parser.add_argument(
        "--kernel_size", type=int, default=2, help="Kernel size for TCN convolutions"
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument(
        "--pooling_method",
        choices=["attention", "average", "max"],
        default="attention",
        help="Pooling method for sequence aggregation",
    )
    parser.add_argument(
        "--decoder_type",
        choices=["lstm", "attention_lstm"],
        default="attention_lstm",
        help="Decoder type for reconstruction: lstm, attention_lstm",
    )

    # Output paths
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="models/tcn_encoder.pt",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save encoded data",
    )

    # Logging
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--log_file", type=str, default="logs/tcn_training.log", help="Path to log file"
    )

    args = parser.parse_args()

    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logger
    logger = setup_logger(log_file=args.log_file)

    logger.info("=" * 60)
    logger.info("TCN History Encoder Training")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Training mode: {args.training_mode}")
    logger.info(f"Hidden dimension: {args.hidden_dim}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Pooling method: {args.pooling_method}")
    logger.info(f"Decoder type: {args.decoder_type}")

    if args.mode == "train_kfold":
        logger.info(f"K-Fold splits: {args.n_splits}")
        logger.info(f"Random state: {args.random_state}")
    else:
        logger.info(f"Validation split ratio: {args.val_split_ratio}")
        logger.info(
            f"Stratify column: {args.stratify_column if args.stratify_column else 'None (improved random split)'}"
        )
        logger.info(f"Stratify by target: {args.stratify_by_target}")
        logger.info(
            f"Validation data: {'Split from train' if args.val_data is None else args.val_data}"
        )

    try:
        # Initialize TCN encoder
        encoder = TCNEncoder()

        if args.mode == "train_kfold":
            logger.info("=" * 60)
            logger.info("Starting TCN encoder training with K-Fold CV")
            logger.info("=" * 60)

            # Check if data files exist
            if not os.path.exists(args.train_data):
                raise FileNotFoundError(f"Training data not found: {args.train_data}")
            if not os.path.exists(args.test_data):
                raise FileNotFoundError(f"Test data not found: {args.test_data}")

            logger.info("K-Fold CV settings:")
            logger.info(f"  - Number of splits: {args.n_splits}")
            logger.info(f"  - Random state: {args.random_state}")
            logger.info(f"  - Training mode: {args.training_mode}")
            logger.info(f"  - Decoder type: {args.decoder_type}")

            # Train and encode for each fold
            encoder.train_kfold(
                train_data_path=args.train_data,
                test_data_path=args.test_data,
                output_dir=args.output_dir,
                n_splits=args.n_splits,
                random_state=args.random_state,
                training_mode=args.training_mode,
                hidden_dim=args.hidden_dim,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                batch_size=args.batch_size,
                decoder_type=args.decoder_type,
            )

            logger.info("=" * 60)
            logger.info("K-Fold TCN training and encoding completed!")
            logger.info("=" * 60)

        elif args.mode == "train":
            logger.info("Starting TCN encoder training...")

            # Check if data files exist
            if not os.path.exists(args.train_data):
                raise FileNotFoundError(f"Training data not found: {args.train_data}")

            # Check validation data if provided
            if args.val_data is not None and not os.path.exists(args.val_data):
                raise FileNotFoundError(f"Validation data not found: {args.val_data}")

            logger.info(f"Using validation split ratio: {args.val_split_ratio}")
            if args.stratify_column:
                logger.info(f"Using stratify column: {args.stratify_column}")
            else:
                logger.info("Using improved random split with the following options:")
                logger.info(f"  - Stratify by target: {args.stratify_by_target}")

            # Train TCN encoder
            encoder.train(
                train_data_path=args.train_data,
                val_data_path=args.val_data,
                val_split_ratio=args.val_split_ratio,
                stratify_column=args.stratify_column,
                stratify_by_target=args.stratify_by_target,
                training_mode=args.training_mode,
                hidden_dim=args.hidden_dim,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                batch_size=args.batch_size,
                decoder_type=args.decoder_type,
            )

            logger.info("TCN encoder training completed successfully!")

            # Encode all data after training
            logger.info("Encoding training data...")
            train_output_path = os.path.join(args.output_dir, "train_with_tcn.parquet")
            encoder.encode_data(
                args.train_data, train_output_path, model_path=args.model_save_path
            )

            logger.info("Encoding test data...")
            test_output_path = os.path.join(args.output_dir, "test_with_tcn.parquet")
            encoder.encode_data(
                args.test_data, test_output_path, model_path=args.model_save_path
            )

            logger.info("Data encoding completed successfully!")

        elif args.mode == "encode":
            logger.info("Starting data encoding with trained model...")

            # Check if model exists
            if not os.path.exists(args.model_save_path):
                raise FileNotFoundError(
                    f"Trained model not found: {args.model_save_path}"
                )

            # Check if data files exist
            if not os.path.exists(args.train_data):
                raise FileNotFoundError(f"Training data not found: {args.train_data}")
            if not os.path.exists(args.test_data):
                raise FileNotFoundError(f"Test data not found: {args.test_data}")

            # Encode data
            train_output_path = os.path.join(args.output_dir, "train_with_tcn.parquet")
            encoder.encode_data(
                args.train_data, train_output_path, model_path=args.model_save_path
            )

            test_output_path = os.path.join(args.output_dir, "test_with_tcn.parquet")
            encoder.encode_data(
                args.test_data, test_output_path, model_path=args.model_save_path
            )

            logger.info("Data encoding completed successfully!")

        logger.info("=" * 60)
        logger.info("TCN History Encoder process completed!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        logger.error("Process failed!")
        raise


if __name__ == "__main__":
    main()
