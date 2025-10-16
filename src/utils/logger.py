import argparse
import logging

import torch
from easydict import EasyDict


def setup_logger(log_file: str):
    # Create a logger object
    logger = logging.getLogger("toss")
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers to prevent duplicates
    if logger.handlers:
        logger.handlers.clear()

    # Create a file handler to log messages to a file
    file_handler = logging.FileHandler(
        log_file, mode="w"
    )  # Open file in write mode to overwrite on each run

    # Set a logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Create a console handler to log messages to the terminal
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def config_logging(
    logger: logging.Logger,
    args: argparse.ArgumentParser,
    config: EasyDict,
    result_path: str,
):
    logger.info("########## Experiment Configuration ##########")
    logger.info("[Common Configs]")
    logger.info(f"selected model: {args.model}")
    logger.info(f"selected loss function: {config.loss.name}")
    logger.info(f"batch size: {config.common.batch_size}")
    logger.info(f"learning rate: {config.common.learning_rate}")
    logger.info(f"regularization: {config.common.regularization}")
    logger.info(f"epochs: {config.common.epochs}")
    logger.info(f"embedding dim: {config.common.embedding_dim}")
    logger.info(f"patience: {config.common.patience}")
    logger.info(f"num_workers: {config.common.num_workers}")
    logger.info(f"result path: {result_path}")
    logger.info(f"stratified fold index: {config.data.fold_idx_for_fm}")
    logger.info(f"use_seq_feature: {args.use_seq_feature}")
    logger.info(f"is_test: {args.is_test}")
    # Add device detection and setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
    logger.info("[Loss Configs]")
    if config.loss.name == "bce_with_logits":
        logger.info(f"positive weight in BCE: {config.loss.bce.pos_weight}")
    if config.loss.name == "focal":
        logger.info(f"gamma in focal: {config.loss.focal.gamma}")
        logger.info(
            f"pos_lr_multiplier in focal: {config.loss.focal.pos_lr_multiplier}"
        )
        logger.info(
            f"neg_lr_multiplier in focal: {config.loss.focal.neg_lr_multiplier}"
        )
    logger.info("[Model Configs]")
    if args.model in ["deepfm", "xdeepfm"]:
        logger.info(f"mlp_dims: {config.model.deepfm.mlp_dims}")
        logger.info(f"cin_layer_dims: {config.model.deepfm.cin_layer_dims}")
    if args.model == "fibinet":
        logger.info(f"reduction_ratio: {config.model.fibinet.reduction_ratio}")
    if args.model == "dcn":
        logger.info(f"num_cross_layers: {config.model.dcn.num_cross_layers}")
        logger.info(f"deep_layers: {config.model.dcn.deep_layers}")
    if args.model == "dcn_v2":
        logger.info(f"num_cross_layers: {config.model.dcn_v2.num_cross_layers}")
        logger.info(f"deep_layers: {config.model.dcn_v2.deep_layers}")
        logger.info(f"low_rank: {config.model.dcn_v2.low_rank}")
        logger.info(f"num_experts: {config.model.dcn_v2.num_experts}")
        logger.info(f"structure: {config.model.dcn_v2.structure}")
    if args.use_seq_feature:
        logger.info(f"d_model: {config.model.seq.d_model}")
        logger.info(f"nhead: {config.model.seq.nhead}")
        logger.info(f"use_causal_mask: {config.model.seq.use_causal_mask}")
