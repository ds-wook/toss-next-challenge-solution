from typing import List
import argparse
import os
import importlib

import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm


from utils.config import load_yaml, replace_config_with_args
from data.fm import FMDataLoader
from data.dataset import create_torch_data_loader

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
CONFIG_PATH = os.path.join(ROOT_PATH, "./config/models/fm.yaml")


def load_model(
    args: argparse.ArgumentParser,
    device: str,
):
    # Load the saved checkpoint
    checkpoint_path = os.path.join(args.result_path, "model_weight.pt")
    checkpoint = torch.load(
        checkpoint_path, map_location=device
    )  # or map_location='cuda' if using GPU

    # Extract the config and state_dict
    config = checkpoint["config"]
    state_dict = checkpoint["state_dict"]

    model_path = (
        f"models.fm.{args.model}_seq"
        if args.use_seq_feature
        else f"models.fm.{args.model}"
    )
    model_module = importlib.import_module(model_path).Model
    model = model_module(**config)

    # Load the weights into the model
    model.load_state_dict(state_dict)

    # Move model to device (GPU/CPU)
    model = model.to(device)

    # Set to evaluation mode if you're doing inference
    model.eval()

    return model


def inference(
    model: nn.Module,
    test_dataloader: DataLoader,
    test_ids: List[str],
    device: str,
    args: argparse.ArgumentParser,
):
    predictions = []

    for batch_idx, batch_data in tqdm(
        enumerate(test_dataloader), desc="Running inference", total=len(test_dataloader)
    ):
        if len(batch_data) == 4:
            num_features, cat_features, seq, labels = batch_data
            seq = seq.to(device)
        else:
            num_features, cat_features, labels = batch_data
            seq = None
        num_features = num_features.to(device)
        cat_features = cat_features.to(device)
        labels = labels.to(device)
        pred = model(num_features, cat_features, seq=seq)
        pred_probs = torch.sigmoid(pred).squeeze(1).cpu().detach().numpy()
        pred_probs = np.clip(
            pred_probs, 1e-15, 1 - 1e-15
        )  # Clip for log loss stability
        predictions.extend(pred_probs.tolist())

    # Create submission DataFrame with ID and prediction columns
    # probabilities are already computed with sigmoid and clipped
    submission_df = pd.DataFrame({"ID": test_ids, "clicked": predictions})

    # Save to CSV
    submission_path = os.path.join(args.result_path, "submission.csv")
    print(f"Inference completed. Submission file generated in {submission_path}")
    submission_df.to_csv(submission_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "lr",
            "fm",
            "deepfm",
            "xdeepfm",
            "ffm",
            "fibinet",
            "dcn",
            "dcn_v2",
        ],
    )
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--fold_idx_for_fm", type=int, default=1)
    parser.add_argument("--use_seq_feature", action="store_true")
    parser.add_argument("--result_path", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    # load config
    config = load_yaml(CONFIG_PATH.format(model=args.model))
    # replace config from args
    config = replace_config_with_args(args, config, args.result_path)
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load test dataset
    data_loader = FMDataLoader(config)
    test_x, test_ids, test_seq = data_loader.load_test_data(is_fm=True)
    test_dataloader = create_torch_data_loader(
        test_x=test_x,
        test_seq=test_seq,
        test_only=True,  # return test_dataloader only
        batch_size=config.common.batch_size,
        num_numeric_feat=len(config.data.num_features),
        num_cat_feat=len(config.data.cat_features),
        num_workers=config.common.num_workers,
        stratified_sampling=True,
        positive_ratio=0.02,  # ratio of positive samples
        seed=config.data.seed,
        use_seq_feature=args.use_seq_feature,
        max_seq_length=config.data.max_seq_length,
        recent_ratio=config.data.recent_ratio,
    )

    model = load_model(
        args=args,
        device=device,
    )

    print("Completed loading model and data. Starting inference...")

    inference(
        model=model,
        test_dataloader=test_dataloader,
        test_ids=test_ids,
        device=device,
        args=args,
    )


if __name__ == "__main__":
    main()
