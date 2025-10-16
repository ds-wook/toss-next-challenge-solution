import argparse


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
    parser.add_argument("--postfix", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--use_seq_feature", action="store_true")
    parser.add_argument("--is_test", action="store_true")
    return parser.parse_args()
