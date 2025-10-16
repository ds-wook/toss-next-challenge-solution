from pathlib import Path

import pandas as pd


def generate_kfold_submission(
    result_path: str, k: int, model_path: str, model_name: str
):
    submissions = []
    for i in range(1, k + 1):
        submissions.append(
            pd.read_csv(Path(result_path) / f"fold{i}" / "submission.csv")
        )

    # ids sanity check
    n = submissions[0].shape[0]
    for i in range(k - 1):
        assert (submissions[i].ID == submissions[i + 1].ID).sum() == n, (
            "IDs do not match across folds"
        )

    cv_submission = pd.DataFrame(
        {
            "ID": submissions[0]["ID"].tolist(),
            "clicked": sum(sub["clicked"] for sub in submissions) / len(submissions),
        }
    )

    cv_submission.to_csv(Path(model_path) / f"{model_name}.csv", index=False)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    generate_kfold_submission(
        args.result_path, args.k, args.model_path, args.model_name
    )


if __name__ == "__main__":
    main()
