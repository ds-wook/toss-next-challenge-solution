# toss-next-challenge-solution
This repository for the 5th Place Model in the Ad Click-Through Rate (CTR) Prediction Competition with toss.

## Setting up environment

We use [poetry](https://github.com/python-poetry/poetry) to manage dependencies of repository.

Use poetry with version `2.1.1`.

```shell
$ poetry --version
Poetry (version 2.1.1)
```

Python version should be `3.11.x`.

```shell
$ python --version
Python 3.11.11
```

If python version is lower than `3.11`, try installing required version using `pyenv`.

Create virtual environment.

```shell
$ poetry env activate
```

If your global python version is not 3.11, run following command.

```shell
$ poetry env use python3.11
```

You can check virtual environment path info and its executable python path using following command.

```shell
$ poetry env info
```

After setting up python version, just run following command which will install all the required packages from `poetry.lock`.

```shell
$ poetry install
```

## Setting up git hook

Set up automatic linting using the following commands:
```shell
# This command will ensure linting runs automatically every time you commit code.
poetry run pre-commit install
```

### Note

If you want to add package to `pyproject.toml`, please use following command.

```shell
$ poetry add "package==1.0.0"
```

Then, update `poetry.lock` to ensure that repository members share same environment setting.

```shell
$ poetry lock
```

## Architecture of Our Solution

### Ensemble Architecture

![Ensemble](https://github.com/user-attachments/assets/6bba8d01-c5e1-4744-a7d0-0ebd6d38ffcf)

### Boosting
**LightGBM DART**

[Dart](https://arxiv.org/abs/1505.01866) introduces dropout into gradient boosting, randomly dropping trees during training to prevent overfitting and improve generalization.

We found DART particularly effective for this task because:
- It handles sparse and high-cardinality categorical features efficiently.
- It achieves stable validation performance across folds.
- It generalizes well under data drift and imbalanced conditions.

While we also experimented with XGBoost, CatBoost, and Deep Cross Network (DCN),DART consistently served as the backbone model and delivered the highest overall reliability.
Final submissions were built around DART and refined through ensemble blending with complementary models.

### Deep Cross Network Architecture

**Seq-aware DCN**

[DCN](https://arxiv.org/abs/1708.05123) with MHA encoded seq feature.

![Seq-aware DCN](https://github.com/user-attachments/assets/44bfb186-313c-401c-80f5-d1a0eb6f9c37)

**Seq-aware DCN V2**

[DCN V2](https://arxiv.org/abs/2008.13535) with MHA encoded seq feature.

![Seq-aware DCN V2](https://github.com/user-attachments/assets/4ae3802d-1e89-4763-892f-830a9634e8be)


## How to Run Our Solution
### 1. Prepare the input data
Place the following files inside the `input/toss-next-challenge/` directory:
```
├── input
   └── toss-next-challenge
       ├── sample_submission.csv
       ├── test.parquet
       └── train.parquet
```

### 2. Run the following script:
- train
    ```shell
    $ sh scripts/train.sh
    ```

- inference

    ```shell
    $ sh scripts/inference.sh
    ```

### 3. The final submission file will be generated in the output folder as
`tree4-dcn2-mha-concatmod-sigmoid-ensemble.csv`.
Please use this CSV file for evaluation.
