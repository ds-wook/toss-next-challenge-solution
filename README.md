# toss-next-challenge-solution
토스 NEXT ML CHALLENGE : 광고 클릭 예측(CTR) 대회 5등 모델 제출용 레포지토리

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

## Run
### 1. Prepare the input data
Place the following files inside the `input/toss-next-challenge/` directory:
```
├── input
│   └── toss-next-challenge
│       ├── sample_submission.csv
│       ├── test.parquet
│       └── train.parquet
```

2. Run the following script:
- train
    ```shell
    $ sh scripts/train.sh
    ```

- inference

    ```shell
    $ sh scripts/inference.sh
    ```

3. The final submission file will be generated in the output folder as
tree4-dcn2-mha-concatmod-sigmoid-ensemble.csv.
Please use this CSV file for evaluation.
