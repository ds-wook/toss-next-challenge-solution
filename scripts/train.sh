#!/bin/bash

poetry run python3 src/preprocessor.py


for model in dcn dcn_v2; do
    echo "=========================================="
    echo "Starting training for model: ${model}"
    echo "=========================================="
    echo ""

    for fold in {1..5}; do
        echo "Running training for ${model} fold${fold}..."
        poetry run python3 src/train_fm.py \
            --model ${model} \
            --batch_size 512 \
            --fold_idx_for_fm ${fold} \
            --use_seq_feature \
            --result_path ./res/models/5fold-${model}-mha-concat/fold${fold}

        echo "Completed training on fold${fold}"
        echo "---"
    done

    echo "Completed all folds for ${model}"
    echo ""
done

echo "All dcn trainings finished!"

echo "=========================================="
echo "Starting training for model: lightgbm"
echo "=========================================="
echo ""

for seed in 42 919 1119; do
    poetry run python3 src/train_boosting.py \
        models=lightgbm \
        models.seed=$seed \
        models.params.data_sample_strategy=bagging \
        models.params.boosting_type=dart \
        models.params.learning_rate=0.05 \
        models.params.bagging_seed=$seed \
        models.num_boost_round=2000 \
        models.results=10fold-lightgbm-denoise-category-dart-seed$seed
done

echo "=========================================="
echo "Starting training for model: xgboost"
echo "=========================================="
echo ""

for seed in 42; do
    poetry run python3 src/train_boosting.py \
        models=xgboost \
        models.seed=$seed \
        models.results=10fold-xgboost-denoise-count-gbtree-seed$seed
done
