# #!/bin/bash
poetry run python3 src/preprocessor.py

for model in dcn dcn_v2; do
    echo "=========================================="
    echo "Starting predictions for model: ${model}"
    echo "=========================================="
    echo ""

    for fold in {1..5}; do
        echo "Running prediction for ${model} fold${fold}..."
        poetry run python3 src/predict_fm.py \
            --model ${model} \
            --batch_size 512 \
            --fold_idx_for_fm ${fold} \
            --use_seq_feature \
            --result_path ./res/models/5fold-${model}-mha-concat/fold${fold}

        echo "Completed inference on fold${fold}"
        echo "---"
    done

    echo "Completed all folds for ${model}"
    echo ""

    poetry run python3 src/utils/submission.py \
        --result_path ./res/models/5fold-${model}-mha-concat \
        --k 5 \
        --model_path ./output \
        --model_name 5fold-${model}-mha-concat

    echo "Generated CV submission for ${model}"
    echo ""
done

echo "All predictions for dcn models finished!"

echo "=========================================="
echo "Starting predictions for model: lightgbm"
echo "=========================================="
echo ""

for seed in 42 919 1119; do
poetry run python3 src/predict_boosting.py \
        models=lightgbm \
        models.results=10fold-lightgbm-denoise-category-dart-seed$seed
done

echo "=========================================="
echo "Starting predictions for model: xgboost"
echo "=========================================="
echo ""

poetry run python3 src/predict_boosting.py \
        models=xgboost \
        models.results=10fold-xgboost-denoise-count-gbtree-seed42

echo "=========================================="
echo "Starting ensemble"
echo "=========================================="
echo ""

poetry run python3 src/ensemble.py \
        blends=total
