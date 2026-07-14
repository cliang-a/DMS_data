#!/bin/bash

# Stop execution immediately if any command fails
set -e

# 🔒 Retrieve and lock the absolute root directory of this script
ROOT_DIR=$(pwd)

# ================= Default Parameters =================
SEEDS=(1 42 100)
TOP_RATIO=0.3
BOTTOM_RATIO=0.3

# Output directory name (aligned with configurations in Python scripts)
OUTPUT_DIR="simulation_results_hit_rate_recall"

# ================= Parse Command Line Arguments =================
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --top) TOP_RATIO="$2"; shift ;;
        --bottom) BOTTOM_RATIO="$2"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

# Ensure the output directory exists to prevent errors during renaming on the first run
mkdir -p "$OUTPUT_DIR"

# ================= Core Loop Over Seeds =================
for SEED in "${SEEDS[@]}"; do

    echo "================================================="
    echo "🚀 Starting workflow (Current seed: ${SEED})..."
    echo "Parameters: random_seed=${SEED}, top_ratio=${TOP_RATIO}, bottom_ratio=${BOTTOM_RATIO}"
    echo "================================================="

    # 🔄 Return to root directory
    cd "$ROOT_DIR"

    # ================= 1. Dataset Splitting =================
    echo ">>> [1/4] Entering ./data_split directory, modifying and running data_split.py"
    cd ./data_split
    # Dynamically replace random_seed, n_top, and n_bottom parameters
    sed -i -E "s/random_seed = [0-9]+/random_seed = ${SEED}/g" data_split.py
    sed -i -E "s/n_top = int\(N \* ?[0-9.]+\)/n_top = int(N *${TOP_RATIO})/g" data_split.py
    sed -i -E "s/n_bottom = int\(N \* ?[0-9.]+\)/n_bottom = int(N *${BOTTOM_RATIO})/g" data_split.py
    python data_split.py

    # ================= 2. Feature Generation =================
    echo ">>> [2/4] Entering ./feature_generation directory, extracting features"
    cd "$ROOT_DIR"/feature_generation
    python extract_prott5.py
    python extract_AAindex_onehot_georgiev.py
    python combine_features.py

    # ================= 3. Model Prediction & Evaluation =================
    echo ">>> [3/4] Returning to root directory, running predictions and evaluations"
    cd "$ROOT_DIR"
    
    # Run the updated prediction and benchmark evaluation script
    python benchmark_models_evaluation.py

    # ================= 4. Dynamic File Renaming =================
    echo ">>> [4/4] Renaming output files based on the current seed..."
    PREFIX="seed${SEED}-${TOP_RATIO}-${BOTTOM_RATIO}"

    # Iterate over generated result files and prepend the prefix to prevent overwriting
    for FILE in "benchmark_results.csv" "plot_data_all_predictions_raw.csv" "plot_data_all_models_enrichment.csv" "test_set_with_best_predictions.csv"; do
        if [ -f "${OUTPUT_DIR}/${FILE}" ]; then
            mv "${OUTPUT_DIR}/${FILE}" "${OUTPUT_DIR}/${PREFIX}-${FILE}"
            echo "✅ Successfully renamed: ${OUTPUT_DIR}/${PREFIX}-${FILE}"
        else
            echo "⚠️ Notice: ${FILE} not found. Skipping renaming."
        fi
    done

    echo "🎉 Task completed for seed: ${SEED}!"
    echo "-------------------------------------------------"

done

echo "================================================="
echo "🔥 All random seed experiments completed. Initiating aggregation of statistics..."
echo "================================================="

# Run the aggregated seed merge script
python merge_aggregated_results.py
