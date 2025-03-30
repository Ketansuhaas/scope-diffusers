#!/bin/bash -l

# ------------------------------------------------------------------------------
# SCC directives
# ------------------------------------------------------------------------------
#$ -P ivc-ml
#$ -t 1-1
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=11:00:00
#$ -N stats_geode
#$ -j y
#$ -o /projectnb/ivc-ml/xthomas/THESIS/MS_Thesis/feature_analysis/qsub_runs

# ------------------------------------------------------------------------------
# Conda environment
# ------------------------------------------------------------------------------
module load miniconda
conda activate diffusion_features

# ------------------------------------------------------------------------------
# Config (edit these values only)
# ------------------------------------------------------------------------------
MODEL_NAME="stabilityai/stable-diffusion-2-1"
NUM_INFERENCE_STEPS=50
SEED=42
INTERPOLATION_METHOD="stagewise_switcher"
CSV_PATH="/projectnb/ivc-ml/xthomas/cs791/scope-diffusers/genai_dataset_schedules_fixed.csv"

# ------------------------------------------------------------------------------
# Print config
# ------------------------------------------------------------------------------
echo "=== Running SCoPE Pipeline + CLIP Scoring ==="
echo "MODEL_NAME           = $MODEL_NAME"
echo "NUM_INFERENCE_STEPS  = $NUM_INFERENCE_STEPS"
echo "SEED                 = $SEED"
echo "INTERPOLATION_METHOD = $INTERPOLATION_METHOD"
echo "CSV_PATH             = $CSV_PATH"
echo "============================================"
echo

# ------------------------------------------------------------------------------
# 1) Run main.py to generate images
# # ------------------------------------------------------------------------------
# python main.py \
#     --model_name "$MODEL_NAME" \
#     --num_inference_steps "$NUM_INFERENCE_STEPS" \
#     --seed "$SEED" \
#     --interpolation_method "$INTERPOLATION_METHOD" \
#     --csv_path "$CSV_PATH"

conda activate t2v
# ------------------------------------------------------------------------------
# 2) Run get_scores.py to get all scores
# ------------------------------------------------------------------------------
python get_scores.py \
    --model_name "$MODEL_NAME" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --seed "$SEED" \
    --interpolation_method "$INTERPOLATION_METHOD" \
    --csv_path "$CSV_PATH" \