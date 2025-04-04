#!/bin/bash -l

# ------------------------------------------------------------------------------
# SCC directives
# ------------------------------------------------------------------------------
#$ -P vkolagrp
#$ -t 1-1
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l h_rt=44:00:00
#$ -N SD2.1_spherical_de_casteljau
#$ -j y
#$ -o /projectnb/vkolagrp/ketanss/scope-diffusers/qsub_runs
#$ -m ea

# ------------------------------------------------------------------------------
# Conda environment
# ------------------------------------------------------------------------------
module load miniconda
conda activate caueeg

# ------------------------------------------------------------------------------
# Config (edit these values only)
# ------------------------------------------------------------------------------
MODEL_NAME="stabilityai/stable-diffusion-2-1"
NUM_INFERENCE_STEPS=50
SEED=42
INTERPOLATION_METHOD="spherical_de_casteljau"
CSV_PATH="/projectnb/vkolagrp/ketanss/scope-diffusers/genai_dataset_schedules_fixed.csv"
HF_CACHE_DIR="/projectnb/vkolagrp/ketanss/scope-diffusers-ketan/sdpcache"
EXP_DIR="exp_dump/sdc"
# ------------------------------------------------------------------------------
# Print config
# ------------------------------------------------------------------------------
echo "=== Running SCoPE Pipeline + CLIP Scoring ==="
echo "MODEL_NAME           = $MODEL_NAME"
echo "NUM_INFERENCE_STEPS  = $NUM_INFERENCE_STEPS"
echo "SEED                 = $SEED"
echo "INTERPOLATION_METHOD = $INTERPOLATION_METHOD"
echo "CSV_PATH             = $CSV_PATH"
echo "HF_CACHE_DIR         = $HF_CACHE_DIR"
echo "EXP_DIR              = $EXP_DIR"
echo "============================================"
echo

# ------------------------------------------------------------------------------
# 1) Run main.py to generate images
# # ------------------------------------------------------------------------------
python main.py \
    --model_name "$MODEL_NAME" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --seed "$SEED" \
    --interpolation_method "$INTERPOLATION_METHOD" \
    --csv_path "$CSV_PATH" \
    --hf_cache_dir "$HF_CACHE_DIR" \
    --exp_dir "$EXP_DIR"\
    # --use_refiner

# conda activate t2v
# # ------------------------------------------------------------------------------
# # 2) Run get_scores.py to get all scores
# # ------------------------------------------------------------------------------
# python get_scores.py \
#     --model_name "$MODEL_NAME" \
#     --num_inference_steps "$NUM_INFERENCE_STEPS" \
#     --seed "$SEED" \
#     --interpolation_method "$INTERPOLATION_METHOD" \
#     --csv_path "$CSV_PATH" \
#     --exp_dir "$EXP_DIR" \