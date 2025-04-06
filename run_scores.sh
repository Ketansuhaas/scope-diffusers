#!/bin/bash -l

# ------------------------------------------------------------------------------
# SCC directives
# ------------------------------------------------------------------------------
#$ -P vkolagrp
#$ -t 1-1
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l h_rt=24:00:00
#$ -N de_casteljau_scores
#$ -j y
#$ -o /projectnb/vkolagrp/ketanss/scope-diffusers/qsub_runs
#$ -m ea


MODEL_NAME="stabilityai/stable-diffusion-2-1"
NUM_INFERENCE_STEPS=50
SEED=42
INTERPOLATION_METHOD="spherical_de_casteljau"
CSV_PATH="/projectnb/vkolagrp/ketanss/scope-diffusers/genai_dataset_schedules_fixed.csv"
# HF_CACHE_DIR="/projectnb/vkolagrp/ketanss/scope-diffusers-ketan/sdpcache"
EXP_DIR="exp_dump/sdc"

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="

module load miniconda
conda activate t2v
python get_scores.py \
    --model_name "$MODEL_NAME" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --seed "$SEED" \
    --interpolation_method "$INTERPOLATION_METHOD" \
    --csv_path "$CSV_PATH" \
    --exp_dir "$EXP_DIR" \