#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml
#$ -t 1-1  # Array job specification
#$ -pe omp 4 # Request 4 CPU cores
#$ -l gpus=1 # Request 1 GPU
#$ -l gpu_c=7.0
#$ -l h_rt=11:00:00
#$ -N stats_geode
#$ -j y # Merge standard output and error
#$ -o /projectnb/ivc-ml/xthomas/THESIS/MS_Thesis/feature_analysis/qsub_runs

module load miniconda
conda activate diffusion_features

python main.py --model_name stabilityai/stable-diffusion-2-1 --num_inference_steps 50 --seed 42 --interpolation_method nlerp_og

python get_clip_scores.py --model_name "stabilityai/stable-diffusion-2-1" --num_inference_steps 50 --seed 42 --interpolation_method "nlerp_og" \
    --exp_dir "exp_dump/eval_output" \
    --csv_path "/projectnb/ivc-ml/xthomas/cs791/scope-diffusers/genai_dataset_schedules_fixed.csv" \
    --output_json "my_clip_scores.json"