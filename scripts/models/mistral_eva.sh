#!/bin/bash
#SBATCH --job-name=eval_mistral
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH --gpus=1
#SBATCH --output=logs/eval_mistral_output.txt
#SBATCH --error=logs/eval_mistral_error.txt

module load Python3.10
source venv/bin/activate
pip install -r requirements.txt
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

python -m src.models.evaluation --model_path "mistralai/Mistral-7B-Instruct-v0.3" --dataset_path "data/test_golden_chunks.json" --output_path "results/test_cru_Mistral_7B_Instruct_v0_3.json" --prompt_path "prompts/python_generation.txt"