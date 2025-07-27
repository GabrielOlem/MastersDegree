#!/bin/bash
#SBATCH --job-name=eval_phi2
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH --gpus=1
#SBATCH --output=logs/eval_phi2_output.txt
#SBATCH --error=logs/eval_phi2_error.txt

module load Python3.10
source venv/bin/activate
pip install -r requirements.txt
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

python -m src.models.evaluation --model_path "microsoft/phi-2" --dataset_path "data/test_golden_chunks.json" --output_path "results/test_cru_phi_2.json" --prompt_path "prompts/python_generation.txt"