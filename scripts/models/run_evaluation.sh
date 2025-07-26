#!/bin/bash
#SBATCH --job-name=evaluation_job
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH --gpus=1
#SBATCH --output=evaluation_output.txt
#SBATCH --error=evaluation_error.txt

# Load Python module
module load Python3.10

# Activate virtual environment
source venv/bin/activate
pip install -r requirements.txt

# Run evaluation script
python -m src.models.evaluation --model_path "microsoft/phi-2" --dataset_path "data/test_golden_chunks.json" --output_path "results/test_cru_phi_2.json" --prompt_path "prompts/python_generation.txt"
python -m src.models.evaluation --model_path "deepseek-ai/deepseek-coder-6.7b-instruct" --dataset_path "data/test_golden_chunks.json" --output_path "results/test_cru_deepseek_coder_6_7b_instruct.json" --prompt_path "prompts/python_generation.txt"
python -m src.models.evaluation --model_path "mistralai/Mistral-7B-Instruct-v0.3" --dataset_path "data/test_golden_chunks.json" --output_path "results/test_cru_Mistral_7B_Instruct_v0_3.json" --prompt_path "prompts/python_generation.txt"
python -m src.models.evaluation --model_path "Qwen/Qwen3-8B" --dataset_path "data/test_golden_chunks.json" --output_path "results/test_cru_Qwen3_8B.json" --prompt_path "prompts/python_generation.txt"
