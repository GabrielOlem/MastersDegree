#!/bin/bash
#SBATCH --job-name=finetune_job
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH -c 16
#SBATCH --gpus=2
#SBATCH --output=logs/finetune_output.txt
#SBATCH --error=logs/finetune_error.txt

# Load Python module
module load Python3.10

# Activate virtual environment
source venv/bin/activate

# Run finetune script
python -m src.models.finetune --model "microsoft/phi-2" --input_path "data/test_golden_chunks.json" --output_dir "models/phi_2_finetuned_test_data.json" --prompt_path "prompts/python_generation.txt"
python -m src.models.finetune --model "deepseek-ai/deepseek-coder-6.7b-instruct" --input_path "data/test_golden_chunks.json" --output_dir "models/deepseek_coder_6.7b_instruct_finetuned_test_data.json" --prompt_path "prompts/python_generation.txt"
python -m src.models.finetune --model "mistralai/Mistral-7B-Instruct-v0.3" --input_path "data/test_golden_chunks.json" --output_dir "models/Mistral_7B_Instruct_v0.3_finetuned_test_data.json" --prompt_path "prompts/python_generation.txt"
python -m src.models.finetune --model "Qwen/Qwen3-8B" --input_path "data/test_golden_chunks.json" --output_dir "models/Qwen3_8B_test_data.json" --prompt_path "prompts/python_generation.txt"
