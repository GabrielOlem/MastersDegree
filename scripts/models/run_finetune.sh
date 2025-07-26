#!/bin/bash
#SBATCH --job-name=finetune_job
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH -c 16
#SBATCH --gpus=2
#SBATCH --output=logs/finetune_cru_output.txt
#SBATCH --error=logs/finetune_cru_error.txt

# Load Python module
module load Python3.10

# Activate virtual environment
source venv/bin/activate
pip install -r requirements.txt

# Run finetune script
python -m src.models.finetune --model "microsoft/phi-2" --input_path "data/test_golden_chunks.json" --output_dir "models/phi_2_finetuned_test_cru" --prompt_path "prompts/python_generation.txt"
python -m src.models.finetune --model "deepseek-ai/deepseek-coder-6.7b-instruct" --input_path "data/test_golden_chunks.json" --output_dir "models/deepseek_coder_6_7b_instruct_finetuned_test_cru" --prompt_path "prompts/python_generation.txt"
python -m src.models.finetune --model "mistralai/Mistral-7B-Instruct-v0.3" --input_path "data/test_golden_chunks.json" --output_dir "models/Mistral_7B_Instruct_v0_3_finetuned_test_cru" --prompt_path "prompts/python_generation.txt"
python -m src.models.finetune --model "Qwen/Qwen3-8B" --input_path "data/test_golden_chunks.json" --output_dir "models/Qwen3_8B_finetuned_test_cru" --prompt_path "prompts/python_generation.txt"
python -m src.models.evaluation --model_path "models/phi_2_finetuned_test_cru" --dataset_path "data/test_golden_chunks.json" --output_path "results/phi_2_finetuned_test_data.json" --prompt_path "prompts/python_generation.txt"
python -m src.models.evaluation --model_path "models/deepseek_coder_6_7b_instruct_finetuned_test_cru" --dataset_path "data/test_golden_chunks.json" --output_path "results/deepseek_coder_6.7b_instruct_finetuned_test_data.json" --prompt_path "prompts/python_generation.txt"
python -m src.models.evaluation --model_path "models/Mistral_7B_Instruct_v0_3_finetuned_test_cru" --dataset_path "data/test_golden_chunks.json" --output_path "results/Mistral_7B_Instruct_v0.3_finetuned_test_data.json" --prompt_path "prompts/python_generation.txt"
python -m src.models.evaluation --model_path "models/Qwen3_8B_finetuned_test_cru" --dataset_path "data/test_golden_chunks.json" --output_path "results/Qwen3_8B_finetuned_test_data.json" --prompt_path "prompts/python_generation.txt"
