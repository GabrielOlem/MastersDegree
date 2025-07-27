#!/bin/bash
#SBATCH --job-name=finetune_phi2
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH -c 16
#SBATCH --gpus=1
#SBATCH --output=logs/finetune_phi2_output.txt
#SBATCH --error=logs/finetune_phi2_error.txt

module load Python3.10
source venv/bin/activate
pip install -r requirements.txt
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

python -m src.models.finetune --model "microsoft/phi-2" --input_path "data/test_golden_chunks.json" --output_dir "models/phi_2_finetuned_test_cru" --prompt_path "prompts/python_generation.txt"
python -m src.models.evaluation --model_path "models/phi_2_finetuned_test_cru" --dataset_path "data/test_golden_chunks.json" --output_path "results/phi_2_finetuned_test_data.json" --prompt_path "prompts/python_generation.txt"