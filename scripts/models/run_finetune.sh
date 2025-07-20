#!/bin/bash
#SBATCH --job-name=finetune_job
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH -c 16
#SBATCH --gpus=2
#SBATCH --output=finetune_output.txt
#SBATCH --error=finetune_error.txt

# Load Python module
module load Python3.10

# Activate virtual environment
source venv/bin/activate

# Run finetune script
python -m scripts.models.run_finetune --model "microsoft/phi-2" --input_path "data/test_golden_chunks.json" --output_dir "results/test_phi_2.json" --prompt_path "prompts/python_generation.txt"
