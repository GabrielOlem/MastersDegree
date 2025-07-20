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

# Run evaluation script
python -m scripts.run_evaluation --model_path "microsoft/phi-2" --dataset_path "data/test_golden_chunks.json" --output_path "results/test_phi_2.json" --prompt_path "prompts/python_generation.txt"
