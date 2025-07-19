#!/bin/bash
#SBATCH --job-name=chunking_job
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --mem 16G
#SBATCH -c 8
#SBATCH -o job.log
#SBATCH --gpus=1
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt

#carregar versão python
module load Python/3.10.8
#criar ambiente
#python -m venv $HOME/env_teste
#ativar ambiente
source venv/bin/activate
#instalar pacotes desejados
# pip install -r requirements.txt
#executar .py
python -m scripts.run_embed_to_qdrant --chunks_path data/train_chunks.json