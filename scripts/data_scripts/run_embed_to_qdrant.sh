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
module load Python3.10
#criar ambiente
#python -m venv $HOME/env_teste
#ativar ambiente
source venv/bin/activate
#instalar pacotes desejados
# pip install -r requirements.txt
#executar .py
python -m src.data_scripts.embed_chunkings_to_qdrant --chunks_path data/train_chunks.json