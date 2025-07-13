#!/bin/bash
#SBATCH --job-name=chunking_job
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --mem 16G
#SBATCH -c 8
#SBATCH -o job.log
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt

#carregar versão python
module load Python/3.10.8
#criar ambiente
python -m venv $HOME/env_teste
#ativar ambiente
source $HOME/env_teste/bin/activate
#instalar pacotes desejados
pip install -r requirements.txt
#executar .py
python $HOME/mestrado/MastersDegree/scripts/run_chunking.py --input_path $HOME/mestrado/MastersDegree/data/dev.json --output_path $HOME/mestrado/MastersDegree/data/dev_chunks.json