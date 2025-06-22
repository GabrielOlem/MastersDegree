# Masters Degree

```plaintext
financial-llm-reasoning/
│
├── data/                        # Dados brutos e processados
│   ├── raw/                     # Dataset original (DocFinQA)
│   ├── processed/               # Dados processados (com golden chunks)
│   ├── augmented/               # Dados com CoT, código, student-teacher
│   └── outputs/                 # Resultados gerados (respostas, logs, etc.)
│
├── notebooks/                   # Jupyter Notebooks para experimentação
│   ├── 01_exploration.ipynb     # Análise exploratória do dataset
│   ├── 02_chunking.ipynb        # Criação dos golden chunks
│   ├── 03_generation.ipynb      # Geração de dados student-teacher
│   ├── 04_training.ipynb        # Fine-tuning dos modelos
│   └── 05_evaluation.ipynb      # Avaliação dos modelos
│
├── src/                         # Código-fonte (pip install -e . opcional)
│   ├── data/                    # Pipeline de dados
│   │   ├── load_dataset.py      # Carregar dados brutos
│   │   ├── chunking.py          # Gerar chunks e golden chunks
│   │   ├── retrieval.py         # Implementação de BM25, FAISS, etc.
│   │   ├── generation.py        # Geração student-teacher (CoT, código)
│   │   └── utils.py             # Funções auxiliares
│   │
│   ├── models/                  # Modelagem
│   │   ├── finetune.py          # Script de fine-tuning (LoRA, QLoRA)
│   │   ├── inference.py         # Pipeline de inferência
│   │   └── evaluation.py        # Avaliação (metrics, plots)
│   │
│   └── config/                  # Configurações gerais
│       └── config.yaml          # Parâmetros, paths, hyperparams
│
├── scripts/                     # Scripts CLI para rodar pipeline
│   ├── run_chunking.py          # Gerar golden chunks
│   ├── run_generation.py        # Gerar dados com raciocínio (teacher)
│   ├── run_training.py          # Fine-tuning
│   └── run_evaluation.py        # Avaliação
│
├── outputs/                     # Logs, checkpoints, gráficos, tabelas
│   ├── logs/                    # Logs de execução
│   ├── checkpoints/             # Pesos dos modelos
│   └── reports/                 # Relatórios, gráficos, tabelas finais
│
├── requirements.txt             # Dependências
├── environment.yml              # (opcional) Ambiente Conda
├── README.md                    # Descrição do projeto
└── setup.py                     # (opcional) Transformar em package Python
```