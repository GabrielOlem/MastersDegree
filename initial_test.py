# 📄 Notebook Inicial — Experimento DocFinQA

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# 🚩 Configurações
model_id = "mistralai/Mistral-7B-Instruct-v0.2"  # ➕ Pode trocar para DeepSeek, Llama 3, etc.
device = "cuda" if torch.cuda.is_available() else "cpu"

# 🔥 Carregando modelo e tokenizer
print("🔄 Carregando modelo...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print("✅ Modelo carregado!")

# 📑 Exemplo de input manual (Golden Chunk)
contexto = """
A empresa XYZ reportou no seu balanço anual um total de receitas de $1,200,000 e despesas operacionais de $950,000.
O lucro líquido foi impactado por impostos no valor de $50,000.
"""

pergunta = "Qual foi o lucro líquido da empresa?"

# 🔗 Montando o prompt
prompt = f"""
Você é um assistente financeiro. Dado o documento abaixo e a pergunta, responda de forma objetiva.

Documento:
{contexto}

Pergunta:
{pergunta}

Responda apenas com o valor numérico correto.
"""

# 🚀 Gerando resposta
output = pipe(prompt, max_new_tokens=100, temperature=0)
resposta = output[0]['generated_text'].split("Pergunta:")[-1]

print("🧠 Resposta do modelo:")
print(resposta)