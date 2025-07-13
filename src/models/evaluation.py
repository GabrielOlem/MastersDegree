import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel, PeftConfig

MAX_TOKENS = 512

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def load_model(model_path):
    try:
        config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto")
        model = PeftModel.from_pretrained(base_model, model_path)
    except:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def format_prompt(question, chunk):
    return f"Context: {chunk}\nQuestion: {question}\nGenerate Python code to solve:"

def generate_code(pipe, question, chunk):
    prompt = format_prompt(question, chunk)
    output = pipe(prompt, max_new_tokens=200, do_sample=False, return_full_text=False)
    return output[0]["generated_text"].strip()

def main(model_path, dataset_path, output_path):
    data = load_data(dataset_path)
    model, tokenizer = load_model(model_path)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

    results = []
    for item in tqdm(data, desc="Generating code"):
        question = item["question"]
        chunk = item["golden_chunk"]
        target = item.get("golden_program_generated", "")
        answer = item.get("answer", "")

        generated = generate_code(pipe, question, chunk)

        results.append({
            "question_id": item["question_id"],
            "question": question,
            "golden_chunk": chunk,
            "answer": answer,
            "golden_program_generated": target,
            "generated_program": generated
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(results)} generations to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model (base or LoRA fine-tuned)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to golden dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save generations")
    args = parser.parse_args()
    main(args.model_path, args.dataset_path, args.output_path)