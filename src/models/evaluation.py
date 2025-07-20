import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel, PeftConfig
import time

MAX_TOKENS = 512

def load_model(model_path):
    try:
        config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto")
        model = PeftModel.from_pretrained(base_model, model_path)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Ensure left padding for decoder-only models
    return model, tokenizer

def load_prompt_template(prompt_path):
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

def generate_code(pipe, prompt_template, question, chunk):
    compiled_prompt = prompt_template.format(question=question, chunk=chunk)
    output = pipe(compiled_prompt, max_new_tokens=200, do_sample=False, return_full_text=False)
    return output[0]["generated_text"].strip()

def answer_exact_match(pred, gold):
    return str(pred).strip() == str(gold).strip()

def program_exact_match(pred, gold):
    return str(pred).strip() == str(gold).strip()   

def safe_exec(code: str):
    local_vars = {}
    try:
        exec(code, {}, local_vars)
        return local_vars.get("answer", None)
    except Exception:
        return None
        
def execution_accuracy(pred_answer, gold_answer, float_tol: float = 1e-3) -> bool:
    if pred_answer is None or gold_answer is None:
        return False
    if isinstance(pred_answer, (int, float)) and isinstance(gold_answer, (int, float)):
        return abs(pred_answer - gold_answer) < float_tol
    else:
        return str(pred_answer).strip() == str(gold_answer).strip()

def main(model_path, dataset_path, output_path, prompt_path):
    # Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from dataset: {dataset_path}")

    # Load model and prompt
    model, tokenizer = load_model(model_path)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    prompt_template = load_prompt_template(prompt_path)

    # Prepare all prompts
    prompts = []
    meta = []
    for item in data[:10]:
        question = item["question"]
        chunk = item.get("golden_chunk") or item.get("chunk") or ""
        target_code = item.get("program", "")
        answer = item.get("answer", "")
        prompts.append(prompt_template.format(question=question, chunk=chunk))
        meta.append({
            "question": question,
            "golden_chunk": chunk,
            "answer": answer,
            "golden_program_generated": target_code
        })

    # Generate outputs in batch
    start_time = time.time()
    outputs = pipe(prompts, batch_size=8, max_new_tokens=1024, do_sample=False, return_full_text=False)
    total_latency = time.time() - start_time

    results = []
    for item_meta, output in zip(meta, outputs):
        generated = output[0]["generated_text"].strip()
        target_code = item_meta["golden_program_generated"]
        answer = item_meta["answer"]

        answer_exec = safe_exec(target_code)
        generated_exec = safe_exec(generated)

        # Metrics
        exec_acc = execution_accuracy(generated_exec, answer_exec)
        ans_em = answer_exact_match(answer_exec, answer)
        prog_em = program_exact_match(generated, target_code)

        results.append({
            **item_meta,
            "generated_program": generated,
            "latency": total_latency / len(outputs),  # Approximate per-sample latency
            "execution_accuracy": exec_acc,
            "answer_exact_match": ans_em,
            "program_exact_match": prog_em
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(results)} generations to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model (base or LoRA fine-tuned)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save generations")
    parser.add_argument("--prompt_path", type=str, required=True, help="Path to prompt template file")
    args = parser.parse_args()
    main(args.model_path, args.dataset_path, args.output_path, args.prompt_path)