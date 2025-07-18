import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel, PeftConfig
from langfuse import get_client
from dotenv import load_dotenv
import time

load_dotenv()
langfuse = get_client()

MAX_TOKENS = 512

    
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

def generate_code(pipe, question, chunk, item_id, run_name, model_name, prompt_label):
    prompt = langfuse.get_prompt("python_generation", label=prompt_label)
    compiled_prompt = prompt.compile(
        question=question,
        chunk=chunk,
    )
    with langfuse.start_as_current_generation(
        name="generate_code",
        input={"question": question, "chunk": chunk},
        metadata={"item_id": item_id, "run": run_name},
        model=model_name,
        prompt=prompt,
    ) as gen:
        output = [{"generated_text": "teste"}]
        #output = pipe(compiled_prompt, max_new_tokens=200, do_sample=False, return_full_text=False)
        gen.update(output={"answer": output[0]["generated_text"].strip()})
        gen.update_trace(
            input={"question": question, "chunk": chunk},
            output=output[0]["generated_text"].strip(),
        )
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
    # Compare answers
    if pred_answer is None or gold_answer is None:
        return False

    if isinstance(pred_answer, (int, float)) and isinstance(gold_answer, (int, float)):
        return abs(pred_answer - gold_answer) < float_tol
    else:
        return str(pred_answer).strip() == str(gold_answer).strip()

def main(model_path, dataset_name, output_path, run_description, prompt_label):
    dataset = langfuse.get_dataset(name=dataset_name)
    print(f"Loaded {len(dataset.items)} items from dataset: {dataset_name}")

    model, tokenizer = load_model(model_path)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

    results = []
    for item in tqdm(dataset.items[:1], desc="Generating code"):
        run_name = f"evaluation_run_{time.time()}"
        with item.run(
            run_name=run_name,
            run_description=run_description,
            run_metadata={
                "model": model_path,
            }
        ) as root_span:
            question = item.input
            chunk = item.metadata["golden_chunk"]
            target_code = item.metadata["program"]
            answer = item.expected_output

            start_time = time.time()
            generated = generate_code(
                pipe=pipe, 
                question=question, 
                chunk=chunk, 
                item_id=item.id,
                run_name=run_name,
                model_name=model_path,
                prompt_label=prompt_label
            )
            latency = time.time() - start_time

            answer_exec = safe_exec(target_code)
            generated_exec = safe_exec(generated)

            # Metrics
            exec_acc = execution_accuracy(generated_exec, answer_exec)
            ans_em = answer_exact_match(answer_exec, answer)
            prog_em = program_exact_match(generated, target_code)

            root_span.score_trace(
                name="execution_accuracy",
                value=exec_acc,
            )
            root_span.score_trace(
                name="answer_exact_match",
                value=ans_em,
            )
            root_span.score_trace(
                name="program_exact_match",
                value=prog_em,
            )
            results.append({
                "question_id": item.id,
                "question": question,
                "golden_chunk": chunk,
                "answer": answer,
                "golden_program_generated": target_code,
                "generated_program": generated,
                "latency": latency,
                "execution_accuracy": exec_acc,
                "answer_exact_match": ans_em,
                "program_exact_match": prog_em
            })
    langfuse.flush()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(results)} generations to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model (base or LoRA fine-tuned)")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of langfuse golden dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save generations")
    parser.add_argument("--run_description", type=str, default="", help="Description for the run in Langfuse")
    parser.add_argument("--prompt_label", type=str, default="production", help="Label for the prompt in Langfuse")
    args = parser.parse_args()
    main(args.model_path, args.dataset_name, args.output_path, args.run_description, args.prompt_label)