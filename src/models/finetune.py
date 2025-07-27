import argparse
import json

from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

MAX_LENGTH = 512

LORA_TARGET_MODULES = {
    "microsoft/phi-2": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    "deepseek-ai/deepseek-coder-6.7b-instruct": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    "mistralai/Mistral-7B-Instruct-v0.3": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "Qwen/Qwen3-8B": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def get_lora_target_modules(model_name):
    for key in LORA_TARGET_MODULES:
        if key in model_name:
            return LORA_TARGET_MODULES[key]
    # Default fallback (may need adjustment)
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def load_dataset(json_path, prompt_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()
    samples = []
    for item in data:
        context = item["golden_chunk"]
        question = item["question"]
        try:
            program = item["golden_program_generated"].strip()
        except KeyError:
            program = item["program"].strip() if "program" in item else None
        # skip if program is missing
        if not program:
            continue
        compiled_prompt = prompt.format(question=question, chunk=context)
        label = f"{compiled_prompt}\n{program}"
        samples.append({"text": label})
    return Dataset.from_list(samples)


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH
    )


def main(input_path, output_dir, model_name, prompt_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Load model in 4bit + prepare for LoRA
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quant_config, device_map="auto"
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=get_lora_target_modules(model_name),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(input_path, prompt_path)
    train_dataset, val_dataset = dataset.train_test_split(test_size=0.1).values()
    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    tokenized_val = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    # tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     per_device_train_batch_size=4,
    #     gradient_accumulation_steps=4,
    #     num_train_epochs=3,
    #     logging_steps=20,
    #     save_strategy="epoch",
    #     eval_strategy="no",
    #     report_to="none",
    #     fp16=True,
    #     save_total_limit=1
    # )
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=100,  # Set high, let early stopping decide
        logging_steps=20,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=100,  # Evaluate every 100 steps
        save_steps=100,
        report_to="none",
        fp16=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_dataset,
    #     processing_class=tokenizer,
    #     data_collator=data_collator
    # )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune small LLM to generate reasoning code from financial context."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to JSON with golden_program_generated",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save fine-tuned model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Base model to fine-tune",
    )
    parser.add_argument("--prompt_path", type=str, required=True, help="Path to prompt")
    args = parser.parse_args()
    main(args.input_path, args.output_dir, args.model, args.prompt_path)
