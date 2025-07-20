import argparse
from src.models.finetune import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune small LLM to generate reasoning code from financial context.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to JSON with golden_program_generated")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save fine-tuned model")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Base model to fine-tune")
    parser.add_argument("--prompt_path", type=str, required=True, help="Path to prompt")
    args = parser.parse_args()
    main(args.input_path, args.output_dir, args.model, args.prompt_path)