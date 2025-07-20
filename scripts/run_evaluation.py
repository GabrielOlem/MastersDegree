import argparse
from src.models.evaluation import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation pipeline.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model (base or LoRA fine-tuned)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save generations")
    parser.add_argument("--prompt_path", type=str, required=True, help="Path to prompt template file")
    args = parser.parse_args()
    main(args.model_path, args.dataset_path, args.output_path, args.prompt_path)