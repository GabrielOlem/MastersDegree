import argparse
from src.data_scripts.generation import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rationale/code generation pipeline.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSON file")
    args = parser.parse_args()
    main(args.input_path, args.output_path)