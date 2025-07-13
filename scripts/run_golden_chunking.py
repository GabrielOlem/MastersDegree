import argparse
from src.data.golden_chunking import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run golden chunking pipeline.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input chunked JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output golden JSON file")
    args = parser.parse_args()
    main(args.input_path, args.output_path)