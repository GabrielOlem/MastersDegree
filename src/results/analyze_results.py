import argparse
import json

import pandas as pd


def answer_exact_match(pred, gold):
    return str(pred).strip() == str(gold).strip()


def program_exact_match(pred, gold):
    return str(pred).strip() == str(gold).strip()


def safe_exec(code: str):
    if code == "":
        return None
    local_vars = {}
    try:
        clean_code = code.strip().replace("\n ", "\n")
        exec(clean_code, {}, local_vars)
        return local_vars.get("answer", None)
    except Exception:
        try:
            clean_code2 = clean_code.split("```")[1].replace("python\n", "")
            exec(clean_code2, {}, local_vars)
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


def main(input_path, output_file):
    data = pd.read_json(input_path)
    print(f"Loaded {len(data)} items from input: {input_path}")
    data["answer_exec"] = data["golden_program_generated"].apply(safe_exec)
    data["generated_exec"] = data["generated_program"].apply(safe_exec)

    data["execution_accuracy"] = data.apply(
        lambda row: execution_accuracy(row["generated_exec"], row["answer_exec"]),
        axis=1,
    )
    data["answer_exact_match"] = data.apply(
        lambda row: answer_exact_match(row["generated_exec"], row["answer"]), axis=1
    )
    data["program_exact_match"] = data.apply(
        lambda row: program_exact_match(
            row["generated_program"], row["golden_program_generated"]
        ),
        axis=1,
    )

    metrics = {
        "file": input_path,
        "execution_accuracy": data["execution_accuracy"].sum() / len(data),
        "answer_exact_match": data["answer_exact_match"].sum() / len(data),
        "program_exact_match": data["program_exact_match"].sum() / len(data),
    }
    with open(output_file, "r+") as f:
        data = json.load(f)
        data.append(metrics)
        f.seek(0)
        json.dump(data, f, indent=4)


# main("../../results/test_cru_deepseek_coder_6_7b_instruct.json")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze results from a JSON file.")
    parser.add_argument("--input_path", type=str, help="Path to the input JSON file.")
    parser.add_argument(
        "--output_file",
        type=str,
        default="../../results/metrics.json",
        help="Path to the output JSON file.",
    )
    args = parser.parse_args()

    main(args.input_path, args.output_file)
