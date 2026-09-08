"""
Evaluates zero-shot, few-shot, LoRA, and QLoRA on the same held-out test set
using the same metrics, so the results are genuinely comparable.

Usage:
    python -m src.evaluation.evaluate --config configs/train_config.yaml
"""

import argparse
import json
import time
from pathlib import Path

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.data.schema import SchemaValidator, TicketLabel


FEW_SHOT_EXAMPLES = [
    {
        "message": "My package was supposed to arrive 3 days ago and tracking hasn't updated at all",
        "label": {"intent": "shipping_delay", "sentiment": "negative", "urgency": "medium",
                   "category": "shipping", "summary": "Package late, tracking not updating"},
    },
    {
        "message": "Just wanted to say the new update is great, keep up the good work!",
        "label": {"intent": "general_inquiry", "sentiment": "positive", "urgency": "low",
                   "category": "other", "summary": "Positive feedback on recent update"},
    },
    {
        "message": "I can't log into my account, it keeps saying my password is wrong even after I reset it",
        "label": {"intent": "account_access", "sentiment": "negative", "urgency": "high",
                   "category": "account", "summary": "Locked out of account after password reset"},
    },
]


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_test_set(processed_dir: str) -> list[dict]:
    records = []
    with open(Path(processed_dir) / "test.jsonl", "r") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def build_few_shot_prompt(validator: SchemaValidator, message: str) -> str:
    base_prompt = validator.build_prompt(message)
    examples_text = "\n\n".join(
        f"Customer message: {ex['message']}\nJSON output: {json.dumps(ex['label'], ensure_ascii=False)}"
        for ex in FEW_SHOT_EXAMPLES
    )
    return f"Here are some examples:\n\n{examples_text}\n\n{base_prompt}"


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.time() - start
    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated, latency


def score_predictions(predictions: list[dict | None], gold_records: list[dict], validator: SchemaValidator) -> dict:
    valid_count = 0
    exact_match_count = 0
    field_correct = {field: 0 for field in ["intent", "sentiment", "urgency", "category"]}
    total = len(predictions)

    for pred, gold in zip(predictions, gold_records):
        if pred is None or not validator.is_valid(pred):
            continue
        valid_count += 1
        all_correct = True
        for field in field_correct:
            if pred.get(field) == gold.get(field):
                field_correct[field] += 1
            else:
                all_correct = False
        if all_correct:
            exact_match_count += 1

    field_f1 = {field: field_correct[field] / total for field, count in field_correct.items()}
    avg_field_f1 = sum(field_f1.values()) / len(field_f1)

    return {
        "json_valid_pct": round(100 * valid_count / total, 1),
        "avg_field_f1": round(avg_field_f1, 3),
        "exact_match_pct": round(100 * exact_match_count / total, 1),
        "per_field_f1": {k: round(v, 3) for k, v in field_f1.items()},
    }


def evaluate_method(method: str, model, tokenizer, validator: SchemaValidator,
                     test_records: list[dict], config: dict) -> dict:
    predictions = []
    latencies = []

    for record in test_records:
        if method == "few_shot":
            prompt = build_few_shot_prompt(validator, record["message"])
        else:
            prompt = validator.build_prompt(record["message"])

        raw_output, latency = generate(
            model, tokenizer, prompt, config["evaluation"]["generation_max_new_tokens"]
        )
        latencies.append(latency)
        predictions.append(validator.parse(raw_output))

    metrics = score_predictions(predictions, test_records, validator)
    metrics["avg_latency_sec"] = round(sum(latencies) / len(latencies), 3)
    return metrics


def main(config_path: str) -> None:
    config = load_config(config_path)
    validator = SchemaValidator(config["schema"])
    test_records = load_test_set(config["data"]["processed_dir"])
    model_name = config["model"]["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    for method in config["evaluation"]["methods"]:
        print(f"\nEvaluating method: {method}")

        if method in ("zero_shot", "few_shot"):
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        elif method == "lora":
            base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
            model = PeftModel.from_pretrained(base, f"{config['training']['output_dir']}/lora/final_adapter")
        elif method == "qlora":
            base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
            model = PeftModel.from_pretrained(base, f"{config['training']['output_dir']}/qlora/final_adapter")
        else:
            raise ValueError(f"Unknown method: {method}")

        model.eval()
        metrics = evaluate_method(method, model, tokenizer, validator, test_records, config)
        results[method] = metrics
        print(json.dumps(metrics, indent=2))

        del model
        torch.cuda.empty_cache()

    output_path = Path(config["training"]["output_dir"]) / "comparison_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull comparison written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    main(args.config)