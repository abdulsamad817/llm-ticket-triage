"""
Fine-tunes the base model with LoRA (standard precision) on the ticket triage task.

Usage:
    python -m src.training.train_lora --config configs/train_config.yaml
"""

import argparse

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.data.schema import SchemaValidator, TicketLabel


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_training_text(example: dict, validator: SchemaValidator) -> dict:
    """Formats a raw record into a single instruction-tuning text sequence."""
    prompt = validator.build_prompt(example["message"])
    label = TicketLabel(
        intent=example["intent"],
        sentiment=example["sentiment"],
        urgency=example["urgency"],
        category=example["category"],
        summary=example["summary"],
    )
    full_text = prompt + " " + label.to_json_str()
    return {"text": full_text}


def tokenize_function(examples, tokenizer, max_length):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main(config_path: str) -> None:
    config = load_config(config_path)
    model_name = config["model"]["base_model"]
    validator = SchemaValidator(config["schema"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    processed_dir = config["data"]["processed_dir"]
    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{processed_dir}/train.jsonl",
            "validation": f"{processed_dir}/val.jsonl",
        },
    )

    dataset = dataset.map(lambda ex: build_training_text(ex, validator))
    dataset = dataset.map(
        lambda ex: tokenize_function(ex, tokenizer, config["model"]["max_seq_length"]),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    training_cfg = config["training"]
    training_args = TrainingArguments(
        output_dir=f"{training_cfg['output_dir']}/lora",
        num_train_epochs=training_cfg["num_train_epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        warmup_ratio=training_cfg["warmup_ratio"],
        logging_steps=training_cfg["logging_steps"],
        save_strategy=training_cfg["save_strategy"],
        eval_strategy=training_cfg["eval_strategy"],
        report_to=training_cfg["report_to"],
        run_name=f"{training_cfg['run_name']}-lora",
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    model.save_pretrained(f"{training_cfg['output_dir']}/lora/final_adapter")
    tokenizer.save_pretrained(f"{training_cfg['output_dir']}/lora/final_adapter")
    print(f"LoRA adapter saved to {training_cfg['output_dir']}/lora/final_adapter")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    main(args.config)