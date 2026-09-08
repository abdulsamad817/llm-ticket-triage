"""
Fine-tunes the base model with QLoRA: LoRA adapters on top of a 4-bit quantized
base model. Same task and data as train_lora.py — the point of this script
existing separately is to produce a clean, comparable memory/speed/accuracy
data point against the full-precision LoRA run.

Usage:
    python -m src.training.train_qlora --config configs/train_config.yaml
"""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.data.schema import SchemaValidator
from src.training.train_lora import build_training_text, load_config, tokenize_function


def main(config_path: str) -> None:
    config = load_config(config_path)
    model_name = config["model"]["base_model"]
    validator = SchemaValidator(config["schema"])
    qlora_cfg = config["qlora"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qlora_cfg["load_in_4bit"],
        bnb_4bit_quant_type=qlora_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, qlora_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=qlora_cfg["bnb_4bit_use_double_quant"],
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

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
        output_dir=f"{training_cfg['output_dir']}/qlora",
        num_train_epochs=training_cfg["num_train_epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        warmup_ratio=training_cfg["warmup_ratio"],
        logging_steps=training_cfg["logging_steps"],
        save_strategy=training_cfg["save_strategy"],
        eval_strategy=training_cfg["eval_strategy"],
        report_to=training_cfg["report_to"],
        run_name=f"{training_cfg['run_name']}-qlora",
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    model.save_pretrained(f"{training_cfg['output_dir']}/qlora/final_adapter")
    tokenizer.save_pretrained(f"{training_cfg['output_dir']}/qlora/final_adapter")
    print(f"QLoRA adapter saved to {training_cfg['output_dir']}/qlora/final_adapter")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    main(args.config)