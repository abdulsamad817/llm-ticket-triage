"""
Loads the fine-tuned model once and exposes a single predict() function used
by the API layer. Keeping this separate from api.py means it's independently
testable and independently reusable (e.g. from a batch script).
"""

from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.schema import SchemaValidator


class TriageModel:
    def __init__(self, config_path: str = "configs/train_config.yaml", adapter_path: str | None = None):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.validator = SchemaValidator(self.config["schema"])
        model_name = self.config["model"]["base_model"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

        default_adapter = Path(self.config["training"]["output_dir"]) / "lora" / "final_adapter"
        adapter_path = adapter_path or str(default_adapter)

        if Path(adapter_path).exists():
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            # Falls back to the un-fine-tuned base model so the API is still runnable
            # before training has happened, e.g. during local development.
            self.model = base_model

        self.model.eval()

    def predict(self, message: str) -> dict:
        prompt = self.validator.build_prompt(message)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config["evaluation"]["generation_max_new_tokens"],
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        raw_output = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        parsed = self.validator.parse(raw_output)

        if parsed is None:
            return {"error": "Model did not return valid JSON.", "raw_output": raw_output}

        if not self.validator.is_valid(parsed):
            return {"error": "Model returned JSON with unexpected fields/values.", "parsed": parsed}

        return parsed