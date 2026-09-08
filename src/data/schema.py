import json
from dataclasses import dataclass


@dataclass
class TicketLabel:
    intent: str
    sentiment: str
    urgency: str
    category: str
    summary: str

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "sentiment": self.sentiment,
            "urgency": self.urgency,
            "category": self.category,
            "summary": self.summary,
        }

    def to_json_str(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class SchemaValidator:
    """Validates model outputs against the allowed vocab for each field."""

    def __init__(self, schema_config: dict):
        self.fields = schema_config["fields"]
        self.allowed_values = {
            "intent": set(schema_config["intents"]),
            "sentiment": set(schema_config["sentiments"]),
            "urgency": set(schema_config["urgencies"]),
            "category": set(schema_config["categories"]),
        }

    def parse(self, raw_output: str) -> dict | None:
        """
        Attempts to parse a raw model output string into a dict.
        Returns None if it isn't valid JSON — callers should treat that as a
        JSON-validity failure, not silently coerce it into something else.
        """
        raw_output = raw_output.strip()

        if raw_output.startswith("```"):
            raw_output = raw_output.strip("`")
            if raw_output.lower().startswith("json"):
                raw_output = raw_output[4:].strip()
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    def is_valid(self, parsed: dict) -> bool:
        """Checks all required fields are present and categorical fields use allowed values."""
        if not all(field in parsed for field in self.fields):
            return False
        for field, allowed in self.allowed_values.items():
            if parsed.get(field) not in allowed:
                return False
        return True

    def build_prompt(self, message: str) -> str:
        """The instruction prompt used consistently across zero-shot, few-shot, and fine-tuned inference."""
        return (
            "You are a support ticket triage assistant. Given a customer message, "
            "output ONLY a JSON object with exactly these fields: "
            f"{', '.join(self.fields)}.\n"
            f"Allowed intent values: {', '.join(self.allowed_values['intent'])}\n"
            f"Allowed sentiment values: {', '.join(self.allowed_values['sentiment'])}\n"
            f"Allowed urgency values: {', '.join(self.allowed_values['urgency'])}\n"
            f"Allowed category values: {', '.join(self.allowed_values['category'])}\n\n"
            f"Customer message: {message}\n\n"
            "JSON output:"
        )