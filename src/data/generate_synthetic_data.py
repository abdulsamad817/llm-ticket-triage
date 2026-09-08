"""
Generates a small synthetic seed dataset so the whole pipeline (prepare -> train
-> evaluate -> serve) is runnable end-to-end without waiting on real labeled data.

This is a starting point, not the final dataset — swap in real support tickets
(with PII removed) or a larger synthetic/augmented set before treating results
as meaningful. The README's "known limitations" section should say which one
you actually used.

Usage:
    python -m src.data.generate_synthetic_data
"""

import json
import random
from pathlib import Path

TEMPLATES = [
    ("billing_dispute", "negative", "high", "billing",
     ["I've been charged twice for the same order and nobody is responding",
      "There's a charge on my card I don't recognize, please explain this",
      "You charged me the wrong amount and support isn't answering my emails"]),
    ("refund_request", "neutral", "medium", "billing",
     ["Can I get a refund for the item I returned last week?",
      "I'd like to request a refund, the product doesn't match the description",
      "How do I get my money back for a canceled order?"]),
    ("technical_issue", "negative", "high", "technical",
     ["The app crashes every time I try to open my profile",
      "Nothing works, the site has been down for me all morning",
      "I keep getting an error code when I try to check out"]),
    ("shipping_delay", "negative", "medium", "shipping",
     ["My package was supposed to arrive 3 days ago and tracking hasn't updated",
      "Order still says 'processing' a week after I placed it",
      "It's been two weeks and I still haven't received my order"]),
    ("account_access", "negative", "high", "account",
     ["I can't log into my account, it keeps saying my password is wrong",
      "I'm locked out of my account after the last update",
      "My account got suspended and I don't know why"]),
    ("general_inquiry", "positive", "low", "other",
     ["Just wanted to say the new update is great, keep up the good work",
      "Quick question, do you offer student discounts?",
      "Loving the new features, when is the next release?"]),
]

SUMMARY_TEMPLATES = {
    "billing_dispute": "Customer disputes a charge on their account",
    "refund_request": "Customer requesting a refund",
    "technical_issue": "Customer reporting a technical problem",
    "shipping_delay": "Customer reporting a delayed shipment",
    "account_access": "Customer unable to access their account",
    "general_inquiry": "Customer general inquiry or feedback",
}


def generate_records(n_per_template: int = 15, seed: int = 42) -> list[dict]:
    random.seed(seed)
    records = []
    for intent, sentiment, urgency, category, messages in TEMPLATES:
        for _ in range(n_per_template):
            message = random.choice(messages)
            records.append({
                "message": message,
                "intent": intent,
                "sentiment": sentiment,
                "urgency": urgency,
                "category": category,
                "summary": SUMMARY_TEMPLATES[intent],
            })
    random.shuffle(records)
    return records


def main(out_path: str = "data/raw/tickets.jsonl") -> None:
    records = generate_records()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    print(f"Wrote {len(records)} synthetic records to {out_path}")
    print("NOTE: this is a small synthetic seed set for pipeline testing. "
          "Replace with real/larger data before trusting the metrics.")


if __name__ == "__main__":
    main()