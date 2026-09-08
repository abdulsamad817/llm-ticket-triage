from src.data.schema import SchemaValidator
from src.evaluation.evaluate import score_predictions

SCHEMA_CONFIG = {
    "fields": ["intent", "sentiment", "urgency", "category", "summary"],
    "intents": ["billing_dispute", "refund_request"],
    "sentiments": ["positive", "negative"],
    "urgencies": ["low", "high"],
    "categories": ["billing", "other"],
}


def make_validator():
    return SchemaValidator(SCHEMA_CONFIG)


def test_score_predictions_perfect_match():
    validator = make_validator()
    gold = [{"intent": "billing_dispute", "sentiment": "negative", "urgency": "high", "category": "billing"}]
    preds = [{"intent": "billing_dispute", "sentiment": "negative", "urgency": "high",
              "category": "billing", "summary": "x"}]
    metrics = score_predictions(preds, gold, validator)
    assert metrics["json_valid_pct"] == 100.0
    assert metrics["exact_match_pct"] == 100.0
    assert metrics["avg_field_f1"] == 1.0


def test_score_predictions_handles_invalid_json():
    validator = make_validator()
    gold = [{"intent": "billing_dispute", "sentiment": "negative", "urgency": "high", "category": "billing"}]
    preds = [None]  # simulates a JSON parse failure
    metrics = score_predictions(preds, gold, validator)
    assert metrics["json_valid_pct"] == 0.0
    assert metrics["exact_match_pct"] == 0.0


def test_score_predictions_partial_field_correctness():
    validator = make_validator()
    gold = [
        {"intent": "billing_dispute", "sentiment": "negative", "urgency": "high", "category": "billing"},
        {"intent": "refund_request", "sentiment": "positive", "urgency": "low", "category": "other"},
    ]
    preds = [
        {"intent": "billing_dispute", "sentiment": "positive", "urgency": "high",
         "category": "other", "summary": "x"},
        {"intent": "refund_request", "sentiment": "positive", "urgency": "low",
         "category": "other", "summary": "x"},
    ]
    metrics = score_predictions(preds, gold, validator)
    assert metrics["json_valid_pct"] == 100.0
    assert metrics["exact_match_pct"] == 50.0
    assert 0.0 < metrics["avg_field_f1"] < 1.0