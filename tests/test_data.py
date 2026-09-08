import json

import pytest

from src.data.schema import SchemaValidator, TicketLabel
from src.data.prepare_dataset import clean_record, split_records


SCHEMA_CONFIG = {
    "fields": ["intent", "sentiment", "urgency", "category", "summary"],
    "intents": ["billing_dispute", "refund_request", "technical_issue"],
    "sentiments": ["positive", "neutral", "negative"],
    "urgencies": ["low", "medium", "high"],
    "categories": ["billing", "technical", "other"],
}


@pytest.fixture
def validator():
    return SchemaValidator(SCHEMA_CONFIG)


def test_ticket_label_serializes_to_valid_json():
    label = TicketLabel(
        intent="billing_dispute", sentiment="negative", urgency="high",
        category="billing", summary="Double charged for order",
    )
    parsed = json.loads(label.to_json_str())
    assert parsed["intent"] == "billing_dispute"
    assert parsed["summary"] == "Double charged for order"


def test_validator_parses_clean_json(validator):
    raw = '{"intent": "billing_dispute", "sentiment": "negative", "urgency": "high", "category": "billing", "summary": "test"}'
    parsed = validator.parse(raw)
    assert parsed is not None
    assert validator.is_valid(parsed)


def test_validator_strips_markdown_code_fences(validator):
    raw = '```json\n{"intent": "refund_request", "sentiment": "neutral", "urgency": "low", "category": "other", "summary": "x"}\n```'
    parsed = validator.parse(raw)
    assert parsed is not None
    assert parsed["intent"] == "refund_request"


def test_validator_rejects_malformed_json(validator):
    assert validator.parse("this is not json at all") is None


def test_validator_rejects_invalid_field_values(validator):
    raw = '{"intent": "not_a_real_intent", "sentiment": "negative", "urgency": "high", "category": "billing", "summary": "x"}'
    parsed = validator.parse(raw)
    assert parsed is not None  # it IS valid json...
    assert not validator.is_valid(parsed)  # ...but not a valid label


def test_validator_rejects_missing_fields(validator):
    raw = '{"intent": "billing_dispute", "sentiment": "negative"}'
    parsed = validator.parse(raw)
    assert not validator.is_valid(parsed)


def test_clean_record_drops_short_messages():
    record = {
        "message": "hi", "intent": "x", "sentiment": "x",
        "urgency": "x", "category": "x", "summary": "x",
    }
    assert clean_record(record) is None


def test_clean_record_drops_incomplete_records():
    record = {"message": "a perfectly long enough message here"}
    assert clean_record(record) is None


def test_clean_record_keeps_valid_records():
    record = {
        "message": "  I need a refund for my broken item  ",
        "intent": "refund_request", "sentiment": "negative",
        "urgency": "medium", "category": "billing", "summary": "Refund request",
    }
    cleaned = clean_record(record)
    assert cleaned is not None
    assert cleaned["message"] == "I need a refund for my broken item"  # whitespace stripped


def test_split_records_respects_proportions():
    config = {"data": {"train_split": 0.8, "val_split": 0.1, "test_split": 0.1, "seed": 42}}
    records = [{"id": i} for i in range(100)]
    splits = split_records(records, config)
    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10


def test_split_records_no_overlap_between_splits():
    config = {"data": {"train_split": 0.8, "val_split": 0.1, "test_split": 0.1, "seed": 42}}
    records = [{"id": i} for i in range(100)]
    splits = split_records(records, config)
    train_ids = {r["id"] for r in splits["train"]}
    val_ids = {r["id"] for r in splits["val"]}
    test_ids = {r["id"] for r in splits["test"]}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)