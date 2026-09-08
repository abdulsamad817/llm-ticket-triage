"""
FastAPI app serving the fine-tuned triage model.

Run with:
    uvicorn src.inference.api:app --reload

Then:
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"message": "I never received my order and it has been two weeks"}'
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference.model import TriageModel

app = FastAPI(
    title="Ticket Triage API",
    description="Turns raw customer support messages into structured triage JSON.",
    version="1.0.0",
)

_model: TriageModel | None = None


def get_model() -> TriageModel:
    global _model
    if _model is None:
        _model = TriageModel()
    return _model


class PredictRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="Raw customer support message")


class PredictResponse(BaseModel):
    intent: str
    sentiment: str
    urgency: str
    category: str
    summary: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    model = get_model()
    result = model.predict(request.message)

    if "error" in result:
        # Surfaced as a 422 rather than silently returning malformed data —
        # a caller should know when the model failed to produce valid output.
        raise HTTPException(status_code=422, detail=result)

    return result