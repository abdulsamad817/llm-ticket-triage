# 🤖 LLM Ticket Triage

An end-to-end **LLM-powered customer support ticket triage system** that converts unstructured customer messages into structured, machine-readable ticket intelligence.

The project explores and compares **zero-shot prompting, few-shot prompting, LoRA fine-tuning, and QLoRA fine-tuning** using the `Qwen2.5-1.5B-Instruct` model. The resulting model can be served through a lightweight **FastAPI REST API**.

> **Project status:** Implementation complete, but full end-to-end execution has not yet been verified on a local/GPU environment. The repository is structured to support reproducible training, evaluation, and inference.

---

## 🚀 Overview

Customer-support teams receive large volumes of unstructured messages that need to be categorized, prioritized, and routed efficiently.

**LLM Ticket Triage** is designed to automate this process by transforming a raw customer message into a structured JSON response containing:

* **Intent**
* **Sentiment**
* **Urgency**
* **Category**
* **Summary**

### Example

**Input**

```text
I never received my order and it has been two weeks.
```

**Expected structured output**

```json
{
  "intent": "shipping_delay",
  "sentiment": "negative",
  "urgency": "medium",
  "category": "shipping",
  "summary": "Customer reporting a delayed shipment"
}
```

This structured representation can then be consumed by customer-support software, routing systems, dashboards, or downstream automation.

---

## ✨ Key Features

* 🧠 LLM-based customer-support ticket classification
* 🎯 Structured JSON output
* 🏷️ Intent classification
* 😊 Sentiment classification
* 🚨 Urgency classification
* 📂 Ticket category classification
* 📝 Automatic ticket summarization
* ⚡ Parameter-efficient **LoRA fine-tuning**
* 💾 **QLoRA / 4-bit quantization** configuration
* 🔬 Zero-shot vs few-shot vs LoRA vs QLoRA evaluation
* 🛡️ Schema-based output validation
* 🚀 FastAPI inference service
* ❤️ Health-check endpoint
* 🧪 Pytest support
* 📊 Weights & Biases experiment tracking
* ⚙️ YAML-based configuration
* 🔄 End-to-end ML pipeline structure

---

## 🛠️ Tech Stack

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black" alt="Hugging Face Transformers"/>
  <img src="https://img.shields.io/badge/Qwen-2.5-6E56CF" alt="Qwen"/>
  <img src="https://img.shields.io/badge/PEFT-LoRA-FF6F00" alt="PEFT LoRA"/>
  <img src="https://img.shields.io/badge/QLoRA-4--bit-8A2BE2" alt="QLoRA"/>
  <img src="https://img.shields.io/badge/FastAPI-0.111%2B-009688?logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Pydantic-2.x-E92063?logo=pydantic&logoColor=white" alt="Pydantic"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Pytest-8%2B-0A9EDC?logo=pytest&logoColor=white" alt="Pytest"/>
  <img src="https://img.shields.io/badge/Weights%20%26%20Biases-Experiment%20Tracking-FFBE00?logo=weightsandbiases&logoColor=black" alt="Weights and Biases"/>
</p>

### Core technologies

| Technology                    | Purpose                                    |
| ----------------------------- | ------------------------------------------ |
| **Python**                    | Application and ML pipeline development    |
| **PyTorch**                   | Deep learning framework                    |
| **Qwen2.5-1.5B-Instruct**     | Base language model                        |
| **Hugging Face Transformers** | Model loading, tokenization and generation |
| **PEFT**                      | Parameter-efficient fine-tuning            |
| **LoRA**                      | Low-rank model adaptation                  |
| **bitsandbytes**              | Quantization / QLoRA support               |
| **Datasets**                  | Dataset loading and processing             |
| **FastAPI**                   | REST API serving                           |
| **Pydantic**                  | API request/response validation            |
| **PyYAML**                    | Configuration management                   |
| **Pytest**                    | Testing framework                          |
| **Weights & Biases**          | Experiment tracking                        |

---

## 🏗️ Architecture

```text
                         ┌─────────────────────┐
                         │ Customer Support     │
                         │ Message             │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Prompt Construction │
                         │ + Schema Definition │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │       Qwen2.5-1.5B            │
                    │          Instruct             │
                    └──────────────┬────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
             Zero/Few-shot                 LoRA / QLoRA
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │ Structured JSON     │
                         │ Prediction          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Schema Validator    │
                         │ JSON + Vocabulary   │
                         │ Validation          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ FastAPI             │
                         │ /predict            │
                         │ /health             │
                         └─────────────────────┘
```

---

## 📁 Project Structure

```text
llm-ticket-triage/
│
├── configs/
│   └── train_config.yaml
│
├── data/
│   ├── raw/
│   │   └── tickets.jsonl
│   └── processed/
│       ├── train.jsonl
│       ├── val.jsonl
│       └── test.jsonl
│
├── src/
│   ├── data/
│   │   ├── generate_synthetic_data.py
│   │   └── schema.py
│   │
│   ├── training/
│   │   └── train_lora.py
│   │
│   ├── evaluation/
│   │   └── evaluate.py
│   │
│   └── inference/
│       ├── model.py
│       └── api.py
│
├── outputs/
│   └── ...
│
├── requirements.txt
└── README.md
```

---

## 🧠 Model

The project uses:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

The model is adapted for the ticket-triage task using **Parameter-Efficient Fine-Tuning (PEFT)**.

Rather than updating every parameter in the base model, LoRA introduces trainable low-rank matrices into selected model layers.

### LoRA configuration

```yaml
r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
bias: none
task_type: CAUSAL_LM
```

The attention projection layers are targeted for adaptation:

```text
q_proj
k_proj
v_proj
o_proj
```

This reduces the number of trainable parameters compared with full-model fine-tuning.

---

## ⚡ QLoRA

The project also defines a QLoRA configuration using 4-bit quantization:

```yaml
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_use_double_quant: true
```

The goal is to provide a more memory-efficient fine-tuning approach for environments with constrained GPU memory.

---

## 🗂️ Ticket Schema

Every prediction follows a consistent schema:

```json
{
  "intent": "...",
  "sentiment": "...",
  "urgency": "...",
  "category": "...",
  "summary": "..."
}
```

### Supported intents

```text
billing_dispute
refund_request
technical_issue
shipping_delay
account_access
general_inquiry
```

### Supported sentiments

```text
positive
neutral
negative
```

### Supported urgency levels

```text
low
medium
high
```

### Supported categories

```text
billing
technical
shipping
account
other
```

---

## 🛡️ Structured Output Validation

One of the core engineering components is the `SchemaValidator`.

The validator checks that the model output:

1. Is valid JSON
2. Is a JSON object
3. Contains all required fields
4. Uses allowed values for categorical fields

Invalid JSON is not silently converted into a valid prediction.

For example, an unexpected value such as:

```json
{
  "sentiment": "angry"
}
```

can be rejected because `angry` is not part of the configured sentiment vocabulary.

This provides a clear contract between the LLM and downstream application code.

---

## 📊 Evaluation

The evaluation pipeline is designed to compare four inference strategies using the **same held-out test set**:

| Method    | Description                            |
| --------- | -------------------------------------- |
| Zero-shot | Base model with the task instruction   |
| Few-shot  | Base model with example demonstrations |
| LoRA      | Base model + trained LoRA adapter      |
| QLoRA     | Base model + QLoRA adapter             |

The evaluation configuration is:

```yaml
methods:
  - zero_shot
  - few_shot
  - lora
  - qlora
```

The pipeline records:

* JSON validity percentage
* Exact-match percentage
* Per-field correctness
* Average field score
* Average generation latency

Results are written to:

```text
outputs/comparison_results.json
```

> **Important:** The repository currently contains the evaluation implementation, but benchmark results should only be reported after the pipeline has actually been executed. No fabricated benchmark numbers are presented here.

---

## 🧪 Dataset

The repository includes a synthetic seed-data generator so the pipeline can be exercised without requiring access to private customer-support data.

The generator creates examples covering six ticket intents and associated metadata.

Current configuration:

```text
6 intents
3 sentiment classes
3 urgency levels
5 categories
15 examples per intent
```

This produces:

```text
90 synthetic records
```

### Dataset limitation

The synthetic dataset is intentionally small and should **not** be considered representative of real-world customer-support traffic.

For production-quality experimentation, the dataset should be replaced or expanded with:

* Real support tickets with personally identifiable information removed
* A larger synthetic dataset
* Human-reviewed examples
* More diverse language and edge cases
* Class-balance analysis
* Robust train/validation/test splits

Therefore, any future benchmark results should be interpreted as **experimental results**, not evidence of production readiness.

---

## ⚙️ Configuration

Training and evaluation settings are centralized in:

```text
configs/train_config.yaml
```

This includes:

* Base model
* Sequence length
* Dataset paths
* Data split ratios
* Random seed
* Schema vocabulary
* LoRA parameters
* QLoRA parameters
* Training hyperparameters
* Evaluation methods
* Generation settings

This makes experiments easier to reproduce and modify without changing the source code.

---

## 🔧 Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd llm-ticket-triage
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Generate the Dataset

Run:

```bash
python -m src.data.generate_synthetic_data
```

This creates:

```text
data/raw/tickets.jsonl
```

The generated dataset is intended as a small seed dataset for pipeline development.

---

## 🔄 Data Preparation

The training configuration expects processed files at:

```text
data/processed/
├── train.jsonl
├── val.jsonl
└── test.jsonl
```

The configured split is:

```yaml
train_split: 0.8
val_split: 0.1
test_split: 0.1
```

The preprocessing stage should create these files before training or evaluation.

---

## 🏋️ LoRA Training

Once the processed dataset is available, the LoRA training command is:

```bash
python -m src.training.train_lora --config configs/train_config.yaml
```

The resulting adapter is expected at:

```text
outputs/lora/final_adapter/
```

The training pipeline uses:

* Hugging Face Transformers
* Hugging Face Datasets
* PEFT
* PyTorch
* bfloat16
* Hugging Face Trainer

---

## 📈 Evaluation

After the required model adapters and processed test set are available:

```bash
python -m src.evaluation.evaluate --config configs/train_config.yaml
```

The comparison is saved to:

```text
outputs/comparison_results.json
```

---

## 🚀 API

The project includes a FastAPI serving layer.

Start the API with:

```bash
uvicorn src.inference.api:app --reload
```

The API will expose:

```text
GET  /health
POST /predict
```

### Health check

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok"
}
```

### Prediction

Send a customer-support message:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"message":"I never received my order and it has been two weeks"}'
```

The API validates the request and passes the message to the `TriageModel`.

If the model produces valid structured output, the API returns the prediction.

If the model produces invalid JSON or unexpected schema values, the API returns an HTTP `422` error rather than silently returning malformed data.

---

## 🔌 API Contract

### `POST /predict`

Request:

```json
{
  "message": "Customer support message goes here"
}
```

The request message must:

* Contain at least one character
* Contain no more than 4,000 characters

Response:

```json
{
  "intent": "shipping_delay",
  "sentiment": "negative",
  "urgency": "medium",
  "category": "shipping",
  "summary": "Customer reporting a delayed shipment"
}
```

### `GET /health`

Response:

```json
{
  "status": "ok"
}
```

---

## 🧩 Design Decisions

### Separation of concerns

The project separates:

```text
Data generation
      ↓
Schema validation
      ↓
Training
      ↓
Evaluation
      ↓
Model inference
      ↓
API serving
```

The FastAPI application does not contain the model implementation directly. Instead, it delegates model operations to `TriageModel`.

This makes the inference component independently reusable and testable.

### Single schema source

The schema configuration defines the allowed fields and categorical vocabulary.

The same schema is used to:

* Construct prompts
* Represent labels
* Validate model responses

This reduces inconsistencies between training and inference.

### Lazy model loading

The API loads the model when it is first required rather than loading it during module import.

This keeps application initialization lightweight and allows the model layer to remain independently configurable.

---

## 🔐 Error Handling

The inference pipeline explicitly handles malformed model responses.

Possible failures include:

### Invalid JSON

```text
Model did not return valid JSON.
```

### Unexpected fields or values

```text
Model returned JSON with unexpected fields/values.
```

The API surfaces these failures with HTTP `422` rather than silently returning an invalid prediction.

---

## 🧪 Testing

Pytest is included in the project dependencies:

```bash
pytest
```

Tests should cover areas such as:

* Schema validation
* JSON parsing
* Required fields
* Allowed categorical values
* Model inference
* API request validation
* API error handling

> Full automated test execution has not yet been verified for this repository.

---

## 📊 Experiment Tracking

The training configuration supports **Weights & Biases**:

```yaml
report_to: "wandb"
run_name: "ticket-triage-lora"
```

This allows training runs and experiment metadata to be tracked when W&B is configured.

---

## ⚠️ Current Limitations

This project is currently a **research/prototype implementation** rather than a production-ready support system.

### Dataset size

The included dataset is synthetic and intentionally small.

### Execution verification

The complete pipeline has not yet been run end-to-end in the current development environment.

### Hardware requirements

LLM fine-tuning and inference can require significant GPU memory, particularly when working with transformer models and bfloat16 computation.

### Evaluation metric naming

The current evaluation implementation reports field-level correctness ratios under the `per_field_f1` key. These values should be treated as **field accuracy/correctness rates**, not conventional F1 scores, until the evaluation implementation is updated to calculate actual precision/recall-based F1.

### Production data

Real-world deployment would require a substantially larger, diverse, privacy-safe dataset and additional validation.

---

## 🔮 Future Improvements

Potential next steps include:

* [ ] Add a complete preprocessing script
* [ ] Add automated unit and integration tests
* [ ] Run and record real benchmark results
* [ ] Add actual F1, precision, and recall metrics
* [ ] Expand the training dataset
* [ ] Add real-world anonymized support-ticket examples
* [ ] Add confusion matrices
* [ ] Add latency and throughput benchmarks
* [ ] Add Docker deployment
* [ ] Add CI/CD
* [ ] Add API authentication
* [ ] Add request logging and observability
* [ ] Add model versioning
* [ ] Add batch inference
* [ ] Add production monitoring

---

## 🎯 Engineering Goals

This project demonstrates an end-to-end approach to building an LLM application rather than focusing exclusively on model training.

The main engineering goals are:

```text
Model adaptation
       +
Structured generation
       +
Output validation
       +
Reproducible evaluation
       +
API serving
       +
Software engineering practices
```

The project is designed to demonstrate how an LLM can be adapted to a specialized task and integrated into an application layer.

---

## 📌 Validation Status

| Component                 | Implementation | End-to-end verified |
| ------------------------- | -------------: | ------------------: |
| Synthetic data generation |              ✅ | ⚠️ Not yet verified |
| Schema validation         |              ✅ | ⚠️ Not yet verified |
| LoRA training pipeline    |              ✅ | ⚠️ Not yet verified |
| QLoRA configuration       |              ✅ | ⚠️ Not yet verified |
| Model evaluation          |              ✅ | ⚠️ Not yet verified |
| FastAPI application       |              ✅ | ⚠️ Not yet verified |
| `/health` endpoint        |              ✅ | ⚠️ Not yet verified |
| `/predict` endpoint       |              ✅ | ⚠️ Not yet verified |
| Automated tests           |     Configured | ⚠️ Not yet verified |

---

## 👨‍💻 Project Author

**Abdul-Samad**

Built as an end-to-end exploration of **LLM fine-tuning, parameter-efficient adaptation, structured model outputs, evaluation, and API serving**.

---

## 📄 License

Add the license appropriate for your repository before publishing.

