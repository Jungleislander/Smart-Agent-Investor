# 💼 Agentic AI for Personal Financial Guidance — Project Proposal

## 🎯 Project Vision

Build an **intelligent, agentic AI system** for real-time **personal financial coaching and micro-investment guidance**, capable of planning, reasoning, and acting across multiple financial domains. The goal is to create an **MVP-level product** that blends **AI decision-making** with **real-world financial actions**, empowering users to understand, optimize, and act on their finances with confidence.

Inspired by systems used by modern investment firms, this platform will utilize **multi-agent workflows** that parse news, analyze behavior, generate insights, and provide **actionable, personalized financial advice** — not just static reports.

---

## 🧪 Detailed Training: Financial News Sentiment Classifier

### 🎯 Goal
Train a sentiment classification model that can categorize financial news headlines or short articles as:
- 🟢 Positive
- ⚪ Neutral
- 🔴 Negative

### 📂 Dataset
- **Primary Training Dataset**: [Financial PhraseBank (Kaggle)](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)
  - Pre-labeled dataset with 3 sentiment classes
  - Clean, balanced, and domain-specific

### 🛠️ Training Notes
- Model type: `bert-base-uncased` or `distilbert-base-uncased` via Hugging Face Transformers
- Simple text classification pipeline using `Trainer` or `text-classification` pipeline

### 🧪 Evaluation & Testing
- After training, use **unlabeled real-world data** from:
  - NewsAPI.org
  - Yahoo Finance news feed
  - Reuters Financial News
- Manually label a subset of ~100–200 examples to evaluate generalization
- Optional: incorporate these labeled examples later to fine-tune the model further

### ✅ Outcome
- Export model to `models/financial_sentiment_model/`
- Save tokenizer and config alongside weights
- Push to Hugging Face Hub or serve via FastAPI for inference

---

## 🧪 Detailed Training: Crypto vs Stock Topic Classifier

### 🎯 Goal
Train a topic classification model that categorizes financial headlines or short articles into:
- 🪙 `crypto`
- 🧾 `stock`
- ⚪ `general`

This is a **topic classifier**, not a sentiment model.

### 📂 Dataset
- Collect ~300–500 financial headlines from sources like NewsAPI.org, Yahoo Finance, and Reuters.
- Manually label each entry as `crypto`, `stock`, or `general`.

Example:

| Headline | Label |
|---------|-------|
| "Bitcoin hits $40K after ETF buzz" | crypto |
| "Apple announces Q4 earnings" | stock |
| "US interest rates remain steady" | general |

### 🛠️ Training Notes
- Model type: `bert-base-uncased` or `distilbert-base-uncased`
- Training: Hugging Face `Trainer` API or `text-classification` pipeline
- Labels are flat classes, not spans — this is not NER.

### 🔁 Alternative: Zero-Shot Classification
You can skip training and use **zero-shot classification** via:
- Hugging Face’s `zero-shot-classification` pipeline
- OpenAI (e.g., GPT-3.5/4o) with candidate label prompting

This enables flexible, dynamic routing of real-time content without pre-training.

### 🧠 How It Fits in the Pipeline
- First, the **sentiment model** determines the tone (positive/neutral/negative).
- Then, the **topic classifier** determines the content type (crypto, stock, general).
- Both outputs are passed to a downstream **Agent** (e.g., an OpenAI-powered summarizer or recommender) for next-step analysis or user suggestions.

This mirrors how modern agentic pipelines work — combining classification with reasoning tools.

---

## 🧪 Rule-Based Logic: Risk Profile Estimator

### 🎯 Goal
Estimate a user's financial risk tolerance using a simple rule-based system — no ML model or training required for MVP.

### 🛠️ Method
- Present users with a short questionnaire (3–5 questions), such as:
  - "How long can you keep your money invested?"
    - Short (0–1 years) → 10 points
    - Medium (1–3 years) → 30 points
    - Long (3+ years) → 50 points
  - "What would you do if your investment dropped 20%?"
    - Sell everything → 0 points
    - Wait and see → 25 points
    - Buy more → 50 points
- Assign numeric values to answers
- Total score determines risk profile

### 🧮 Risk Scoring Bands (Initial Example)
- 0–30 → Low Risk
- 31–70 → Medium Risk
- 71–100 → High Risk

> ⚠️ These bands are rough and may be revised for more granularity or financial accuracy.

### ✅ Outcome
- Return `risk_score` (0–100)
- Categorize user as `low`, `medium`, or `high` risk
- Feed into Planner Agent for personalized ETF/GIC suggestions

## 🧠 Core Agentic Patterns

The system implements three essential **agentic AI workflows**:

### 1. Prompt Chaining
Sequential LLM tasks:
```
Ingest News → Clean/Preprocess → Classify → Extract Entities → Summarize
```

### 2. Routing
A Java-based core agent intelligently routes content or events to the appropriate specialized agent:
- Market movement → ETF summarizer
- Payment due → Credit assistant
- User saved money → Investment recommender

### 3. Evaluator–Optimizer
Each action (recommendation or insight) is **evaluated** by a reflection agent. Feedback is used to:
- Re-prompt for better answers
- Fine-tune recommendations
- Improve user alignment

---

## 💡 MVP Concepts

We will begin by focusing on **two MVP flows**:

### ✅ Micro-Investor Guide
Helps users automatically invest spare savings into GICs, ETFs, or low-risk instruments.

#### Example Flow:
> "You've saved $68 this week. Want to invest it?"  
> → Suggest 3 ETFs based on risk profile  
> → User confirms  
> → System adds recurring option or follow-up

### ✅ SmartPay Coach
Optimizes credit card payments based on balance, due dates, and income schedule.

#### Example Flow:
> "Pay $150 now to avoid $18 interest."  
> → Explains impact on credit score & cashback  
> Offers to schedule with Google Pay/Calendar

---

## ⚙️ Technical Stack

### 🧱 Backend & Infra (Production-Ready)
| Component | Tech |
|----------|------|
| API & Logic | Java Spring Boot (MCP-style Dispatcher) |
| Messaging | Apache Kafka (multi-agent event routing) |
| Cloud | AWS (ECS, S3, RDS, CloudFront, ECR) |
| Data Storage | S3 (documents, JSON), H2 or RDS (user metadata) |
| Auth & Users | Spring Security / JWT |

### 🤖 AI & NLP Stack
| Component | Tool |
|----------|------|
| NLP Toolkit | Hugging Face Transformers + Datasets |
| Prompt Execution | OpenAI GPT-4o-mini (fast), GPT-3.5 |
| Agent Framework | LangChain (Python), LangChain4J (Java) |
| Tokenizer & Preprocessing | `datasets`, `nltk`, `spaCy`, etc. |
| Embedding & Vector Search | Optional: FAISS or Qdrant (for RAG) |

### 🔌 External SDKs
| Use Case | Tool |
|----------|------|
| Calendar/Reminders | Google Calendar SDK (Java or Python) |
| Wallet Integration | Google Pay SDK |
| Financial Data APIs | Yahoo Finance, NewsAPI, SEC EDGAR, FRED |

### 🌐 Frontend
| Component | Tech |
|----------|------|
| Web UI | React |
| Mobile Optimization | Responsive, PWA-style dashboard |
| Features | Dashboard cards, prompt responses, confirmation dialogs |

---

## 🔄 Agent Architecture Example

```mermaid
graph TD
    UserInput[User: "I saved $68"]
    UI[React Frontend]
    Dispatcher[Java Agentic Dispatcher]
    PlannerAgent[Java Planner Agent]
    RiskAnalyzer[Python Risk Classifier]
    ETFInfoAgent[LangChain: Market Summary]
    OpenAIAgent[LLM: Summarization]
    EvaluatorAgent[Feedback Evaluator]
    ReminderAgent[Google SDK: Calendar Reminder]

    UserInput --> UI --> Dispatcher
    Dispatcher --> PlannerAgent --> ETFInfoAgent
    Dispatcher --> RiskAnalyzer
    ETFInfoAgent --> OpenAIAgent --> UI
    UI --> EvaluatorAgent --> Dispatcher
    Dispatcher --> ReminderAgent
```

---

## 🔁 Hugging Face Integration

| Use | Method |
|-----|--------|
| Pretraining | Use Kaggle datasets, News API, Yahoo Finance |
| Model Hosting | Upload to HF Hub or S3 |
| Inference | Serve via Hugging Face Inference Endpoints or FastAPI |
| Fine-tuning | Use `Trainer` API or `AutoTrain` |
| Evaluation | Use HF `evaluate` package + test prompts |
| LoRA Tuning | Lightweight fine-tuning for new user data |

---

## 📡 System Interactions (Live Example)

- Java Spring Boot service receives Kafka event: `user.saved.money`
- Dispatcher routes to:
    - Planner Agent → assess income/expense
    - ETF Agent → fetch & summarize options
    - LLM Agent → simplify for end user
- User accepts → Google Calendar Agent schedules follow-up
- Evaluator Agent logs behavior and adapts prompts

---

## 🧪 Testing & Evaluation

- Automated test prompts to validate outputs
- Human-in-the-loop to score summary quality
- Evaluator Agent feedback loop → stores performance metadata

---

## 🧭 Final Notes

This project represents a powerful step toward real-world **agentic AI systems** that don’t just chat — they **plan, decide, and act** on behalf of users.

It combines:
- 🔒 Secure, Java-based backend orchestration (ideal for banks)
- 🤖 Best-in-class NLP tooling (HF + OpenAI)
- 🧠 Agentic decision patterns (planner, critic, router)
- ⚡ Real-time integration with APIs + calendars

---

## Summarization Classifier

The summarization classifier is responsible for taking all the inputs generated by the previous components—such as the sentiment classification, topic classifier, and the risk profile estimator—and generating clear, simplified outputs that users can understand and act upon.

### Description:
- This is not a RAG (Retrieval-Augmented Generation) at MVP stage.
- This component will use OpenAI (or a local Hugging Face model) to:
  - Summarize financial news or earnings reports.
  - Rephrase and simplify ETF and GIC product descriptions.
  - Provide user-specific recommendations in natural language.
    - Example: "The market is volatile. We suggest a short-term GIC based on your risk profile."

### Pipeline Flow:
- Run sentiment classifier on latest news samples.
- Run topic classifier to label asset type (e.g., Tesla, BTC).
- Merge with user-specific inputs (from risk estimator).
- Use a prompt + abstraction layer (e.g., LangChain) to structure the query.
- Call OpenAI to summarize and generate recommendations.

This component acts as the final layer translating raw analysis into user-facing insights.
