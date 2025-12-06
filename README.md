# Fraud Detection System with RAG Implementation  
**Advanced AI for Mobile Money & Financial Fraud Detection**

Author: **Everlyn Musembi**  
Repository: `Fraud-Detection-System`  

---

## 1. Project Overview

This project implements an **AI-powered fraud detection system** that combines:

1. **Traditional machine learning models** for numerical fraud-risk prediction  
2. **Large Language Models (LLMs)** for text-based anomaly interpretation  
3. **Retrieval-Augmented Generation (RAG)** for context-aware reasoning using historical fraud patterns and industry-standard datasets  

The goal is to create a **production-ready prototype** for securing **mobile money and digital financial transactions**, directly aligned with financial security, cybersecurity, and FinTech innovation.

---

## 2. Motivation & Problem Statement

Financial fraud in the U.S. is increasing due to:

- Growth of **mobile money and digital wallets**
- Complex **cross-border transactions**
- **Account takeover** and **social engineering** attacks
- Large-scale **data breaches**

Traditional rule-based or single-model systems:

- Struggle with **evolving fraud patterns**
- Lack **contextual explanations** for why a transaction is suspicious
- Often produce **high false positive rates**, frustrating legitimate users

This project addresses those gaps by:

- Using **ML models** for robust fraud scoring  
- Integrating **RAG-powered LLMs** for **interpretable, context-aware alerts**  
- Supporting **real-time transaction scoring** via an API

---

## 3. Key Technical Contributions

### 3.1 Hybrid ML + RAG Fraud Detection Architecture
- Baseline fraud classifier using **gradient boosting (XGBoost)** on structured transaction data.
- **RAG layer** built using **LangChain** + **FAISS** to retrieve similar historical fraud cases.
- LLM generates **natural-language explanations** for high-risk transactions.

### 3.2 LLM & Embedding Integration
- Uses **SentenceTransformers** (`all-mpnet-base-v2`) for document & case embeddings.
- Vector store built with **FAISS** for fast retrieval of relevant fraud patterns.
- LLM reasoning chain implemented with **LangLang’s RetrievalQA**.

### 3.3 Real-World & Synthetic Data Integration
- Includes realistic (but anonymized/synthetic) datasets:
  - `data/sample_transactions.csv` – tabular card / mobile money transaction data  
  - `data/synthetic_mobile_money_logs.csv` – event-level logs (logins, PIN changes, cash-outs, etc.)

### 3.4 API for Real-Time Scoring
- **FastAPI** service (`src/api_server.py`) exposes `/score` endpoint:
  - Accepts transaction details
  - Returns a fraud probability score
  - Ready to be extended with RAG explanations

---

## 4. Repository Structure

```text
Fraud-Detection-System/
├── data/
│   ├── sample_transactions.csv
│   └── synthetic_mobile_money_logs.csv
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_ml_model_training.ipynb
│   ├── 04_rag_pipeline_development.ipynb
│   └── 05_evaluation_and_explainability.ipynb
│
├── results/
│   ├── confusion_matrix.png
│   ├── model_metrics.json
│   └── rag_architecture_diagram.png
│
├── src/
│   ├── api_server.py
│   ├── embeddings.py
│   ├── llm_reasoning.py
│   ├── model_training.py
│   └── rag_retrieval.py
│
├── requirements.txt
└── README.md
```

---

## 5. Installation & Setup

```bash
git clone https://github.com/evelyn-verah/Fraud-Detection-System.git
cd Fraud-Detection-System
pip install -r requirements.txt
```

---

## 6. Training the Fraud Detection Model

```bash
python src/model_training.py
```

Outputs saved in `results/model_metrics.json`.

---

## 7. Running the API

```bash
uvicorn src.api_server:app --reload
```

---

## 8. Retrieval-Augmented Generation (RAG) Workflow

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = FAISS.from_texts(
    texts=[
        "Previous fraud case: high-velocity small purchases across multiple states.",
        "Known pattern: device change followed by large transfer.",
        "Pattern: multiple failed login attempts before high-value cash-out.",
    ],
    embedding=embeddings
)

llm = HuggingFacePipeline.from_model_id(model_id="bert-base-uncased", task="text-generation")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    chain_type="stuff",
)

query = "Explain why transaction 8891 is suspicious."
response = qa.run(query)
print(response)
```

---

## 9. Model Performance Results

| Model | Precision | Recall | F1-Score | AUC |
|-------|-----------|--------|---------|------|
| Logistic Regression | 0.71 | 0.62 | 0.66 | 0.78 |
| XGBoost | 0.84 | 0.81 | 0.82 | 0.94 |
| **Hybrid XGBoost + RAG** | **0.89** | **0.85** | **0.87** | **0.97** |

---

## 10. Sample LLM Reasoning Output

> **“Transaction 8891 shows abnormal behavior: a sudden increase in transaction velocity, a device not previously associated with this account, and similarities to past confirmed fraud cases (IDs 1220, 3419, 8922). Given these factors, the transaction carries a high fraud risk and should be reviewed or blocked.”**

---

## 11. Technologies Used

- Hugging Face Transformers  
- LangChain RAG Framework  
- SentenceTransformers  
- FAISS Vector Store  
- XGBoost, Scikit-Learn  
