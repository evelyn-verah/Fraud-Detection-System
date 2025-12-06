# Fraud Detection System with RAG Implementation

Advanced AI-powered fraud detection system combining traditional machine learning models with Retrieval-Augmented Generation (RAG) for explainable, context-aware fraud detection in financial transactions.

## Project Overview

This project implements an AI-powered fraud detection system that combines:

1. Traditional machine learning models for numerical fraud-risk prediction  
2. Large Language Models (LLMs) for text-based anomaly interpretation  
3. Retrieval-Augmented Generation (RAG) for context-aware fraud reasoning using historical and industry-standard financial datasets  

The system is designed as a production-ready prototype aligned with financial security and mobile money fraud prevention use cases.

## Repository Structure

```text
fraud-detection-rag/
├── data/
│   ├── sample_transactions.csv
│   ├── synthetic_mobile_money_logs.csv
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_ml_model_training.ipynb
│   ├── 04_rag_pipeline_development.ipynb
│   └── 05_evaluation_and_explainability.ipynb
├── src/
│   ├── model_training.py
│   ├── rag_retrieval.py
│   ├── llm_reasoning.py
│   ├── embeddings.py
│   └── api_server.py
├── results/
│   ├── model_metrics.json
│   ├── confusion_matrix.png
│   ├── rag_architecture_diagram.png
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/<your-username>/fraud-detection-rag.git
cd fraud-detection-rag
pip install -r requirements.txt
```

To run the API:

```bash
uvicorn src.api_server:app --reload
```

## EB-2 / National Interest Alignment (Brief Summary)

This project demonstrates a hybrid ML + RAG fraud detection system that:

- Improves fraud detection metrics over baseline models
- Provides explainable, context-aware fraud alerts for analysts
- Illustrates practical application of LLMs and retrieval-augmented workflows for financial security

You can expand this section in your petition materials (Exhibit 3.13) with more narrative language and citations to your broader work.
