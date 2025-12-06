from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

app = FastAPI(title="Fraud Detection with RAG API")


class Transaction(BaseModel):
    amount: float
    txn_hour: int


MODEL_PATH = Path("results/xgb_model.pkl")


def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


model = load_model()


@app.post("/score")
def score_transaction(txn: Transaction):
    if model is None:
        return {"error": "Model not loaded. Train and save model first."}

    df = pd.DataFrame(
        [
            {
                "amount": txn.amount,
                "txn_hour": txn.txn_hour,
            }
        ]
    )
    prob = float(model.predict_proba(df)[:, 1][0])
    return {
        "fraud_probability": prob,
    }
