import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import json
from pathlib import Path


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def train_model(df: pd.DataFrame):
    feature_cols = [
        "amount",
        "txn_hour",
    ]
    X = df[feature_cols]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)

    metrics = {
        "classification_report": report,
        "auc": auc,
    }

    return model, metrics


def save_metrics(metrics: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    data_path = "data/sample_transactions.csv"
    metrics_path = "results/model_metrics.json"

    df = load_data(data_path)
    model, metrics = train_model(df)
    save_metrics(metrics, metrics_path)
    print(f"Saved metrics to {metrics_path}")
