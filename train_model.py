#!/usr/bin/env python3
"""
Train a simple Iris classifier and save it to disk.
Usage:
    python train_model.py
Outputs:
    models/scaler.joblib
    models/model.joblib
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np
import os

def main():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=None))
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    print("Test accuracy:", accuracy_score(y_test, preds))
    print("\nClassification report:\n", classification_report(y_test, preds, target_names=iris.target_names))

    os.makedirs("models", exist_ok=True)
    # Save the whole pipeline (scaler+model) as one artifact for simplicity
    joblib.dump(pipe, "models/iris_pipeline.joblib")
    print("Saved pipeline to models/iris_pipeline.joblib")

if __name__ == "__main__":
    main()
