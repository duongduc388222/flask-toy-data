from flask import Flask, render_template, request
import numpy as np
import joblib
from sklearn.datasets import load_iris

app = Flask(__name__)

# Load pipeline (scaler + classifier)
pipeline = joblib.load("models/iris_pipeline.joblib")
iris = load_iris()
TARGET_NAMES = iris.target_names

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        sl = float(request.form.get("sepal_length", 0.0))
        sw = float(request.form.get("sepal_width", 0.0))
        pl = float(request.form.get("petal_length", 0.0))
        pw = float(request.form.get("petal_width", 0.0))

        X = np.array([[sl, sw, pl, pw]])
        pred_idx = int(pipeline.predict(X)[0])
        proba = pipeline.predict_proba(X)[0]

        result = {
            "class_index": pred_idx,
            "class_label": TARGET_NAMES[pred_idx],
            "proba_by_class": {TARGET_NAMES[i]: float(p) for i, p in enumerate(proba)}
        }
        return render_template("index.html", result=result, sl=sl, sw=sw, pl=pl, pw=pw)
    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    # For local dev only. In production use gunicorn: gunicorn -w 2 -b 0.0.0.0:8000 app:app
    app.run(host="0.0.0.0", port=5000, debug=True)
