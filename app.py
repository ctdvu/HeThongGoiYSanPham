"""Flask web app for the RNN product recommender."""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

# Render's starter instances have limited CPU/RAM. These must be set before
# TensorFlow is imported so one inference worker does not create many threads.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from flask import Flask, render_template, request
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
app = Flask(__name__)


def load_resources():
    """Load the trained neural network and its inference metadata once at startup."""
    model_path = MODEL_DIR / "rnn_product_recommender.keras"
    artifact_path = MODEL_DIR / "artifacts.pkl"
    catalog_path = MODEL_DIR / "catalog.json"
    if not all(path.exists() for path in (model_path, artifact_path, catalog_path)):
        return None, None, None
    with artifact_path.open("rb") as file:
        artifacts = pickle.load(file)
    with catalog_path.open(encoding="utf-8") as file:
        catalog = {str(item["product_id"]): item for item in json.load(file)}
    # compile=False avoids loading the optimizer, which is only needed for training.
    return keras.models.load_model(model_path, compile=False), artifacts, catalog


MODEL, ARTIFACTS, CATALOG = load_resources()

if MODEL is not None:
    # Make the first TensorFlow graph execution happen during service startup,
    # rather than making the first user's POST request wait for initialization.
    _warmup_tokens = np.zeros((1, ARTIFACTS["sequence_length"]), dtype="int32")
    MODEL(_warmup_tokens, training=False).numpy()


def product_details(product_id: str, confidence: float) -> dict:
    """Attach display metadata to a predicted product ID."""
    item = CATALOG.get(product_id, {})
    return {
        "product_id": product_id,
        "brand": item.get("brand") or "Không rõ thương hiệu",
        "category": item.get("category_code") or "Danh mục khác",
        "price": float(item.get("price") or 0),
        "confidence": round(confidence * 100, 2),
    }


def predict(history: list[str], k: int = 5) -> list[dict]:
    """Convert product history to padded tokens and return five ranked products."""
    mapping = ARTIFACTS["product_to_token"]
    inverse = {int(token): product for token, product in ARTIFACTS["token_to_product"].items()}
    length = ARTIFACTS["sequence_length"]
    tokens = [mapping.get(product_id, 0) for product_id in history][-length:]
    tokens = [0] * (length - len(tokens)) + tokens
    # Calling the model directly is leaner than Model.predict for one request.
    probabilities = MODEL(np.asarray([tokens], dtype="int32"), training=False).numpy()[0]
    ranked = probabilities.argsort()[-(k + len(history)):][::-1]
    result = []
    for token in ranked:
        product_id = inverse.get(int(token))
        if product_id and product_id not in history:
            result.append(product_details(product_id, float(probabilities[token])))
        if len(result) == k:
            break
    return result


@app.route("/", methods=["GET", "POST"])
def index():
    recommendations, error, history_text = [], None, ""
    if request.method == "POST":
        history_text = request.form.get("history", "").strip()
        history = [item.strip() for item in history_text.replace("\n", ",").split(",") if item.strip()]
        if not MODEL:
            error = "Chưa có mô hình. Hãy chạy toàn bộ notebook để tạo thư mục models/."
        elif not history:
            error = "Nhập ít nhất một product ID, ngăn cách bằng dấu phẩy."
        else:
            recommendations = predict(history)
            if not recommendations:
                error = "Không có sản phẩm phù hợp trong từ điển mô hình. Thử các product ID xuất hiện trong dữ liệu train."
    return render_template("index.html", recommendations=recommendations, error=error, history=history_text)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
