import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm

# 加载模型、嵌入器、scaler
clf = joblib.load("saved_model/sbert_model/logistic_model.pkl")
scaler = joblib.load("saved_model/sbert_model/scaler.pkl")
model = SentenceTransformer("saved_model/sbert_model")


def predict_abstracts(texts):
    embeddings = model.encode(texts)
    embeddings_scaled = scaler.transform(embeddings)
    preds = clf.predict(embeddings_scaled)
    probs = clf.predict_proba(embeddings_scaled)
    return preds, probs


def batched_predict_abstracts(all_texts, batch_size=32):
    all_preds = []
    all_probs = []

    for i in tqdm(range(0, len(all_texts), batch_size), desc="Predicting"):
        batch_texts = all_texts[i:i+batch_size]
        preds, probs = predict_abstracts(batch_texts)
        all_preds.extend(preds)
        all_probs.extend(probs)

    return all_preds, all_probs


# 示例调用
if __name__ == "__main__":
    df = pd.read_csv("./data_preprocessing/no_duplication.csv")
    id2label = {0: 'not_ok', 1: 'ok'}
    texts = df['Abstract'].tolist()
    predictions, probabilities = batched_predict_abstracts(texts)
    df["Prediction"] = [id2label[prediction] for prediction in predictions]
    df["Probability"] = [max(probability.tolist()) for probability in probabilities]
    df.to_csv("predictions.csv", index=False)


