import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from xgboost import XGBClassifier
import random
import numpy as np
import torch
from nltk.tokenize import sent_tokenize
import nltk
from tqdm import tqdm
tqdm.pandas()

nltk.data.path.append("./nltk_data")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def extract_middle_part(text, ratio=0.4):
    if not isinstance(text, str) or not text.strip():
        return ""
    sentences = sent_tokenize(text)
    total = len(sentences)
    if total <= 2:
        return text  # 不足时直接返回原文
    start = max(0, int(total * (0.5 - ratio / 2)))
    end = min(total, int(total * (0.5 + ratio / 2)))
    return " ".join(sentences[start:end])


# ✅ Step 1: 加载数据
def load_data(path):
    df = pd.read_csv(path)
    df = df[['Abstract', 'label']].dropna()
    df['label'] = df['label'].astype(int)
    return df


train_df = load_data("train.csv")
eval_df = load_data("eval.csv")

train_df['Abstract'] = train_df['Abstract'].apply(lambda x: extract_middle_part(x))
eval_df['Abstract'] = eval_df['Abstract'].apply(lambda x: extract_middle_part(x))

# ✅ Step 2: 加载 SentenceTransformer 模型并提取嵌入
print("Loading model...")
model = SentenceTransformer('all-mpnet-base-v2')  # 自动下载或已缓存

print("Encoding sentences...")
X_train = model.encode(train_df['Abstract'].tolist(), show_progress_bar=True)
X_eval = model.encode(eval_df['Abstract'].tolist(), show_progress_bar=True)
y_train = train_df['label'].values
y_eval = eval_df['label'].values

# ✅ Step 3: 标准化（可选）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_eval_scaled = scaler.transform(X_eval)

# ✅ Step 4: 训练分类器

clf = XGBClassifier(n_estimators=100, max_depth=6)
# clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_scaled, y_train)

# ✅ Step 5: 评估结果
y_pred = clf.predict(X_eval_scaled)
print("\n📊 Classification Report:")
print(classification_report(y_eval, y_pred, digits=3))

# ✅ Step 6: 保存模型
joblib.dump(clf, "saved_model/sbert_model/logistic_model.pkl")
joblib.dump(scaler, "saved_model/sbert_model/scaler.pkl")

# 保存句向量模型（用 SBERT 自带）
model.save("saved_model/sbert_model")  # 会保存成文件夹形式
