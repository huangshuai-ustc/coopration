import pandas as pd
from tqdm import tqdm
from keybert import KeyBERT
import spacy
from sentence_transformers import SentenceTransformer
tqdm.pandas()


from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def jaccard_similarity(a, b):
    a_set, b_set = set(a.split()), set(b.split())
    intersection = a_set & b_set
    union = a_set | b_set
    if not union:
        return 0
    return len(intersection) / len(union)

def filter_similar_phrases(phrases, threshold=0.8):
    filtered = []
    for i, phrase in enumerate(phrases):
        keep = True
        for selected in filtered:
            if jaccard_similarity(phrase, selected) >= threshold:
                keep = False
                break
        if keep:
            filtered.append(phrase)
    return filtered

# -------------------------------
# 1. 加载模型
# -------------------------------
# ✅ 加载 SciBERT 封装版本，用于 KeyBERT
model = SentenceTransformer("./model/scibert-nli")
kw_model = KeyBERT(model)

# ✅ 加载 spaCy 英文模型，用于 POS 分析
nlp = spacy.load("en_core_web_sm")

# 定义实词 POS 类型
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}


# -------------------------------
# 2. 定义函数：计算实词数
# -------------------------------
def count_content_words(phrase):
    doc = nlp(phrase)
    return sum(1 for token in doc if token.pos_ in CONTENT_POS)


# -------------------------------
# 3. 定义函数：提取关键词并过滤
# -------------------------------
def extract_filtered_phrases(text):
    if pd.isnull(text):
        return []
    # 只保留包含3~5个实词的短语
    filtered = []
    filter_redundant = []
    text = text.lower().split('.')
    for txt in text:
        keywords = kw_model.extract_keywords(
            txt,  # 防止长文本拖慢模型
            keyphrase_ngram_range=(2, 5),
            stop_words='english',
            use_mmr=True,
            diversity=0.3,
            nr_candidates=20,
            top_n=8
        )

        for phrase, _ in keywords:
            if 2 <= count_content_words(phrase) <= 5:
                filtered.append(phrase)
    filter_redundant = filter_similar_phrases(filtered)
    return filter_redundant

# -------------------------------
# 4. 加载 CSV 文件，应用处理
# -------------------------------
df = pd.read_csv("../initialize/filtered_prediction(1096).csv")  # 👈 替换为你的文件路径

df["key_phrases"] = df["Abstract"].progress_apply(extract_filtered_phrases)

# -------------------------------
# 5. 保存结果
# -------------------------------
df.to_csv("abstracts_with_phrases_filtered.csv", index=False)
print("✅ 关键词提取完成，文件已保存为 'abstracts_with_phrases_filtered.csv'")
