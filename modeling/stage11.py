# 安装所需库（第一次运行时执行）
# pip install keybert sentence-transformers scikit-learn numpy
from cv2 import phase
from keybert import KeyBERT
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from tqdm import tqdm, trange

tqdm.pandas()

# 初始化模型
kw_model = KeyBERT(model='all-MiniLM-L6-v2')
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv("../initialize/filtered_prediction(1096).csv")  # 确保数据已加载
def deal(text):
    phrases = []
    for txt in text.split("."):
        raw_keywords = kw_model.extract_keywords(
            txt,
            keyphrase_ngram_range=(2, 5),
            stop_words='english',
            use_maxsum=True,
            nr_candidates=16,
            top_n=8
        )
        for kw in raw_keywords:
            phrases.append(kw[0])
    embeddings = embedding_model.encode(phrases)
    sim_matrix = cosine_similarity(embeddings)
    selected = []
    used = set()

    for i, phrase in enumerate(phrases):
        if i in used:
            continue
        selected.append(phrase)
        for j in range(i + 1, len(phrases)):
            if sim_matrix[i][j] >= 0.5:
                used.add(j)
    return selected

res = []
for i in trange(len(df)):
    res.append(deal(df.loc[i, 'Abstract']))

df['key_phrases'] = res
df.to_csv('abstracts_with_phrases_filtered(1).csv', index=False)

# print(res)