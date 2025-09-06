import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from ast import literal_eval
from tqdm import tqdm
import time
import json
import re

# ---------- CONFIG ----------
DEEPSEEK_API_KEY = "sk-73b1e9510c504a54a5eeabeadde2d51e"  # ← 替换成你的 key
INPUT_CSV = "abstracts_with_phrases_clustered.csv"
OUTPUT_CSV = "final_phrases_filtered.csv"
SIMILARITY_THRESHOLD = 0.5
EMBEDDING_MODEL_NAME = "../initialize/all-MiniLM-L6-v2"
TOPIC_SEEDS = [
    "trust in AI", "human-AI trust", "cognitive offloading",
    "perceived control", "explainability", "AI transparency"
]
# ----------------------------

# Initialize clients
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Load CSV
df = pd.read_csv(INPUT_CSV).iloc[:, :]  # For testing, only take the first 5 rows
phrase_column = df.columns[-2]
df["key_phrases_list"] = df[phrase_column].apply(literal_eval)

# Compute topic center embedding
topic_embeddings = model.encode(TOPIC_SEEDS, convert_to_tensor=True)
topic_center = topic_embeddings.mean(dim=0)

def filter_by_similarity(phrases, threshold=SIMILARITY_THRESHOLD):
    phrase_embeddings = model.encode(phrases, convert_to_tensor=True)
    similarities = util.cos_sim(phrase_embeddings, topic_center)
    return [phrases[i] for i in range(len(phrases)) if similarities[i] >= threshold]

def classify_with_deepseek(phrases):
    prompt = (
        "你是人工智能信任机制研究专家。以下是一些短语，请识别与”人类在运营决策情境中对人工智能功能的信任及其心理学机制’相关的文献，特别包括以下研究方向：信任校准、认知卸载与自动化偏误、AI可解释性、人-AI互动中的元认知、情绪反应、AI决策绩效、人类在回路中的角色、个体差异（如人格、专业水平、风险感知）、以及伦理与控制感等”\n"
        "请用 JSON 格式返回：每个短语为键，值为“是”或“否”。\n\n"
        "短语列表：\n" + "\n".join(f"- {p}" for p in phrases)
    )
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是人工智能信任机制领域的研究专家。"},
                {"role": "user", "content": prompt}
            ]
        )
        reply = response.choices[0].message.content.strip()
        match = re.search(r'\{.*\}', reply, re.DOTALL)
        if match:
            result_dict = json.loads(match.group())
            return [p for p in phrases if result_dict.get(p, "否").startswith("是")]
        else:
            return []
    except Exception as e:
        print("⚠️ ERROR:", e)
        return []

# 执行
tqdm.pandas(desc="处理记录")

def process_record(phrases):
    similar_phrases = filter_by_similarity(phrases)
    if not similar_phrases:
        return []
    time.sleep(1.2)
    return classify_with_deepseek(similar_phrases)

df["relevant_phrases"] = df["key_phrases_list"].progress_apply(process_record)
df.to_csv(OUTPUT_CSV, index=False)
print("✅ 已保存至", OUTPUT_CSV)
