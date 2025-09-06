import pandas as pd
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# 加载语义模型
model = SentenceTransformer('../initialize/all-MiniLM-L6-v2')

# ============ STEP 1: 加载数据 =============
df = pd.read_excel("final_phrases_filtereds.xlsx")  # 替换成你的文件名
original_phrases_col = df.columns[-2]
filtered_phrases_col = df.columns[-1]

# ============ STEP 2: 提取被手动剔除的短语 ============
manual_all = df[original_phrases_col].explode().dropna().tolist()
manual_filtered = df[filtered_phrases_col].explode().dropna().tolist()

# 找出“被剔除”的短语
manual_removed = list(set(manual_all) - set(manual_filtered))
manual_removed = [p for p in manual_removed if isinstance(p, str) and p.strip()]

# 获取“剔除风格”的向量平均
removed_embeddings = model.encode(manual_removed, convert_to_tensor=True)
removed_mean_embedding = removed_embeddings.mean(dim=0)

# ============ STEP 3: 定义规则过滤器 ============
def rule_based_filter(phrase):
    if not isinstance(phrase, str):
        return False
    phrase = phrase.lower()
    if re.search(r"\b(19\d{2}|20\d{2})\b", phrase):  # 含年份
        return True
    if re.search(r"\b(john|taylor|wiley|francis|georgia|karasek|academy|sons)\b", phrase):
        return True
    if len(phrase.split()) <= 2 and any(w in phrase for w in ['model', 'system', 'theory', 'development']):
        return True
    if "copyright" in phrase or "group" in phrase:
        return True
    return False

# ============ STEP 4: 定义语义相似度过滤器 ============
def semantic_filter(phrase, threshold=0.7):
    emb = model.encode(phrase, convert_to_tensor=True)
    sim = util.cos_sim(emb, removed_mean_embedding).item()
    return sim >= threshold

# ============ STEP 5（可选）DeepSeek判断 ============
def deepseek_judge(phrase):
    client = OpenAI(
        api_key="sk-73b1e9510c504a54a5eeabeadde2d51e",
        base_url="https://api.deepseek.com"
    )
    prompt = f"""
你是人工智能信任机制研究专家。请判断以下短语是否与“人类对人工智能的信任机制”这一研究主题相关。只回答“是”或“否”，并简要说明。

短语：{phrase}
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是人工智能信任机制研究专家"},
                {"role": "user", "content": prompt}
            ]
        )
        reply = response.choices[0].message.content.strip()
        return reply.startswith("否")
    except Exception as e:
        return False  # 如果出错就保留这个短语

# ============ STEP 6: 批量处理所有短语 ============
def filter_phrases(phrase_list):
    cleaned = []
    phrase_list = eval(phrase_list)
    for phrase in phrase_list:
        if rule_based_filter(phrase):
            continue
        elif semantic_filter(phrase) and not deepseek_judge(phrase):
            cleaned.append(phrase)
    return cleaned

# ============ STEP 7: 应用过滤器 ============
tqdm.pandas(desc="Filtering phrases")
df["auto_filtered_phrases"] = df[original_phrases_col].progress_apply(filter_phrases)

# ============ STEP 8: 保存结果 ============
df.to_excel("auto_filtered_phrases_output_deepseek.xlsx", index=False)
print("✅ 筛选完成，结果已保存为 auto_filtered_phrases_output_deepseek.xlsx")
