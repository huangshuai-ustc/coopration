import pandas as pd
from openai import OpenAI
from ast import literal_eval
import time
from tqdm import tqdm

# 初始化 DeepSeek API
client = OpenAI(api_key="sk-73b1e9510c504a54a5eeabeadde2d51e", base_url="https://api.deepseek.com")

# 读取 CSV（这里只取前 5 条做测试）
df = pd.read_csv("abstracts_with_phrases_clustered.csv").iloc[:5, :]
phrase_column = df.columns[-2]

# 解析短语列
df["key_phrases_list"] = df[phrase_column].apply(literal_eval)

def is_phrase_relevant(phrase):
    prompt = f"""
    你是人工智能信任机制研究专家。请识别与”人类在运营决策情境中对人工智能功能的信任及其心理学机制’相关的文献，特别包括以下研究方向：信任校准、认知卸载与自动化偏误、AI可解释性、人-AI互动中的元认知、情绪反应、AI决策绩效、人类在回路中的角色、个体差异（如人格、专业水平、风险感知）、以及伦理与控制感等”。请只回答“是”或“否”，并简要说明原因。
    
    短语：{phrase}
"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是人工智能信任机制领域的研究专家。"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def filter_relevant_phrases(phrases):
    relevant = []
    for phrase in tqdm(phrases, leave=False, desc="短语判定"):
        reply = is_phrase_relevant(phrase)
        if reply.startswith("是"):
            relevant.append(phrase)
        time.sleep(1.2)
    return relevant

# 在 apply 上加进度条
tqdm.pandas(desc="处理记录")
df["relevant_phrases"] = df["key_phrases_list"].progress_apply(filter_relevant_phrases)

df.to_csv("phrases_with_deepseek_analysis.csv", index=False)
print("✅ 已保存处理结果：phrases_with_deepseek_analysis.csv")
