import pandas as pd
from openai import OpenAI
from ast import literal_eval
import time
from tqdm import tqdm

# 初始化 DeepSeek API
client = OpenAI(api_key="sk-73b1e9510c504a54a5eeabeadde2d51e", base_url="https://api.deepseek.com")

# 读取 CSV（这里只取前 5 条做测试）
df = pd.read_csv("abstracts_with_phrases_clustered.csv").iloc[:, :]
phrase_column = df.columns[-2]

# 解析短语列
df["key_phrases_list"] = df[phrase_column].apply(literal_eval)


def is_batch_relevant(phrases):
    # 构建批量 prompt
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

        # 简单处理 JSON 响应
        import json, re
        json_str = re.search(r'\{.*\}', reply, re.DOTALL)
        if json_str:
            result_dict = json.loads(json_str.group())
            return [p for p in phrases if result_dict.get(p, "否").startswith("是")]
        else:
            return []
    except Exception as e:
        print("⚠️ ERROR:", e)
        return []



# 在 apply 上加进度条
tqdm.pandas(desc="批量处理记录")
df["relevant_phrases"] = df["key_phrases_list"].progress_apply(is_batch_relevant)


df.to_csv("phrases_with_deepseek_analysis(batch).csv", index=False)
print("✅ 已保存处理结果：phrases_with_deepseek_analysis(batch).csv")
