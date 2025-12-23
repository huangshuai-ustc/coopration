import json
import time
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures
import threading
from queue import Queue

# 读取数据
df = pd.read_csv("../initialize/filtered_prediction(1096).csv")

# 创建新列来存储结果
df["classification"] = ""
df["levels"] = ""
df["reason"] = ""
df["brief"] = ""

client = OpenAI(
    api_key="sk-73b1e9510c504a54a5eeabeadde2d51e",  # ⚠️ 替换为你自己的
    base_url="https://api.deepseek.com"
)


def req_v3(promptxxxxx):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": promptxxxxx}]
        )
        reply = response.choices[0].message.content.strip()
        return reply
    except Exception as e:
        print("⚠️ DeepSeek API 调用失败:", e)
        return "Unknown"


def get_prompt(title, abstract, keywords, prompt_info):
    """ get_prompt """
    if not isinstance(keywords, str):
        keywords = ""
    prompt_tmp = prompt_info
    prompt_tmp = prompt_tmp.replace("<TITLE>", title)
    prompt_tmp = prompt_tmp.replace("<ABSTRACT>", abstract)
    prompt_tmp = prompt_tmp.replace("<KEYWORDS>", keywords)
    return prompt_tmp


def make_offline_input(title, abstract, keywords, promptxxx):
    prompt_infos = get_prompt(title, abstract, keywords, promptxxx)
    req_dict = {
        "request_body": {
            "system": "",
            "messages": [
                {
                    "role": "user",
                    "content": prompt_infos
                }
            ],
            "top_p": 0.8,
            "temperature": 1,
            "penalty_score": 1
        }
    }

    out_list = json.dumps(req_dict, ensure_ascii=False)
    return out_list


# 处理单个数据项的函数
def process_item(args):
    i, row, prompts = args
    title_info = row["Title"]
    abstract_info = row["Abstract"]
    keywords_info = row["Keywords"]

    # 生成提示
    prompt = get_prompt(title_info, abstract_info, keywords_info, prompts)

    # 调用API
    result = req_v3(prompt)

    # 尝试解析JSON结果
    try:
        result_data = json.loads(result.replace("json", "").replace("`", '').strip())
        classification = result_data.get("classification", "Unknown")
        levels = result_data.get("levels", "Unknown")
        reason = result_data.get("reason", "Unknown")
        brief = result_data.get("brief", "Unknown")
        result = result
    except json.JSONDecodeError:
        # 如果结果不是JSON格式，直接存储原始结果
        classification = "Parse Error"
        levels = "Parse Error"
        reason = result
        brief = "Parse Error"
        result = result

    return i, classification, levels, reason, brief, result


# 读取提示模板
with open("v4_full.txt", 'r', encoding='utf-8') as f:
    prompts = f.read()

# 准备任务参数
tasks = [(i, row, prompts) for i, row in df.iterrows()]

# 使用线程池处理数据，但保持顺序
max_workers = 10  # 根据API限制调整线程数

# 方法1: 使用map保持顺序
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    results = list(tqdm(executor.map(process_item, tasks), total=len(tasks)))

# 按原始顺序更新DataFrame
for i, classification, levels, reason, brief, result in results:
    df.at[i, "classification"] = classification
    df.at[i, "levels"] = levels
    df.at[i, "reason"] = reason
    df.at[i, "brief"] = brief
    df.at[i, "result"] = result  # 标记为已处理

    # 每处理100条记录保存一次
    if (i + 1) % 100 == 0:
        df.to_csv("53_100_classification.csv", index=False)
        print(f"已保存前 {i + 1} 条结果")

# 最终保存
df.to_csv("53_full_classification.csv", index=False)
print("处理完成，结果已保存到 53_full_classification.csv")