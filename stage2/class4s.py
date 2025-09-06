import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm, trange

# ========== 配置部分 ==========
client = OpenAI(
    api_key="sk-73b1e9510c504a54a5eeabeadde2d51e",   # ⚠️ 替换为你自己的 key
    base_url="https://api.deepseek.com"
)

# 输入文件 & 输出文件
input_file = "../classify/clustered_phrases_bottom2up.xlsx"
output_file = "classified_summary.xlsx"

# ========== API 调用函数 ==========
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

# ========== 读取数据 ==========
df = pd.read_excel(input_file)

# 假设文件列名为：短语列: Phrase, 一级分类: Level1, 二级分类: Level2, 三级分类: Level3
phrase_col = df.columns[1]
level1_col = df.columns[2]
level2_col = df.columns[3]
level3_col = df.columns[4]

# 新列
for col in ["level1_summary", "level2_summary", "level3_summary"]:
    if col not in df.columns:
        df[col] = ""

# ========== 定义提示词 ==========
prompt_template = """请根据给定短语列表执行以下任务:
1. **内容总结**: 给出一个短语，精准概括给定短语列表的中心思想，三到五个英文单词。

2. **备注**:
   - 给出的总结短语千万不可以过长，保持三到五个英文单词即可。
   - 请自行总结分类词汇，要求需要贴合输入的短语列表。

**输出要求**:
- 以JSON格式输出,输出在一行内
- 包含三个字段:
  "summary": 一个英文单词组成的字符串,
  "reason": 一个英文单词组成的字符串，说明总结出summary的原因,
  "brief": 对输入短语的简单概括。

**输入**:
短语列表:{phrases}
"""

# ========== 分级汇总 ==========
# 一级分类汇总
level1_summary = {}
for lv1, group in tqdm(df.groupby(level1_col), desc="Processing Level1 summaries"):
    phrases = group[phrase_col].dropna().tolist()
    prompt = prompt_template.format(phrases=phrases)
    result = req_v3(prompt)
    level1_summary[lv1] = result

# 二级分类汇总
level2_summary = {}
for (lv1, lv2), group in tqdm(df.groupby([level1_col, level2_col]), desc="Processing Level2 summaries"):
    phrases = group[phrase_col].dropna().tolist()
    prompt = prompt_template.format(phrases=phrases)
    result = req_v3(prompt)
    level2_summary[(lv1, lv2)] = result

# 三级分类汇总
level3_summary = {}
for (lv1, lv2, lv3), group in tqdm(df.groupby([level1_col, level2_col, level3_col]), desc="Processing Level3 summaries"):
    phrases = group[phrase_col].dropna().tolist()
    prompt = prompt_template.format(phrases=phrases)
    result = req_v3(prompt)
    level3_summary[(lv1, lv2, lv3)] = result

# ========== 写回结果 ==========
for i in trange(len(df), desc="Writing back results"):
    lv1 = df.iloc[i][level1_col]
    lv2 = df.iloc[i][level2_col]
    lv3 = df.iloc[i][level3_col]

    if pd.notna(lv1) and lv1 in level1_summary:
        df.at[i, "level1_summary"] = level1_summary[lv1]
    if pd.notna(lv2) and (lv1, lv2) in level2_summary:
        df.at[i, "level2_summary"] = level2_summary[(lv1, lv2)]
    if pd.notna(lv3) and (lv1, lv2, lv3) in level3_summary:
        df.at[i, "level3_summary"] = level3_summary[(lv1, lv2, lv3)]

# 保存结果
df.to_excel(output_file, index=False)
print(f"✅ 处理完成，结果已保存到 {output_file}")
