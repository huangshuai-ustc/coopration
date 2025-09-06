import json
import time
import pandas as pd
from openai import OpenAI
from tqdm import trange

# 读取数据
df = pd.read_csv("53_full_classification.csv")

for i in trange(len(df)):
    if df.at[i, "classification"] == "Parse Error":
        result = df.at[i, "reason"]
        result = result.replace("json", "").replace("`", '')
        print(result)
        result = json.loads(result)

        df.at[i, "classification"] = result.get("classification", "Unknown")
        df.at[i, "levels"] = result.get("levels", "Unknown")
        df.at[i, "reason"] = result.get("reason", "Unknown")
        df.at[i, "brief"] = result.get("brief", "Unknown")


df.to_csv("53_full_classificationxxxx.csv", index=False)

