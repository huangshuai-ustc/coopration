import numpy as np
import pandas as pd

df = pd.read_csv("predictions.csv")
# df = df.iloc[:10]
df["Probability"] = df["Probability"].apply(lambda x: round(float(x), 4))
# df.to_csv("prediction.csv", index=False)
filtered_df = df[(df['Prediction'] == 'ok') & (df['Probability'] >= 0.9997)]
filtered_df.to_csv("filtered_prediction(1096).csv", index=False)
# print(filtered_df)
print(filtered_df.shape)
