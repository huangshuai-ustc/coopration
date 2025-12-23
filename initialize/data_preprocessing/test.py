import pandas as pd

df = pd.read_excel('ok_data_enhancement.xlsx')

df.to_csv('ok_data_enhancement.csv', index=False)