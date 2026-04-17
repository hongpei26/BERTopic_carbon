import pandas as pd

file_path = "/home/carbon/carbon/data/part-000000000000.parquet"

df = pd.read_parquet(file_path)

print("欄位名稱：")
print(df.columns.tolist())

print("\n前 5 筆資料：")
print(df.head())
