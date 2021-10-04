import pandas as pd
import numpy as np

df1 = pd.read_csv("data/unprocessed/train.csv")
df2 = pd.read_csv("data/unprocessed/evaluation.csv")
df3 = pd.read_csv("data/unprocessed/test.csv")

df_full = pd.concat([df1,df2,df3],axis="index").reset_index(drop="True")
print(df1.head(-1))
print(df3.head(-1))
print(df_full.head(-1))

df_full.to_csv("combined_data_from_labs.csv")
