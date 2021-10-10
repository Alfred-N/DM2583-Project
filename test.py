import numpy as np
import pandas as pd

df1 = pd.read_csv("Twitter_2015_to_2021.csv")
df1 = df1.loc[:, ["created_at","id","text"]]
df2 = pd.read_csv("Twitter_Paris_Agreement.csv")
df2 = df2.loc[:, ["created_at","id","text"]]
full_df = pd.concat([df1,df2])
full_df = full_df.sort_values(by="created_at").reset_index(drop=True)

full_df = full_df.drop_duplicates().reset_index(drop=True)
unique_ids = full_df["id"].unique()
print(len(unique_ids))
print(full_df.head(-1))