import pandas as pd
import numpy as np

twitter2015_to_2017 = pd.read_csv("Twitter_Paris_Agreement_2015_to_2017.csv")
twitter2015_to_2017["created_at"] = pd.to_datetime(twitter2015_to_2017["created_at"])
twitter2015_to_2017.sort_values(by="created_at").reset_index(drop=True)
twitter2015_to_2017 = twitter2015_to_2017.loc[:, ["created_at","id","text"]]
len1=len(twitter2015_to_2017.index.values)


twitter2017_to_2021 = pd.read_csv("Twitter_Paris_Agreement_2017_to_2021.csv")
twitter2017_to_2021["created_at"] = pd.to_datetime(twitter2017_to_2021["created_at"])
twitter2017_to_2021.sort_values(by="created_at").reset_index(drop=True)
twitter2017_to_2021 = twitter2017_to_2021.loc[:, ["created_at","id","text"]]
first_date = twitter2017_to_2021["created_at"].iloc[0]
last_date = twitter2017_to_2021["created_at"].iloc[-1]
# print(first_date,last_date)
len2=len(twitter2017_to_2021)
# print(twitter2017_to_2021.head(-1))

twitter_2019_to_2021 = pd.read_csv("Twitter_Paris_Agreement_2019-11-11_to_2021-10-11.csv")
twitter_2019_to_2021["created_at"] = pd.to_datetime(twitter_2019_to_2021["created_at"])
twitter_2019_to_2021.sort_values(by="created_at").reset_index(drop=True)
twitter_2019_to_2021 = twitter_2019_to_2021.loc[:, ["created_at","id","text"]]
twitter_2021 = twitter_2019_to_2021[lambda x: x.created_at > "2021-01-18"]
len3=len(twitter_2021.index.values)
print(len1+len2+len3)
# print(twitter_2021.head())

full_df = pd.concat([twitter2015_to_2017,twitter2017_to_2021, twitter_2021])
full_df = full_df.sort_values(by="created_at").reset_index(drop=True)
unique_ids = full_df["created_at"].unique()
print(len(unique_ids))
full_df = full_df.drop_duplicates().reset_index(drop=True)
print(full_df.head(-1))