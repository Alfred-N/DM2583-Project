import numpy as np
import pandas as pd
import requests
import os
import json
from datetime import datetime
import time

API_key = "YI2n5kzKUY5hl5vbf7r5tOsAJ"
API_key_secret = "OKEJ2SVpCmlo2j6uUhEWHSoKhIC3ExEOjZsq8z3qtJMHKfp91H"
bearer_token = "AAAAAAAAAAAAAAAAAAAAAPz3UQEAAAAAMc5NQE9ySRNInb8vkIXUbsVaoAY%3DBHbaaTJAqYkZ1OquUCFAdC2tCG4H46BtPPnOktPY41xn9uDV3p"



# dates = pd.DataFrame(columns={"start":[],"end":[]})
# dates["start"] = pd.date_range(start="2021-10-08",end="2021-10-10",freq="M") + pd.DateOffset(days=8)
# dates["end"] = pd.date_range(start="2021-10-10",end="2021-10-10",freq="M") + pd.DateOffset(days=10)
dates = pd.DataFrame(data={"start":pd.to_datetime(["2021-10-08"]),"end":pd.to_datetime(["2021-10-10"])})
# dates.loc[len(dates)-1,"end"] = datetime.now()

dates["start"] = dates["start"].dt.strftime('%Y-%m-%d')
dates["end"] = dates["end"].dt.strftime('%Y-%m-%d')
dates["start"] += "T00:00:00Z"
dates["end"] += "T00:00:00Z"


search_url = "https://api.twitter.com/2/tweets/search/all"

# Optional params: start_time,end_time,since_id,until_id,max_results,next_token,
# expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields

def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2FullArchiveSearchPython"
    return r


def connect_to_endpoint(url, params):
    response = requests.request("GET", search_url, auth=bearer_oauth, params=params)
    print(response.status_code)
    if response.status_code != 200:
        print("Request failed:", response.status_code, response.text)
        # raise Exception(response.status_code, response.text)
    return response.json(), response.status_code

def make_request(start_date,end_date,next_token):
    query_params = {'query': '(#ParisAgreement OR #ParisAccords OR #ParisClimateAccords OR "Paris Agreement" '
                                +'OR "Paris Accords" OR "Paris Climate Accords" OR "Paris Climate Agreement") lang:en -is:retweet'
                                ,'tweet.fields': 'created_at'
                                ,'start_time': start_date
                                ,'end_time': end_date
                                ,'max_results': 500
                                ,'next_token': next_token
                                }
    json_response, status_code = connect_to_endpoint(search_url, query_params)
    meta_dict = json_response['meta']
    return meta_dict,json_response, status_code


def main():
    twitter_df = pd.read_csv("Twitter_Paris_Agreement.csv")
    counter = 0
    fail_limit=10
    for i in range(len(dates)):
        next_token=None
        start = dates.loc[i,"start"]
        end = dates.loc[i,"end"]
        print("Date: ", start)
        fail_counter=0
        while True:
            time.sleep(4)
            meta_dict, json_response, status_code = make_request(start,end,next_token)

            if status_code!=200: #Detect error
                if status_code==429:
                    time.sleep(15*60)
                else:
                    fail_counter+=1
                    print(f"Fail counter = {fail_counter}")
            else: #If there is no error
                counter+=int(meta_dict['result_count'])
                data_dict = json_response['data']
                response_df = pd.json_normalize(data=data_dict)
                response_df["created_at"] = pd.to_datetime(response_df["created_at"])
                response_df.sort_values(by="created_at", inplace=True)
                print(response_df.head(-1))
                print(f"Tweet counter = {counter}")
                twitter_df = twitter_df.append(response_df, ignore_index=True)
                twitter_df.to_csv("Twitter_Paris_Agreement.csv",index=False)

                if 'next_token' in meta_dict.keys():
                    next_token = meta_dict['next_token']
                else: #break if all available pages are read
                    break

            if fail_counter>=fail_limit:
                print(f"Fail limit of {fail_limit} reached.")
                quit()

if __name__ == "__main__":
    main()