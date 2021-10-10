from pandas.core.reshape.concat import concat
import requests
import os
import json
import pandas as pd
from datetime import datetime
import time
bearer_token = "AAAAAAAAAAAAAAAAAAAAAPz3UQEAAAAAMc5NQE9ySRNInb8vkIXUbsVaoAY%3DBHbaaTJAqYkZ1OquUCFAdC2tCG4H46BtPPnOktPY41xn9uDV3p"

search_url = "https://api.twitter.com/2/tweets/counts/all"

dates = pd.DataFrame(columns={"start":[],"end":[]})
# for greendeal:
dates["start"] = pd.date_range(start="2019-11-11",end="2021-09-11",freq="M") + pd.DateOffset(days=11)
dates["end"] = pd.date_range(start="2019-12-11",end="2021-10-11",freq="M") + pd.DateOffset(days=11)
dates.loc[len(dates)-1,"end"] = datetime.now()

# #for paris agreement:
# dates["start"] = pd.date_range(start="2017-06-11",end="2021-09-11",freq="M") 
# dates["end"] = pd.date_range(start="2017-07-11",end="2021-10-11",freq="M") 

dates["start"] = dates["start"].dt.strftime('%Y-%m-%d')
dates["end"] = dates["end"].dt.strftime('%Y-%m-%d')
dates["start"] += "T00:00:00Z"
dates["end"] += "T00:00:00Z"



def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2FullArchiveTweetCountsPython"
    return r


def connect_to_endpoint(url, params):
    response = requests.request("GET", search_url, auth=bearer_oauth, params=params)
    # print(response.status_code)
    if response.status_code != 200:
        print("Request failed:", response.status_code, response.text)
        # raise Exception(response.status_code, response.text)
        return None, response.status_code
    else:
        return response.json(), response.status_code

def make_request(start_date,end_date,next_token):
    query_params = {'query': '(#EUGreenDeal OR "European Green Deal" OR "EU Green Deal" OR "EUGreenDeal"'
                        +'OR "EU\'s Green Deal" OR "The European Union\'s Green Deal") lang:en '
                        ,'granularity': 'day'
                        ,'start_time': start_date
                        ,'end_time': end_date
                        ,'next_token': next_token
                        }
    json_response, status_code = connect_to_endpoint(search_url, query_params)
    if json_response is not None:
        meta_dict = json_response['meta']
        return meta_dict,json_response, status_code
    else:
        return None, None, status_code


def main():
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
                    print("Too many requests, sleeping 15 min")
                else:
                    fail_counter+=1
                    print(f"Fail counter = {fail_counter}")
            else: #If there is no error
                counter+=int(meta_dict['total_tweet_count'])
                print(f"Tweet counter = {counter}")
                if 'next_token' in meta_dict.keys():
                    print("Next token: ",meta_dict['next_token'])
                    next_token = meta_dict['next_token']
                else: #break if all available pages are read
                    break

            if fail_counter>=fail_limit:
                print(f"Fail limit of {fail_limit} reached.")
                quit()

        if counter >= 1e6:
            break

if __name__ == "__main__":
    main()