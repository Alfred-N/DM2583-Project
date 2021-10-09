from pandas.core.reshape.concat import concat
import requests
import os
import json
import pandas as pd
from datetime import datetime

bearer_token = "AAAAAAAAAAAAAAAAAAAAAPz3UQEAAAAAMc5NQE9ySRNInb8vkIXUbsVaoAY%3DBHbaaTJAqYkZ1OquUCFAdC2tCG4H46BtPPnOktPY41xn9uDV3p"

search_url = "https://api.twitter.com/2/tweets/counts/all"

dates = pd.DataFrame(columns={"start":[],"end":[]})
dates["start"] = pd.date_range(start="2019-11-11",end="2021-09-11",freq="M") + pd.DateOffset(days=11)
dates["end"] = pd.date_range(start="2019-12-11",end="2021-10-11",freq="M") + pd.DateOffset(days=11)
dates.loc[len(dates)-1,"end"] = datetime.now()

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
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


def main():
    counter = 0
    for i in range(len(dates)):
        start = dates.loc[i,"start"]
        end = dates.loc[i,"end"]
        print(start,end)
        green_deal_query = {'query': '(#EUGreenDeal OR "European Green Deal" OR "EU Green Deal" OR "EUGreenDeal"'
                        +'OR "EU\'s Green Deal" OR "The European Union\'s Green Deal") lang:en'
                        ,'granularity': 'day'
                        ,'start_time': start
                        ,'end_time': end
                        }
        paris_query = {'query': '(#ParisAgreement OR #ParisAccords OR #ParisClimateAccords OR "Paris Agreement" '
                        +'OR "Paris Accords" OR "Paris Climate Accords") lang:en'
                        ,'granularity': 'day'
                        ,'start_time': start
                        ,'end_time': end
                        }
        
        json_response = connect_to_endpoint(search_url, paris_query)
        meta_dict = json_response["meta"]
        counter += int(meta_dict["total_tweet_count"])
        print("Total tweets = ", counter)

if __name__ == "__main__":
    main()