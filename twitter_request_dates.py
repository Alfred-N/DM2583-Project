import pandas as pd
from datetime import datetime

dates = pd.DataFrame(columns={"start":[],"end":[]})
dates["start"] = pd.date_range(start="2019-11-11",end="2021-09-11",freq="M") + pd.DateOffset(days=11)
dates["end"] = pd.date_range(start="2019-12-11",end="2021-10-11",freq="M") + pd.DateOffset(days=11)
dates.loc[len(dates)-1,"end"] = datetime.now()

dates["start"] = dates["start"].dt.strftime('%Y-%m-%d')
dates["end"] = dates["end"].dt.strftime('%Y-%m-%d')
dates["start"] += "T00:00:00Z"
dates["end"] += "T00:00:00Z"

for i in range(len(dates)):
    start = dates.loc[i,"start"]
    end = dates.loc[i,"end"]
    print(start,end)