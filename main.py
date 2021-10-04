import pandas as pd
import numpy as np
from data_processor import DataProcessor
from models.svc import SVC

processor = DataProcessor(filename="./data/unprocessed/test_data.csv")
processor.load_csv()
train,val,test= processor.split_data(frac={ "train": 0.6,"val": 0.2,"test": 0.2 })

model = SVC(train,val,test)
print(model)