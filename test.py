import pandas as pd
import numpy as np

from models.sequence_dataset import SequenceDataset
from data_processor import DataProcessor

processor = DataProcessor(filename="./data/unprocessed/combined_data_from_labs.csv")
processor.load_csv()
train,val,test= processor.split_data(frac={"train": 0.6,"val": 0.2,"test": 0.2 })
X_train = train["text"].values
print('train_x shape is {}' .format({X_train.shape}))
y_train = train["score"].values

for idx in range(10):
    print(SequenceDataset(X_train, y_train).__getitem__(idx)[1])
    print(train.loc[idx,"score"])
