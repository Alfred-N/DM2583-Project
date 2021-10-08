import pandas as pd
import numpy as np

from models.sequence_dataset import SequenceDataset
from data_processor import DataProcessor

processor = DataProcessor(filename="data/unprocessed/US_Airlines_Tweets.csv")
processor.preprocess_US_Airlines_data()
processor.get_onehot_encoding()
print(processor.full_dataset.head(-1))
processor.full_dataset.to_csv("data/processed/US_Airlines_Tweets.csv")

