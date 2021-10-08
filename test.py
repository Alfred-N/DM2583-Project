import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.sequence_dataset import SequenceDataset
from data_processor import DataProcessor
from helper_functions import plot_training_result, print_sentiment_distribution

processor = DataProcessor(filename="data/processed/US_Airlines_Tweets.csv")
processor.preprocess_US_Airlines_data()
processor.get_onehot_encoding()
processor.get_even_distribution()
print_sentiment_distribution(processor.full_dataset,plot=False)
processor.full_dataset.to_csv("data/processed/US_Airlines_Tweets_EVEN_DISTRIB.csv")

