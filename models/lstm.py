from models.model_api import ModelInterface
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import torch

class LSTM(ModelInterface):
    model: None
    vectorizer: CountVectorizer
    train_arr: np.ndarray
    val_arr: np.ndarray
    test_arr: np.ndarray

    def __init__(self, train_data, val_data, test_data):
        super(LSTM, self).__init__(train_data, val_data, test_data)
        self.vectorizer = CountVectorizer(stop_words='english', max_features=5)

        # self.countWords()
        print("Vectorizing ...")
        self.vectorize()

    def vectorize(self):
        print(self.train_df["text"].apply(lambda x: len(x)).max())
        self.train_arr = self.vectorizer.fit_transform(self.train_df["text"]).toarray()
        self.val_arr = self.vectorizer.transform(self.val_df["text"]).toarray()
        self.test_arr = self.vectorizer.transform(self.test_df["text"]).toarray()
        counts = pd.DataFrame(self.train_arr)
        print(counts.head(-1))
        pass
    

    def train(self, n_epochs=1, CV=False, verbose=False):
        pass

    def test(self):
        test_loss=None
        test_accuracy=None
        test_confusion_matrix = None
        return test_loss, test_accuracy, test_confusion_matrix
    
    def classify_sentiment(self, unlabelled_data):
        pass

    def explode_strings(self, data, save_TSDS=True):
        data = data.loc[:]
        dataset_len = data.shape[0]
        series = pd.Series(data["text"], dtype="string")
        series = series.str.replace("[<][a-zA-Z]+ [/][>]+", "", case=False, regex=True)
        # normalize string
        series = series.str.lower()
        series = series.str.findall("[a-zA-Z]+")
        data.loc[:,"text"] = series
        data.loc[:,"time_idx"] = data.loc[:,"text"].apply(lambda x: np.arange(0, len(x),1))
        data.loc[:,"ID"] = data.index.values
        data = data.explode(column=["text","time_idx"])
        # print(data.head(-1))
        data = data.reset_index(drop=True)
        # print(data.head(-1))
        data["time_idx"] = data["time_idx"].astype(dtype=np.int64)
        return data