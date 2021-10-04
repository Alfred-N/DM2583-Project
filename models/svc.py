from models.model_api import ModelInterface
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC


class SVC(ModelInterface):
    model: LinearSVC
    vectorizer: TfidfVectorizer
    C: float
    train_arr: np.ndarray
    val_arr: np.ndarray
    test_arr: np.ndarray

    def __init__(self, train_data, val_data, test_data, C=1):
        super(SVC, self).__init__(train_data, val_data, test_data)
        self.C = C
        self.model = LinearSVC(C=self.C)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        print("Vectorizing")
        self.vectorize()


    def vectorize(self):
        # self.train_arr = self.vectorizer()
        pass

    def train(self):
        pass

    def classify_sentiment(self, unlabelled_data):
        pass

