from numpy.lib.npyio import save
from models.model_api import ModelInterface
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NaiveBayes(ModelInterface):
    model: None
    vectorizer: None
    C: float
    train_arr: np.ndarray
    val_arr: np.ndarray
    test_arr: np.ndarray

    def __init__(self, train_data, val_data, test_data, C=1):
        super(NaiveBayes, self).__init__(train_data, val_data, test_data)
        self.C=C

    def train(self, n_epochs=50, CV=False, verbose=False):
        train_accuracy=None
        val_accuracy=None
        train_confusion_matrix = None
        val_confusion_matrix = None
        return train_accuracy, val_accuracy, train_confusion_matrix, val_confusion_matrix

    def test(self):
        pass
    def eval(self):
        test_loss=None
        test_accuracy=None
        test_confusion_matrix = None
        return test_loss, test_accuracy, test_confusion_matrix
    
    def classify_sentiment(self, unlabelled_data):
        pass