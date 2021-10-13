from numpy.lib.npyio import save
from models.model_api import ModelInterface
from helper_functions import plot_confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

class NaiveBayes(ModelInterface):
    model: MultinomialNB
    vectorizer: TfidfVectorizer
    C: float
    train_arr: np.ndarray
    val_arr: np.ndarray
    test_arr: np.ndarray

    def __init__(self, train_data, val_data, test_data, C=1):
        super(NaiveBayes, self).__init__(train_data, val_data, test_data)
        self.C=C
        self.model = MultinomialNB()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        print("Vectorizing ...")
        self.vectorize()

    def vectorize(self):
        self.train_arr = self.vectorizer.fit_transform(self.train_df["text"]).toarray()
        self.val_arr = self.vectorizer.transform(self.val_df["text"]).toarray()
        self.test_arr = self.vectorizer.transform(self.test_df["text"]).toarray()

    def train(self, n_epochs=50, CV=False, verbose=False, plot_matrix=False):
        print("Training ...")
        self.model.fit(self.train_arr,self.train_df["score"])
        train_acc = self.model.score(self.train_arr, self.train_df["score"].values)
        val_acc = self.model.score(self.val_arr, self.val_df["score"].values)

        if plot_matrix:
          confusion_matrix = plot_confusion_matrix(self.val_df["score"], 
          self.model.predict(self.val_arr), 
          title="Confusion matrix of sentiments predicted by NBC", 
          path="results/nbc/confusion_eval.png")
          confusion_matrix.show()

        return train_acc, val_acc

    def test(self, plot_matrix=False):
        if plot_matrix:
          confusion_matrix = plot_confusion_matrix(self.test_df["score"], 
          self.model.predict(self.test_arr), 
          title="Confusion matrix of sentiments predicted by NBC", 
          path="results/nbc/confusion_test.png")
          confusion_matrix.show()

        print("Testing model on the test set ...")
        predictions = self.model.predict(self.test_arr)
        test_acc = self.model.score(self.test_arr,self.test_df["score"])
        return predictions, test_acc

    def eval(self):
        test_loss=None
        test_accuracy=None
        test_confusion_matrix = None
        return test_loss, test_accuracy, test_confusion_matrix
    
    def classify_sentiment(self, unlabelled_data, save_csv=False):
        print("Vectorizing 1M tweets")
        tweet_arr = self.vectorizer.transform(unlabelled_data["text"]).toarray()
        print("Predicting sentiments of 1M tweets ...")
        new_predictions = self.model.predict(tweet_arr)
        sentiment_df=unlabelled_data.loc[:,["created_at","id","text"]]
        sentiment_df["score"] = new_predictions
        print(sentiment_df.head(-1))
        if save_csv:
            print("Saving predicted sentiments ...")
            time = datetime.now().strftime("%Y%m%d_%H%M")
            save_file=f"results/nbc/Tweet_sentiments_" +  time + ".csv"
            try:
                mode = 'a' if os.path.exists(save_file) else 'wb'
                with open(save_file,mode) as f:
                    sentiment_df.to_csv(save_file)
            except IOError as e:
                raise Exception("Failed to save to %s: %s" % (save_file, e))

        return sentiment_df

