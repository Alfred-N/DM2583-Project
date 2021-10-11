from numpy.lib.npyio import save
from torch._C import unify_type_list
from models.model_api import ModelInterface
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
import pickle
from datetime import datetime
import os

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
        print("Vectorizing ...")
        self.vectorize()

    def load_from_pickle(self, file):
        try:
            with open(file,'rb') as f:
                saved_model = pickle.load(file=f)
        except IOError as e:
            raise Exception("Failed to load from %s: %s" % (file, e))
        self.model = saved_model

    def vectorize(self):
        self.train_arr = self.vectorizer.fit_transform(self.train_df["text"]).toarray()
        self.val_arr = self.vectorizer.transform(self.val_df["text"]).toarray()
        self.test_arr = self.vectorizer.transform(self.test_df["text"]).toarray()

    def train(self, n_epochs=50, CV=False, verbose=0, save_model=False):
        #TODO: copy plot code from lab1,2 to plot confusion matrix and store them in /results/
        if CV is not False:
            print("Vectorizing CV dataset ...")
            cv_df = pd.concat([self.train_df,self.val_df],axis="index").reset_index(drop="True")
            cv_arr = self.vectorizer.fit_transform(cv_df["text"]).toarray()
            self.test_arr = self.vectorizer.transform(self.test_df["text"]).toarray()
            print("Performing cross-validation ...")
            cv_result = cross_validate(self.model, cv_arr, cv_df["score"], cv=CV, 
                verbose=verbose, return_estimator=True, return_train_score=True)
            best_idx = np.argmax(cv_result["test_score"])
            
            val_acc=cv_result["test_score"][best_idx]
            train_acc=cv_result["train_score"][best_idx]
            #save model that performed the best on the val fold
            self.model = cv_result["estimator"][best_idx]
            
        else:
            print("Training ...")
            self.model.fit(self.train_arr,self.train_df["score"])
            train_acc=self.model.score(self.train_arr, self.train_df["score"].values)
            val_acc=self.model.score(self.val_arr, self.val_df["score"].values)
        
        if save_model:
            print("Saving model ...")
            time = datetime.now().strftime("%Y%m%d_%H%M")
            save_file="models/saved_weights/SVC_" +time + ".pkl"
            try:
                mode = 'a' if os.path.exists(save_file) else 'wb'
                with open(save_file,mode) as f:
                    pickle.dump(self.model, file=f)
            except IOError as e:
                raise Exception("Failed to save to %s: %s" % (save_file, e))
        
        return train_acc, val_acc
    
    def test(self):
        #TODO: copy plot code from lab1,2 to plot confusion matrix and store them in /results/
        print("Testing model on the test set ...")
        predictions = self.model.predict(self.test_arr)
        test_acc = self.model.score(self.test_arr,self.test_df["score"])
        return predictions, test_acc

    def classify_sentiment(self, unlabelled_data, save_csv=False):
        print("Vectorizing 1M tweets")
        Tweet_arr = self.vectorizer.transform(unlabelled_data["text"]).toarray()
        print("Predicting sentiments of 1M tweets ...")
        new_predictions = self.model.predict(Tweet_arr)
        sentiment_df=unlabelled_data.loc[:,["created_at","id","text"]]
        sentiment_df["score"] = new_predictions
        print(sentiment_df.head(-1))
        if save_csv:
            print("Saving predicted sentiments ...")
            time = datetime.now().strftime("%Y%m%d_%H%M")
            save_file=f"results/svc/Tweet_sentiments_" +  time + ".csv"
            try:
                mode = 'a' if os.path.exists(save_file) else 'wb'
                with open(save_file,mode) as f:
                    sentiment_df.to_csv(save_file)
            except IOError as e:
                raise Exception("Failed to save to %s: %s" % (save_file, e))

        return sentiment_df

