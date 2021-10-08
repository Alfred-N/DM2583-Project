import pandas as pd
import numpy as np

class DataProcessor():
    full_dataset:pd.DataFrame

    def __init__(self, filename):
        self.filename = filename
        self.load_csv()

    def load_csv(self):
        path=self.filename
        try:
            df = pd.read_csv(path)
        except IOError as e:
            raise Exception("Failed to load %s: %s" % (path, e))

        df["text"] = df["text"].str.lower()
        self.full_dataset = df.loc[:]
        return self.full_dataset
    
    def preprocess_lab_data(self):
        """ Reformat lab data from [0,1] encoding to [-1,1] encoding       
        """
        print("Processing dataset from the labs ...")
        self.full_dataset["score"] = self.full_dataset["score"].apply(lambda x: -1 if x==0 else 1)

    def preprocess_US_Airlines_data(self):
        """ Reformat lab data from "negative/neutral/positive" encoding to [-1,0,1] encoding       
        """
        print("Processing US Airlines dataset ...")
        self.full_dataset = self.full_dataset.loc[:,["airline_sentiment", "text", "tweet_created", "tweet_location"]]
        self.full_dataset["score"]  = self.full_dataset["airline_sentiment"].apply(lambda x: self.string_sentiment_to_idx(x))

    def get_onehot_encoding(self):        
        data = self.full_dataset
        data["one_hot_score"] = data["score"].apply(lambda x: self.index_to_3D_onehot(x))
        self.full_dataset=data



    def split_data(self, frac={"train": 0.6, "val": 0.2, "test": 0.2}):
        if frac['train'] + frac['val'] + frac['test'] != 1.0:
            raise Exception(f"Error: fractions {frac} do not sum to one")
        dataset_len = self.full_dataset.shape[0]
        train_cutoff=np.ceil(dataset_len*frac["train"])
        val_cutoff=np.ceil(dataset_len * (frac["val"]+frac["train"]))

        train_df = self.full_dataset.loc[:train_cutoff,:]
        val_df = self.full_dataset.loc[train_cutoff:val_cutoff,:]
        test_df = self.full_dataset.loc[val_cutoff:,:]
        return train_df, val_df, test_df

    def index_to_3D_onehot(self,idx):
        if idx==-1:
            return [1,0,0]
        elif idx == 0:
            return [0,1,0]
        elif idx == 1:
            return [0,0,1]
    
    @staticmethod
    def predictions_to_idx(idx):
        if idx==0:
            return -1
        elif idx == 1:
            return 0
        elif idx == 2:
            return 1

    def string_sentiment_to_idx(self,sentiment_str):        
        if sentiment_str == "negative":
            return -1
        elif sentiment_str == "neutral":
            return 0
        elif sentiment_str == "positive":
            return 1

    
