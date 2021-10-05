import pandas as pd
import numpy as np

class DataProcessor():
    full_dataset:pd.DataFrame

    def __init__(self, filename):
        self.filename = filename

    def load_csv(self):
        path=self.filename
        try:
            df = pd.read_csv(path)
        except IOError as e:
            raise Exception("Failed to load %s: %s" % (path, e))

        #since we always want lowercase:
        df["text"] = df["text"].str.lower()
        self.full_dataset = df.loc[:]
        return self.full_dataset

    def reformat_data(self, label_dim=2):
        """ Reformat data into csv containing one "score" column and one "text" column.
            Might have to be multiple functions depending on source format
            If label_dim>=1, also create a column with one-hot encodings of the labels
        """
        data = self.full_dataset
        if label_dim==2:
            #[0,1] means positive, [1,0] means negative
            data["one_hot_score"] = data["score"].apply(lambda x: [0,1] if x==1 else [1,0])
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

    
