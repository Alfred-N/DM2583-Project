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

        self.full_dataset = df.loc[:]
        return self.full_dataset

    def reformat_data(self):
        """ Reformat data into csv containing one "score" column and one "text" column.
            Might have to be multiple functions depending on sources format
        """
        return self.full_dataset

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