import pandas as pd

class ModelInterface():
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    def __init__(self,train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame):
        self.train_df, self.val_df, self.test_df = train_data, val_data, test_data
    

    def train(self, n_epochs=50, CV=False, verbose=False):
        """[summary]

        Parameters
        ----------
        n_epochs : int, optional
            [description], by default 50
        CV : int/bool, optional
            [description], by default False
        verbose : str/bool, optional
            [description], by default False

        Returns
        -------
        [type]
            [description]
        """
        train_loss=None
        val_loss=None
        train_accuracy=None
        val_accuracy=None
        train_confusion_matrix = None
        val_confusion_matrix = None
        return train_loss, val_loss, train_accuracy, val_accuracy, train_confusion_matrix, val_confusion_matrix

    def test(self):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        test_loss=None
        test_accuracy=None
        test_confusion_matrix = None
        return test_loss, test_accuracy, test_confusion_matrix
    
    def classify_sentiment(self, unlabelled_data):
        pass