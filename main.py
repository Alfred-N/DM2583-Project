import pandas as pd
import numpy as np
from data_processor import DataProcessor
from models.svc import SVC
from models.lstm import LSTM
from models.bert_model import DistilBERT
import torch
import matplotlib.pyplot as plt
from helper_functions import plot_training_result, print_sentiment_distribution

###------------------------------SVC---------------------------------
# processor = DataProcessor(filename="./data/unprocessed/combined_data_from_labs.csv")
# processor.load_csv()
# train,val,test= processor.split_data(frac={"train": 0.6,"val": 0.2,"test": 0.2 })
# model = SVC(train,val,test)
# train_acc, val_acc = model.train(CV=False,verbose=1,save_model=True)
# print("Training acc: ", train_acc, "Val acc: ", val_acc)
# _ , test_acc = model.test()
# print("Test acc: ", test_acc)

##Load model instead of training
# model.load_from_pickle(file="models\saved_weights\SVC_20211005_0050.pkl")
# _ , test_acc = model.test()
# print("Test acc: ", test_acc)

##------------------------------LSTM (Not implemented)---------------------------------
# processor = DataProcessor(filename="./data/unprocessed/combined_data_from_labs.csv")
# processor.load_csv()
# train,val,test= processor.split_data(frac={"train": 0.6,"val": 0.2,"test": 0.2 })
# model = LSTM(train,val,test)
# model.train()
# print(LSTM)

#------------------------------DistilBERT----------------------------
processor = DataProcessor(filename="data/processed/US_Airlines_Tweets_EVEN_DISTRIB.csv")
processor.preprocess_US_Airlines_data()
processor.get_onehot_encoding()
print_sentiment_distribution(processor.full_dataset, plot=False)
train,val,test= processor.split_data(frac={"train": 0.6,"val": 0.2,"test": 0.2 })
model = DistilBERT(train,val,test, batch_size=16, max_seq_len=256)
#Train
n_epochs=10
result = model.train(n_epochs=n_epochs)
plot_training_result(result,n_epochs)

##Test
model.load_from_state_dict("models\saved_weights\distilbert_epoch_9_20211007_1346.pth")
predictions, test_acc=model.test()
print("test acc = ", test_acc)
n_positive = train["score"].values.sum()
print("Fraction postive tweets: ",n_positive/len(train["score"].values))

#Predict (unlabeled) Twitter data
# model.load_from_state_dict("models\saved_weights\distilbert_epoch_2_20211007_1258.pth")
# twitter_proc = DataProcessor(filename="./data/processed/lab_data_new_encoding.csv")
# unlabelled_df = twitter_proc.full_dataset
# predicted_df=model.classify_sentiment(unlabelled_df, save_csv=True)