import pandas as pd
import numpy as np
from data_processor import DataProcessor
from models.svc import SVC
from models.lstm import LSTM
from models.bert_model import DistilBERT
import torch
import matplotlib.pyplot as plt
from helper_functions import plot_training_result, print_sentiment_distribution, plot_predicted_sentiments

###------------------------------SVC---------------------------------
processor = DataProcessor(filename="./data/processed/US_Airlines_Tweets_EVEN_DISTRIB.csv")
processor.load_csv()
train,val,test= processor.split_data(frac={"train": 0.8,"val": 0.1,"test": 0.1})
model = SVC(train,val,test)
train_acc, val_acc = model.train(CV=5,verbose=3,save_model=True)
print("Training acc: ", train_acc, "Val acc: ", val_acc)
_ , test_acc = model.test()
print("Test acc: ", test_acc)

##Load model instead of training
# model.load_from_pickle(file="models\saved_weights\SVC_20211005_0050.pkl")
# _ , test_acc = model.test()
# print("Test acc: ", test_acc)

twitter_proc = DataProcessor(filename="data/unprocessed/Twitter/Twitter_2015_to_2021.csv")
unlabelled_df = twitter_proc.full_dataset
predicted_df=model.classify_sentiment(unlabelled_df, save_csv=True)

plot_predicted_sentiments(  csv_path="results\svc\Tweet_sentiments_20211011_1945.csv",
                            plot_dir="results/svc", model_name="LinearSVC")

##------------------------------LSTM (Not implemented)---------------------------------
# processor = DataProcessor(filename="./data/unprocessed/combined_data_from_labs.csv")
# processor.load_csv()
# train,val,test= processor.split_data(frac={"train": 0.6,"val": 0.2,"test": 0.2 })
# model = LSTM(train,val,test)
# model.train()
# print(LSTM)

#------------------------------DistilBERT----------------------------
# data_dir="data/processed/"
# dataset="US_Airlines_Tweets_EVEN_DISTRIB"
# processor = DataProcessor(filename=data_dir+dataset+".csv")
# processor.preprocess_US_Airlines_data()
# processor.get_onehot_encoding()
# print_sentiment_distribution(processor.full_dataset, plot=False)
# train,val,test= processor.split_data(frac={"train": 0.8,"val": 0.1,"test": 0.1 })
# model = DistilBERT(train,val,test, batch_size=16, max_seq_len=256)
# model.load_from_state_dict("models/saved_weights/distilbert_epoch_3_20211009_0635.pth")
#Train
# n_epochs=3
# result = model.train(n_epochs=n_epochs)
# plot_training_result(result,n_epochs, model="distilbert", dataset=dataset)


##Test
# predictions, test_acc=model.test()
# print("test acc = ", test_acc)

#Predict (unlabeled) Twitter data
# model.load_from_state_dict("models/saved_weights/US_Airways_pretrained_on_Lab_data/distilbert_epoch_9_20211008_2333.pth")
# model.batch_size=256
# twitter_proc = DataProcessor(filename="data/unprocessed/Twitter/Twitter_2015_to_2021.csv")
# unlabelled_df = twitter_proc.full_dataset
# predicted_df=model.classify_sentiment(unlabelled_df, save_csv=True)
# print(predicted_df.head(-1))

#Plot sentiment

# plot_predicted_sentiments(  csv_path="results/distilbert/Tweet_sentiments_20211011_0005.csv",
#                             plot_dir="results/distilbert", model_name="DistilBERT")