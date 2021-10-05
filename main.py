import pandas as pd
# import numpy as np
from data_processor import DataProcessor
from models.svc import SVC
from models.lstm import LSTM
from models.bert_model import DistilBERT
import torch
import matplotlib.pyplot as plt
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
processor = DataProcessor(filename="./data/unprocessed/combined_data_from_labs.csv")
processor.load_csv()
processor.reformat_data(label_dim=2)
train,val,test= processor.split_data(frac={"train": 0.6,"val": 0.2,"test": 0.2 })
model = DistilBERT(train,val,test, batch_size=16)
result = model.train(n_epochs=1)
train_loss_list, train_avg_loss_list, val_loss_list, train_acc_list, val_acc_list = result
plt.plot(train_avg_loss_list)
plt.ylabel("Loss")
plt.xlabel("Step")
plt.title("Smooth loss")
plt.show()
print("Train acc = ",train_acc_list)
print("Val acc = ",val_acc_list)
plt.savefig("results/distilbert/smooth_loss_1_epoch.png")