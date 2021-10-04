import pandas as pd
import numpy as np
from data_processor import DataProcessor
from models.svc import SVC

processor = DataProcessor(filename="./data/unprocessed/combined_data_from_labs.csv")
processor.load_csv()
train,val,test= processor.split_data(frac={"train": 0.6,"val": 0.2,"test": 0.2 })

model = SVC(train,val,test)

train_acc, val_acc = model.train(CV=False,verbose=1,save_model=True)
print("Training acc: ", train_acc, "Val acc: ", val_acc)
_ , test_acc = model.test()
print("Test acc: ", test_acc)
model.load_from_pickle(file="models\saved_weights\SVC_20211005_0050.pkl")
_ , test_acc = model.test()
print("Test acc: ", test_acc)