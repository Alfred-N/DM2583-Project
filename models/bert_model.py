import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from models.model_api import ModelInterface
from models.distilbert_for_sequences import DistilBertForSequenceClassification
from models.sequence_dataset import SequenceDataset
from torch.utils.data import DataLoader
from transformers import DistilBertConfig
from torch.optim import lr_scheduler
from datetime import datetime
import os
from tqdm import tqdm

class DistilBERT(ModelInterface):
    model: DistilBertForSequenceClassification
    config: DistilBertConfig
    train_dl: DataLoader
    val_dl: DataLoader
    test_dl: DataLoader

    def __init__(self, train_data, val_data, test_data, batch_size=16):
        super(DistilBERT, self).__init__(train_data, val_data, test_data)
        print("Creating model ...")
        self.config=DistilBertConfig(   vocab_size_or_config_json_file=32000, 
                                        hidden_size=768,dropout=0.1,num_labels=2,
                                        num_hidden_layers=12, num_attention_heads=12, 
                                        intermediate_size=3072)
        print("Loading sequences datasets ...")
        self.model = DistilBertForSequenceClassification(config=self.config)
        train= SequenceDataset(self.train_df["text"].values,self.train_df["one_hot_score"].values)
        val= SequenceDataset(self.val_df["text"].values,self.val_df["one_hot_score"].values)
        test= SequenceDataset(self.test_df["text"].values,self.test_df["one_hot_score"].values)
        print("Loading data loaders ...")
        self.train_dl = DataLoader(train, batch_size=batch_size, shuffle=False)
        self.val_dl = DataLoader(val, batch_size=batch_size, shuffle=False)
        self.test_dl = DataLoader(test, batch_size=batch_size, shuffle=False)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        print("Using device: ", self.device)
        self.model.to(self.device)
        

    def train(self, n_epochs=1, save_model=True):
        max_lr = 0.1
        min_lr = 3e-5
        optimizer = torch.optim.Adam(self.model.parameters(),min_lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, n_epochs*len(self.train_dl), 
        #                                 eta_min=min_lr, last_epoch=-1, verbose=False)
        criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.CrossEntropyLoss()
        print("Training ...")
        train_loss_list=[]
        train_avg_loss_list=[]
        val_loss_list=[]
        train_acc_list=[]
        val_acc_list=[]

        self.model.train()
        for epoch in range(n_epochs):
            print('Epoch {}/{}'.format(epoch+1, n_epochs))
            moving_avg_loss = 0.0
            tot_loss = 0.0
            train_acc = 0.0
            for it, (inputs, labels) in enumerate(tqdm(self.train_dl)):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                torch.set_grad_enabled(True)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels.float())
                # loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                tot_loss+=loss.item()
                moving_avg_loss = (moving_avg_loss + loss.item())/2.0
                train_avg_loss_list.append(moving_avg_loss)
                predictions = torch.argmax(outputs,dim=1).cpu().detach().numpy()
                actuals = torch.argmax(labels,dim=1).cpu().detach().numpy()
                train_acc += (predictions == actuals).sum()
                
                msg = f"Step: {it}/" + str(len(self.train_dl)) + " Loss = " + str(moving_avg_loss)
                tqdm.write(msg)
                if it%10==0:
                    tqdm.write(msg)
            
            scheduler.step()
            train_acc = train_acc/len(self.train_df["score"].values)
            train_loss_list.append(tot_loss)
            train_acc_list.append(train_acc)
            
            tqdm.write("Validaiting ...")
            val_loss=0.0
            val_acc=0.0
            for it, (inputs, labels) in enumerate(tqdm(self.val_dl)):                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                torch.set_grad_enabled(False)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels.float())
                # loss = criterion(outputs, labels)
                val_loss+=loss.item()
                predictions = torch.argmax(outputs,dim=1).cpu().detach().numpy()
                actuals = torch.argmax(labels,dim=1).cpu().detach().numpy()
                val_acc += (predictions == actuals).sum()

            val_acc = val_acc/len(self.val_df["score"].values)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)
            msg = str(val_acc)
            tqdm.write(msg)


            if save_model:
                print("Saving model ...")
                time = datetime.now().strftime("%Y%m%d_%H%M")
                save_file=f"models/saved_weights/distilbert_epoch_{epoch}_" +  time + ".pth"
                try:
                    mode = 'a' if os.path.exists(save_file) else 'wb'
                    with open(save_file,mode) as f:
                        torch.save(self.model.state_dict(), save_file)
                except IOError as e:
                    raise Exception("Failed to save to %s: %s" % (save_file, e))
        
        return train_loss_list, train_avg_loss_list, val_loss_list, train_acc_list, val_acc_list


    def test(self):
        test_loss=None
        test_accuracy=None
        test_confusion_matrix = None
        return test_loss, test_accuracy, test_confusion_matrix
    
    def classify_sentiment(self, unlabelled_data):
        pass