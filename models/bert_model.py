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

    def __init__(self, train_data, val_data, test_data, batch_size=16, max_seq_len=256):
        super(DistilBERT, self).__init__(train_data, val_data, test_data)
        print("Creating model ...")
        self.config=DistilBertConfig(   vocab_size_or_config_json_file=32000, 
                                        hidden_size=768,dropout=0.1,num_labels=2,
                                        num_hidden_layers=12, num_attention_heads=12, 
                                        intermediate_size=3072)
        print("Loading sequences datasets ...")
        self.model = DistilBertForSequenceClassification(config=self.config)
        train= SequenceDataset(self.train_df["text"].values,self.train_df["one_hot_score"].values,max_seq_length=max_seq_len)
        val= SequenceDataset(self.val_df["text"].values,self.val_df["one_hot_score"].values,max_seq_length=max_seq_len)
        test= SequenceDataset(self.test_df["text"].values,self.test_df["one_hot_score"].values,max_seq_length=max_seq_len)
        print("Loading data loaders ...")
        self.train_dl = DataLoader(train, batch_size=batch_size, shuffle=False)
        self.val_dl = DataLoader(val, batch_size=batch_size, shuffle=False)
        self.test_dl = DataLoader(test, batch_size=batch_size, shuffle=False)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        print("Using device: ", self.device)
        self.model.to(self.device)
        

    def train(self, n_epochs=1, save_model=True):
        max_lr = 1e-4
        min_lr = 1e-5
        step_size=np.ceil(len(self.train_dl)/2)
        # step_size=1
        optimizer = torch.optim.Adam(self.model.parameters(),min_lr)
        scheduler = lr_scheduler.CyclicLR(optimizer,base_lr=min_lr,max_lr=max_lr,
                                step_size_up=step_size, step_size_down=step_size, mode="triangular2",
                                cycle_momentum=False)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
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
                with torch.set_grad_enabled(True):
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels.float())
                    # loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                tot_loss+=loss.item()
                moving_avg_loss = (moving_avg_loss + loss.item())/2.0
                train_avg_loss_list.append(moving_avg_loss)
                predictions = torch.argmax(outputs,dim=1).cpu().detach().numpy()
                actuals = torch.argmax(labels,dim=1).cpu().detach().numpy()
                train_acc += (predictions == actuals).sum()
                
                msg = f"Step: {it}/" + str(len(self.train_dl)) + " Loss = " + str(moving_avg_loss)
                # tqdm.write(msg)
                if it%10==0:
                    tqdm.write(msg)
                    msg= "LR="+str(scheduler.get_last_lr()[0])
                    tqdm.write(msg)
            
            
            train_acc = train_acc/len(self.train_df["score"].values)
            train_loss_list.append(tot_loss)
            train_acc_list.append(train_acc)
            
            tqdm.write("Validaiting ...")
            val_loss=0.0
            val_acc=0.0
            for it, (inputs, labels) in enumerate(tqdm(self.val_dl)):                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    outputs = self.model(inputs)
                    outputs = torch.sigmoid(outputs)
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
        predictions = []
        test_acc=0.0
        for it, (inputs, labels) in enumerate(tqdm(self.test_dl)):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                outputs = torch.sigmoid(outputs)
                predictions = torch.argmax(outputs,dim=1).cpu().detach().numpy()
                actuals = torch.argmax(labels,dim=1).cpu().detach().numpy()
                test_acc += (predictions == actuals).sum()
        test_acc=test_acc/len(self.test_df["score"].values)
        return predictions, test_acc
    
    def classify_sentiment(self, unlabelled_data):
        pass

    def load_from_state_dict(self, file=""):
        try:
            state_dict = torch.load(file)
        except IOError as e:
            raise Exception("Failed to load %s: %s" % (file, e))
        self.model.load_state_dict(state_dict)