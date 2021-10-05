import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer

class SequenceDataset(Dataset):
    def __init__(self,x,y, max_seq_length=256, transform=None):
        self.x = x
        self.y = y
        self.transform = transform
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.max_seq_length = max_seq_length
    def __getitem__(self,index):
        
        tokenized_comment = self.tokenizer.tokenize(self.x[index])
        
        if len(tokenized_comment) > self.max_seq_length:
            tokenized_comment = tokenized_comment[:self.max_seq_length]
            
        id_sequence  = self.tokenizer.convert_tokens_to_ids(tokenized_comment)

        padding = [0] * (self.max_seq_length - len(id_sequence))        
        id_sequence += padding        
        assert len(id_sequence) == self.max_seq_length
        ids_review = torch.tensor(id_sequence)

        # label = torch.tensor(self.y[index])
        # label = torch.tensor([self.y[index]])
        label = torch.tensor(self.y[index])
        return ids_review, label
    
    def __len__(self):
        return len(self.x)