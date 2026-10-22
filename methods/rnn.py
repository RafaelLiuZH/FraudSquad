import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import os
import pandas as pd

class YelpDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }
        return item
        
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(RNNClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])
        return self.fc(hidden)    

def train_and_pred_rnn(model, texts, labels, train_mask, val_mask, args):
    train_texts = texts[train_mask]
    val_texts = texts[val_mask]
    test_texts = texts[:]

    train_labels = labels[train_mask]
    val_labels = labels[val_mask]
    test_labels = labels[:]    
    
    # tokenizer = BertTokenizer.from_pretrained(args["llm"])
    tokenizer = BertTokenizer.from_pretrained('models/bert-base-uncased/')
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)
    
    train_input_ids = torch.tensor(train_encodings['input_ids'])
    train_attention_mask = torch.tensor(train_encodings['attention_mask'])
    val_input_ids = torch.tensor(val_encodings['input_ids'])
    val_attention_mask = torch.tensor(val_encodings['attention_mask'])
    test_input_ids = torch.tensor(test_encodings['input_ids'])
    test_attention_mask = torch.tensor(test_encodings['attention_mask'])    
    
    train_dataset = YelpDataset(train_input_ids, train_attention_mask, train_labels)
    val_dataset = YelpDataset(val_input_ids, val_attention_mask, val_labels)
    test_dataset = YelpDataset(test_input_ids, test_attention_mask, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)    

    train_labels = torch.tensor(labels[train_mask])
    val_labels = torch.tensor(labels[val_mask])
    test_labels = torch.tensor(labels)    
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    
    epoch_losses = []
    epoch_aucs = []
    epoch_val_aucs = []
    best_val_auc = 0.0
    
    model.train()
    for epoch in range(args['max_epochs']):
        epoch_labels = []
        epoch_preds = []
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(args['device'])
            labels = batch['labels'].float().unsqueeze(1).to(args['device'])
    
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            epoch_labels.extend(labels.cpu().numpy())
            epoch_preds.extend(torch.sigmoid(outputs).cpu().detach().numpy())
    
            total_loss += loss.item()
    
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
    
        epoch_labels = np.array(epoch_labels).flatten()
        epoch_preds = np.array(epoch_preds).flatten()
        auc = roc_auc_score(epoch_labels, epoch_preds)
        epoch_aucs.append(auc)
    
        print(f'Epoch {epoch + 1}, Loss: {avg_loss}, AUC: {auc}')
    
        with torch.no_grad():
            total, correct = 0, 0
            prob = []
            val_label = []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(args['device'])
                    labels = batch['labels'].float().unsqueeze(1).to(args['device'])
                    outputs = model(input_ids)
                    predicted = torch.sigmoid(outputs)
                    prob.extend(predicted.cpu().numpy())
                    val_label.extend(labels.cpu().numpy())
            # accuracy = correct / total
            auc = roc_auc_score(y_score=prob, y_true=val_label)
            epoch_val_aucs.append(auc)
            # print(f'Validation Accuracy: {accuracy}, AUC: {auc}')
            print(f'Validation AUC: {auc}')
            if auc > best_val_auc:
                best_val_auc = auc
                path_saver = os.path.join('models/pytorch_models/', '{}_{}.pkl'.format(args['dataset'], 'RNN'))
                torch.save(model.state_dict(), path_saver)
    model.load_state_dict(torch.load(path_saver))    

    model.eval()
    total, correct = 0, 0
    prob = []
    test_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(args['device'])
            labels = batch['labels'].float().unsqueeze(1).to(args['device'])
            outputs = model(input_ids)
            predicted = torch.sigmoid(outputs)
            # prob.extend(predicted.cpu().numpy())torch.sigmoid(outputs).cpu().detach().numpy()
            prob.extend(torch.sigmoid(outputs).cpu().detach().numpy())
            test_labels.extend(labels.cpu().numpy())  
    prob = np.array(prob)
    return model, prob, epoch_aucs, epoch_val_aucs        
