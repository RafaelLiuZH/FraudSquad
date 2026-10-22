import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        # self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(args['in_dim'], args['hidden_dim']),
            nn.ReLU(),
            nn.Linear(args['hidden_dim'], args['out_dim']),
        )

    def forward(self, inputs):
        # inputs = self.flatten(inputs)
        logits = self.model(inputs)
        return logits

def train_and_pred_mlp(model, features, labels, train_mask, val_mask, args, validation_metric='none'):
    if args['device'] != 'cpu':
        model = model.to(args['device'])
        features = torch.Tensor(features).to(args['device'])
        labels = torch.Tensor(labels).to(torch.long).to(args['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    
    loss_f = nn.CrossEntropyLoss()
    
    train_loss = []
    val_loss = []
    dur = []
    
    for epoch in range(args['max_epochs']):
        if epoch >= 3:
            t0 = time.time()
    
        logits = model(features)
        loss = loss_f(logits[train_mask],labels[train_mask])
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        with torch.no_grad():
            loss_val = loss_f(logits[val_mask],labels[val_mask])
            val_loss.append(loss_val.item())
        if epoch >= 3:
            dur.append(time.time() - t0)
    
        print(
            "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), np.mean(dur)
            )
        )   
    
    with torch.no_grad():
        logits = model(features)
        probs = torch.softmax(logits, 1)
        probs = probs.to('cpu').detach().numpy()
        
    return model, probs, train_loss, val_loss        