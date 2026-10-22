import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn import GATConv
import numpy as np
import time

# class GAT(nn.Module):
#     def __init__(self, args):
#         super(GAT, self).__init__()
#         self.layer1 = GATConv(args['in_dim'], args['hidden_dim'], args['num_heads'])
#         # Be aware that the input dimension is hidden_dim*num_heads since
#         # multiple head outputs are concatenated together. Also, only
#         # one attention head in the output layer.
#         self.layer2 = GATConv(args['hidden_dim']*args['num_heads'], args['out_dim'], 1,
#                               feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, 
#                               residual=False, activation=None, allow_zero_in_degree=False, bias=True)

#     def forward(self, h):
#         h = self.layer1(h)
#         h = F.elu(h)
#         h = self.layer2(h)
#         return h

class GAT(nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()      
        in_dim =args['in_dim']
        self.layer1 = GATConv(in_dim, args['hidden_dim'], args['num_heads'],
                              feat_drop=args['dropout'][0], attn_drop=args['dropout'][1], negative_slope=0.2, 
                              residual=False, activation=None, allow_zero_in_degree=False, bias=True)
        in_dim = args['hidden_dim']*args['num_heads'] + args['in_dim']
        self.layer2 = GATConv(in_dim, args['hidden_dim'], args['num_heads'],
                              feat_drop=args['dropout'][0], attn_drop=args['dropout'][1], negative_slope=0.2, 
                              residual=False, activation=None, allow_zero_in_degree=False, bias=True)
        in_dim = args['hidden_dim']*args['num_heads'] + (args['hidden_dim']*args['num_heads']+ args['in_dim'])
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, args['hidden_dim']),
            nn.ReLU(),
            nn.Linear(args['hidden_dim'], args['out_dim']),
        )           

    def forward(self, g, features):
        # print(features.shape)
        h1 = self.layer1(g, features)
        h1 = torch.flatten(h1, 1)
        h1 = torch.cat((features, h1), dim=1)
        # print(h.shape)
        h1 = F.relu(h1)
        # print(h.shape)
        h2 = self.layer2(g, h1)
        h2 = torch.flatten(h2, 1)
        h2 = torch.cat((h1,h2), dim=1)
        h2 = F.relu(h2)
        h3 = self.mlp(h2)
        # print(h.shape)
        return h3

# def train_and_pred_gnn(model, graph, features, labels, train_mask, val_mask, args, validation_metric='none'):
#     # TODO: implementing using valiation metric
#     if args['device'] != 'cpu':
#         graph = graph.to(args['device'])
#         model = model.to(args['device'])
#         features = torch.Tensor(features).to(args['device'])
#         labels = torch.Tensor(labels).to(torch.long).to(args['device'])
#     optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])  
    
#     # The following code is updated from the example code showing GNN training in GDL (Deep Graph Library)
#     dur = []
#     loss_f = nn.CrossEntropyLoss()
#     train_loss = []
#     val_loss = []
#     for epoch in range(args['max_epochs']):
#         if epoch >= 3:
#             t0 = time.time()
    
#         logits = model(graph, features)
#         loss = loss_f(logits[train_mask],labels[train_mask])
    
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss.append(loss.item())
#         with torch.no_grad():
#             loss_val = loss_f(logits[val_mask],labels[val_mask])
#             val_loss.append(loss_val.item())
#         if epoch >= 3:
#             dur.append(time.time() - t0)
    
#         print(
#             "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
#                 epoch, loss.item(), np.mean(dur)
#             )
#         )   
    
#     with torch.no_grad():
#         logits = model(graph, features)
#         probs = torch.softmax(logits, 1)
#         probs = probs.to('cpu').detach().numpy()

#     return model, probs, train_loss, val_loss

def train_and_pred_gnn(model, graph, features, labels, train_mask, val_mask, args, validation_metric='none'):
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])  
    print(features.shape)
    dur = []
    loss_f = nn.CrossEntropyLoss()
    train_loss = []
    val_loss = []
    
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(args['max_epochs']):
        if epoch >= 3:
            t0 = time.time()
    
        logits = model(graph, features)
        loss = loss_f(logits[train_mask], labels[train_mask])
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        
        with torch.no_grad():
            logits_val = model(graph, features)
            loss_val = loss_f(logits_val[val_mask], labels[val_mask])
            val_loss.append(loss_val.item())
            
            # Save the best model state if validation_metric is 'loss'
            if validation_metric == 'loss' and loss_val.item() < best_val_loss:
                best_val_loss = loss_val.item()
                best_model_state = model.state_dict()

        if epoch >= 3:
            dur.append(time.time() - t0)
    
        print(
            "Epoch {:05d} | Train Loss {:.4f} | Val Loss {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), loss_val.item(), np.mean(dur)
            )
        )
    
    # Load the best model state if validation_metric is 'loss'
    if validation_metric == 'loss' and best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Perform inference with the best model if validation_metric is 'loss'
    with torch.no_grad():
        logits = model(graph, features)
        probs = torch.softmax(logits, 1)
        probs = probs.to('cpu').detach().numpy()

    return model, probs, train_loss, val_loss

