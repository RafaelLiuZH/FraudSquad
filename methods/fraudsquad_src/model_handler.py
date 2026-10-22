import numpy as np
import dgl
import torch
import os
import copy
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
import torch.optim as optim
from scipy.io import loadmat
import pandas as pd
import pickle
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.dataloading import DataLoader # NodeDataLoader
from torch.optim.lr_scheduler import MultiStepLR
from .model import GraphAttnModel
from .language_model import text2emb
from . import *

def fraudsquad_handler(texts, graph, train_mask, val_mask, test_mask, labels, args, cat_features=[], val_strategy='auc'):
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    labels = pd.Series(labels)
    print('Text2emb')
    text_embs = text2emb(texts, args['device'], args['lm'])
    feat_df = pd.DataFrame(text_embs)
    # Modified from GTAN repository
    device = args['device']
    if not isinstance(device, list):
        graph = graph.to(device)
    else:
        subgraphs = dgl.distributed.partition_graph(graph, num_parts=len(device))
        subgraphs_on_devices = []
        # Place each subgraph on a different GPU
        for i,g in enumerate(subgraphs):
            g = g.to(device[i]) # g.to(torch.device(device[i]))
            subgraphs_on_devices.append(g)        
    oof_predictions = torch.from_numpy(
        np.zeros([len(feat_df), 2])).float().to(device)
    test_predictions = torch.from_numpy(
        np.zeros([len(feat_df), 2])).float().to(device)

    y_target = labels.iloc[train_idx].values
    num_feat = torch.from_numpy(feat_df.values).float().to(device)
    cat_feat = {col: torch.from_numpy(feat_df[col].values).long().to(
        device) for col in cat_features}

    y = labels
    labels = torch.from_numpy(y.values).long().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    trn_ind, val_ind = torch.from_numpy(np.array(train_idx)).long().to(
        device), torch.from_numpy(np.array(val_idx)).long().to(device)
    print('MultiLayerFullNeighborSampler')
    train_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
    train_dataloader = DataLoader(graph,
                                        trn_ind,
                                        train_sampler,
                                        device=device,
                                        use_ddp=False,
                                        batch_size=args['batch_size'],
                                        shuffle=True,
                                        drop_last=False,
                                        num_workers=0
                                        )
    val_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
    val_dataloader = DataLoader(graph,
                                    val_ind,
                                    val_sampler,
                                    use_ddp=False,
                                    device=device,
                                    batch_size=args['batch_size'],
                                    shuffle=True,
                                    drop_last=False,
                                    num_workers=0,
                                    )
    # TODO
    model = GraphAttnModel(in_feats=feat_df.shape[1],
                            # 为什么要整除4？
                            hidden_dim=args['hidden_dim']//4,
                            n_classes=2,
                            heads=[4]*args['n_layers'],  # [4,4,4]
                            activation=nn.PReLU(),
                            n_layers=args['n_layers'],
                            drop=args['dropout'],
                            device=device,
                            gated=args['gated'],
                            ref_df=feat_df.iloc[train_idx],
                            cat_features=cat_feat).to(device)
    lr = args['lr'] * np.sqrt(args['batch_size']/1024)  # 0.00075
    optimizer = optim.Adam(model.parameters(), lr=lr,
                            weight_decay=args['wd'])
    lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=[
                                4000, 12000], gamma=0.3)

    earlystoper = early_stopper(
        patience=args['early_stopping'], verbose=True)
    best_model_on_val = None
    best_val_auc = 0.0
    start_epoch, max_epochs = 0, 2000
    print('GNN training')
    for epoch in range(start_epoch, args['max_epochs']):
        train_loss_list = []
        # train_acc_list = []
        model.train()
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            # print('load_lpa_subtensor')
            batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, labels,
                                                                                            seeds, input_nodes, device)
            # (|input|, feat_dim); null; (|batch|,); (|input|,)
            blocks = [block.to(device) for block in blocks]
            train_batch_logits = model(
                blocks, batch_inputs, lpa_labels, batch_work_inputs)
            mask = batch_labels == 2
            train_batch_logits = train_batch_logits[~mask]
            batch_labels = batch_labels[~mask]
            # batch_labels[mask] = 0

            train_loss = loss_fn(train_batch_logits, batch_labels)
            # backward
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_loss_list.append(train_loss.cpu().detach().numpy())

            if step % 10 == 0:
                tr_batch_pred = torch.sum(torch.argmax(train_batch_logits.clone(
                ).detach(), dim=1) == batch_labels) / batch_labels.shape[0]
                score = torch.softmax(train_batch_logits.clone().detach(), dim=1)[
                    :, 1].cpu().numpy()

                # if (len(np.unique(score)) == 1):
                #     print("all same prediction!")
                try:
                    print('In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, '
                            'train_ap:{:.4f}, train_acc:{:.4f}, train_auc:{:.4f}'.format(epoch, step,
                                                                                        np.mean(
                                                                                            train_loss_list),
                                                                                        average_precision_score(
                                                                                            batch_labels.cpu().numpy(), score),
                                                                                        tr_batch_pred.detach(),
                                                                                        roc_auc_score(batch_labels.cpu().numpy(), score)))
                except:
                    pass

        # mini-batch for validation
        val_loss_list = 0
        val_acc_list = 0
        val_all_list = 0
        model.eval()
        with torch.no_grad():
            for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
                batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, labels,
                                                                                                seeds, input_nodes, device)

                blocks = [block.to(device) for block in blocks]
                val_batch_logits = model(
                    blocks, batch_inputs, lpa_labels, batch_work_inputs)
                oof_predictions[seeds] = val_batch_logits
                mask = batch_labels == 2
                val_batch_logits = val_batch_logits[~mask]
                batch_labels = batch_labels[~mask]
                # batch_labels[mask] = 0
                val_loss_list = val_loss_list + \
                    loss_fn(val_batch_logits, batch_labels)
                # val_all_list += 1
                val_batch_pred = torch.sum(torch.argmax(
                    val_batch_logits, dim=1) == batch_labels) / torch.tensor(batch_labels.shape[0])
                val_acc_list = val_acc_list + val_batch_pred * \
                    torch.tensor(batch_labels.shape[0])
                val_all_list = val_all_list + batch_labels.shape[0]
                if step % 10 == 0:
                    score = torch.softmax(val_batch_logits.clone().detach(), dim=1)[
                        :, 1].cpu().numpy()
                    try:
                        print('In epoch:{:03d}|batch:{:04d}, val_loss:{:4f}, val_ap:{:.4f}, '
                                'val_acc:{:.4f}, val_auc:{:.4f}'.format(epoch,
                                                                        step,
                                                                        val_loss_list/val_all_list,
                                                                        average_precision_score(
                                                                            batch_labels.cpu().numpy(), score),
                                                                        val_batch_pred.detach(),
                                                                        roc_auc_score(batch_labels.cpu().numpy(), score)))
                    except:
                        pass
        if val_strategy=='auc':
            val_score = torch.softmax(oof_predictions, dim=1)[val_ind, 1].cpu().numpy()
            y_val_target = labels[val_ind].cpu().numpy()
            val_score1 = torch.argmax(oof_predictions, dim=1)[val_ind].cpu().numpy()

            mask = y_val_target!=2
            val_score = val_score[mask]
            y_val_target = y_val_target[mask]
            val_auc = roc_auc_score(y_val_target, val_score)                
            if val_auc > best_val_auc:
                b_model = copy.deepcopy(model).to('cpu')
                best_val_auc = val_auc + 0
                print("Better validation auc {:.7f}".format(val_auc))
        elif val_strategy=='early_stop':
            earlystoper.earlystop(val_loss_list/val_all_list, model)
            if earlystoper.is_earlystop:
                print("Early Stopping!")
                break
    if val_strategy=='early_stop':
        print("Best val_loss is: {:.7f}".format(earlystoper.best_cv))
    test_ind = torch.from_numpy(np.array(test_idx)).long().to(device)
    test_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
    test_dataloader = DataLoader(graph,
                                        test_ind,
                                        test_sampler,
                                        use_ddp=False,
                                        device=device,
                                        batch_size=args['batch_size'],
                                        shuffle=True,
                                        drop_last=False,
                                        num_workers=0,
                                        )

    if val_strategy=='auc':
        b_model = b_model.to(device)
    elif val_strategy=='early_stop':
        b_model = earlystoper.best_model.to(device)
    b_model.eval()
    with torch.no_grad():
        for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
            # print(input_nodes)
            batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, labels,
                                                                                            seeds, input_nodes, device)

            blocks = [block.to(device) for block in blocks]
            test_batch_logits = b_model(
                blocks, batch_inputs, lpa_labels, batch_work_inputs)
            test_predictions[seeds] = test_batch_logits
            test_batch_pred = torch.sum(torch.argmax(
                test_batch_logits, dim=1) == batch_labels) / torch.tensor(batch_labels.shape[0])
            if step % 10 == 0:
                print('In test batch:{:04d}'.format(step))
    mask = y_target == 2
    y_target[mask] = 0
    my_ap = average_precision_score(y_target, torch.softmax(
        oof_predictions, dim=1).cpu()[train_idx, 1])
    print("NN out of fold AP is:", my_ap)
    b_models, val_gnn_0, test_gnn_0 = b_model.to('cpu'), oof_predictions, test_predictions

    test_score = torch.softmax(test_gnn_0, dim=1)[test_idx, 1].cpu().numpy()
    y_target = labels[test_idx].cpu().numpy()
    test_score1 = torch.argmax(test_gnn_0, dim=1)[test_idx].cpu().numpy()

    mask = y_target != 2
    test_score = test_score[mask]
    y_target = y_target[mask]
    test_score1 = test_score1[mask]
    
    test_auc = roc_auc_score(y_target, test_score)
    f1 = f1_score(y_target, test_score1, average="macro")
    ap = average_precision_score(y_target, test_score)
    print("test AUC:", test_auc)
    print("test f1:", f1)
    print("test AP:", ap)

    # Added by Xin Liu
    val_score = torch.softmax(val_gnn_0, dim=1)[val_ind, 1].cpu().numpy()
    y_val_target = labels[val_ind].cpu().numpy()
    val_score1 = torch.argmax(val_gnn_0, dim=1)[val_ind].cpu().numpy()
    
    mask = y_val_target!=2
    val_score = val_score[mask]
    y_val_target = y_val_target[mask]
    val_auc = roc_auc_score(y_val_target, val_score)
    return test_score, test_auc, f1, ap, val_auc
