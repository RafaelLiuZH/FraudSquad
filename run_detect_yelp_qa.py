import os
import sys
import numpy as np 
import yaml
import torch
import argparse
from tqdm import tqdm

import pandas as pd
from methods.evaluate import evaluate


def main(args_main):
    CONFIG_FILE = args_main.config
    with open(CONFIG_FILE) as file:
        args = yaml.safe_load(file)
    if args_main.device != 'config':
        args['device'] = args_main.device

    if 'yelp' in args_main.config:
        print(f'Processing Yelp dataset: {args_main.config}')
        from preprocess.preprocess import preprocess_df, train_val_test_split, get_fraud_features, text2emb_bertbase, get_graphs
        df = pd.read_csv('data/meta_yelpHotelData_NRYRcleaned.txt', sep=' ',
                        names=['reviewTime', 'reviewID', 'user', 'product', 'label', 'c1', 'c2', 'c3','star'])

        with open('data/review_yelpHotelData_NRYRcleaned.txt', 'r') as handle:
            review_texts = handle.readlines()

        review_ids = [len(text)>0 for text in review_texts]
        review_texts = [text for text in review_texts if len(text)>0]

        df = df.loc[review_ids,:]

        review_n = df.shape[0]
        df.index = range(review_n)

        df = df.loc[review_ids,:]
        df = preprocess_df(df, df_type='yelp')

        review_n = df.shape[0]
        df.index = range(review_n)
        labels = (df.label.values=='Y').astype(int)
        df.label = labels

        review_texts = [review_texts[i] for i in review_ids]          
        graphs, adjs, graph_homo = get_graphs(df, connect_columns=['user', 'product_star', 'product_month'], self_loop=True)
        other_features = get_fraud_features(df, review_texts)           

    if 'qa' in args_main.config:
        print(f'Processing QA dataset: {args_main.config}')
        from preprocess.preprocess_qa import read_qa_data, preprocess_df_qa, get_df_features, get_fraud_features_qa
        from preprocess.preprocess import train_val_test_split, text2emb_bertbase, get_graphs    

        df = read_qa_data()

        review_texts = []
        for row_id, qa in tqdm(df.iterrows()):
            review_text = '提问标题：%s\n提问内容：%s\n回答：%s'%(qa['qTitle'], qa['qCon'], qa['aCon'])
            review_texts.append(review_text)

        review_ids = [len(text)>0 for text in review_texts]
        review_texts = [text for text in review_texts if len(text)>0]
        review_n = len(review_texts)
        df = df.loc[review_ids,:]
        df = preprocess_df_qa(df)
        df_features = get_df_features(df)
        other_features = get_fraud_features_qa(df, review_texts, df['qTitle'].values, df['aCon'].values)
        other_features = np.hstack((other_features, df_features.values))          
        graphs, adjs, graph_homo = get_graphs(df, connect_columns=['qid', 'asker_month', 'answer_month'], 
                                      drop_values=[[], [], ['nan_']],
                                      self_loop=True)
        #     
    print('train_ratio:', (1-args['test_size'])*(1-args['val_size']))
    print('val_ratio:', (1-args['test_size'])*args['val_size'])
    print('test_ratio:', args['test_size'])
    print(args)

    labels = df.label.values
    train_mask, val_mask, test_mask = train_val_test_split(labels, args)

    graph_homo = graph_homo.to(args['device'])
    labels = torch.Tensor(labels)
    labels = labels.to(torch.long).to(args['device'])

    # %%
    # graph_homo
    print('Labels: num 0, num1, ratio1, total')
    print((labels==0).sum(), (labels==1).sum(), (labels==1).sum()/review_n, labels.shape,)

    from methods.fraudsquad_src.model_handler import fraudsquad_handler
    from methods.evaluate import evaluate

    test_score, test_auc, f1, ap, val_auc = fraudsquad_handler(review_texts, graph_homo, train_mask, val_mask, test_mask, labels.cpu().numpy(), args)
    auc, ap, precision, recall, f1 = evaluate(test_score, labels.cpu().numpy()[test_mask], args['top_ratio'])

    print('Fraudsquad AUC:', auc)
    print('Fraudsquad AP:', ap)
    print('Fraudsquad Precision:', precision)
    print('Fraudsquad Recall:', recall)
    print('Fraudsquad F1:', f1)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    # Modify the argument below to run Yelp or QA datasets.
    arg_parser.add_argument('--config', type=str, default='config/yelp_cfg_train1.yml')
    arg_parser.add_argument('--device', type=str, default='config')
    
    args_main = arg_parser.parse_args()
    main(args_main)


