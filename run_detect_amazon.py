import os
import yaml
import torch
import argparse
import pandas as pd
from preprocess.preprocess import preprocess_df, train_val_test_split, get_fraud_features, text2emb_bertbase, get_graphs
from methods.evaluate import evaluate

def main(args_main):

    with open(args_main.config) as file:
        args = yaml.safe_load(file)
    if args_main.device != 'config':
        args['device'] = args_main.device
    
    print('train_ratio:', (1-args['test_size'])*(1-args['val_size']))
    print('val_ratio:', (1-args['test_size'])*args['val_size'])
    print('test_ratio:', args['test_size'])
    print('Training arguments:')
    for k, v in args.items():   
        print(f'{k}: {v}')

    print(f'Genuine review file: {args_main.genuine_review_file}')
    print(f'Spam review file: {args_main.spam_review_file}')

    df_genuine = pd.read_csv(os.path.join(args_main.data_dir, args_main.genuine_review_file))
    df_genuine['label'] = 0
    df_spam = pd.read_csv(os.path.join(args_main.data_dir, args_main.spam_review_file))
    df = pd.concat([df_genuine, df_spam], axis=0)
    df.index = range(df.shape[0])

    df = preprocess_df(df)

    df.head()

    # %%
    print(df.shape, df_genuine.shape, df_spam.shape)

    # %%
    review_texts = df['reviewText'].values
    review_ids, review_embs = text2emb_bertbase(review_texts, args['device']) 
    # review_ids = [len(text)>0 for text in review_texts]

    df = df.loc[review_ids,:]
    labels = df.label.values
    review_texts = df['reviewText'].values
    review_n = df.shape[0]
    df.index = range(review_n)
    print(df.shape)

    df.label.value_counts()

    labels = df.label.values
    train_mask, val_mask, test_mask = train_val_test_split(labels, args)

    # %%
    graphs, adjs, graph_homo = get_graphs(df, connect_columns=['user', 'product_star', 'product_month'], self_loop=True)
    other_features = get_fraud_features(df, review_texts)

    graph_homo = graph_homo.to(args['device'])
    labels = torch.Tensor(labels)
    labels = labels.to(torch.long).to(args['device'])

    # %%
    # graph_homo
    print('Labels: num 0, num1, ratio1, total')
    print((labels==0).sum(), (labels==1).sum(), (labels==1).sum()/review_n, labels.shape,)
   
    from methods.fraudsquad_src.model_handler import fraudsquad_handler
    from methods.evaluate import evaluate

    test_score, test_auc, f1, ap, val_auc = fraudsquad_handler(review_texts.tolist(), graph_homo, train_mask, val_mask, test_mask, labels.cpu().numpy(), args)
    auc, ap, precision, recall, f1 = evaluate(test_score, labels.cpu().numpy()[test_mask], args['top_ratio'])

    print('Fraudsquad AUC:', auc)
    print('Fraudsquad AP:', ap)
    print('Fraudsquad Precision:', precision)
    print('Fraudsquad Recall:', recall)
    print('Fraudsquad F1:', f1)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', type=str, default='config/amazon_cfg_0503.yml')
    arg_parser.add_argument('--data_dir', type=str, default='data/Amazon')
    arg_parser.add_argument('--genuine_review_file', type=str, default='amazon_2022_5core.csv')
    # Modify the argument below to run different LLM-generated spam review datasets.
    arg_parser.add_argument('--spam_review_file', type=str, default='reviews_from_qwen_dsr1.csv') 
    arg_parser.add_argument('--device', type=str, default='config')
    
    args_main = arg_parser.parse_args()
    main(args_main)


