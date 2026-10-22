import pandas as pd
import numpy as np
import torch
import dgl
import random
import os
import time
import re
import scipy
import scipy.sparse as sp
from tqdm import tqdm
from itertools import combinations
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertModel, BertTokenizer
from transformers import AutoModel, AutoTokenizer
from dgl.data.utils import load_graphs, save_graphs

def preprocess_df(df, df_type='amazon'):
    if df_type=='amazon':
        # used df columns: star, product, user, reviewTime (Unix time)
        for col in ['star', 'product', 'user', 'reviewTime']:
            if col not in df.columns:
                print('Column', col, 'not in the input dataframe')
                return    
        df['reviewDay'] = list(map(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d'), df['reviewTime']))
        df['reviewMonth'] = list(map(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m'), df['reviewTime']))
        df['reviewYear'] = list(map(lambda x: datetime.fromtimestamp(x).strftime('%Y'), df['reviewTime']))
        def con_str(x): 
            return str(x[0])+'_'+str(x[1])
        df['product_star'] = list(map(con_str, zip(df['product'].values, df['star'].values)))
        df['product_month'] = list(map(con_str, zip(df['product'].values, df['reviewMonth'].values)))
        df['product_day'] = list(map(con_str, zip(df['product'].values, df['reviewDay'].values)))    
        review_texts = df['reviewText'].values
        print(len(review_texts))
        valid_review_ids = list(filter(lambda x:type(review_texts[x]) is str, np.arange(len(review_texts))))
        print(len(valid_review_ids))
        valid_review_ids = list(filter(lambda x:len(review_texts[x])>0, valid_review_ids))
        print(len(valid_review_ids))
        df = df.loc[valid_review_ids,:]
        df.index = range(df.shape[0])
    elif df_type=='yelp':
        def get_month(x):
            m,d,y = x.split('/')
            return '/'.join([m,y])            
        def con_str(x): 
            return str(x[0])+'_'+str(x[1])
        def get_unixtime(x):
            x = x.replace('/','-')
            date_obj = datetime.strptime(x + ' 12:00:00', '%m-%d-%Y %H:%M:%S')
            timestamp = date_obj.timestamp()
            return int(timestamp)
        df['reviewMonth'] = list(map(get_month, df['reviewTime']))   
        df['reviewDay'] = df['reviewTime']
        df['reviewTime'] = list(map(get_unixtime, df['reviewDay']))
        df['product_star'] = list(map(con_str, zip(df['product'].values, df['star'].values)))
        df['product_month'] = list(map(con_str, zip(df['product'].values, df['reviewMonth'].values)))
        df['product_month_star'] = list(map(con_str, zip(df['product_month'].values, df['star'].values)))
        df['product_day'] = list(map(con_str, zip(df['product'].values, df['reviewTime'].values)))        
    return df

def text2emb_bertbase(review_texts, device, size='base', return_ids=True):
    if size == 'base':
        model_directory = 'models/bert-base-uncased/'
    elif size == 'large':
        model_directory = 'models/bert-large-uncased/'
    tokenizer = BertTokenizer.from_pretrained(model_directory)
    model = BertModel.from_pretrained(model_directory)
    model.to(device)
    review_embs = []
    review_ids = []
    for i, text in tqdm(enumerate(review_texts)):
        try:
            input_ids = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True).to(device)
            with torch.no_grad():
                last_hidden_states = model(input_ids)[0]  # 获取最后一个隐藏状态
            pooled_output = last_hidden_states[:, 0, :][0]  # [batch_size, 768]
            review_embs.append(pooled_output.cpu().numpy())
            review_ids.append(i)
        except:
            pass   
    review_embs = np.array(review_embs)
    review_ids = np.array(review_ids)
    if return_ids:
        return review_ids, review_embs
    else:
        return review_embs

def text2emb_gte(review_texts, device, size='base', batch_size=10):
    if size == 'base':
        model_directory = 'models/Alibaba-NLP/gte-base-en-v1.5/'
    elif size == 'large':
        model_directory = 'models/Alibaba-NLP/gte-large-en-v1.5'
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = AutoModel.from_pretrained(model_directory, trust_remote_code=True)
    model.to(device)
    review_embs = [] 
    for i in tqdm(range(0, len(review_texts), batch_size)):
        input_texts = review_texts[i:(i+batch_size)]
        print(len(input_texts))
        batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt').to(device)
        outputs = model(**batch_dict)
        embeddings = outputs.last_hidden_state[:, 0]
        review_embs.append(embeddings)
    review_embs = torch.cat(review_embs, dim)
    return review_embs.cpu().numpy()

def text2emb(review_texts, device, llm, batch_size=10):
    if llm=='bert-base-uncased':
        return text2emb_bertbase(review_texts, device, size='base', return_ids=False)
    elif llm=='bert-large-uncased':
        return text2emb_bertbase(review_texts, device, size='large', return_ids=False)
    elif llm=='gte-base-en-v1.5':
        return text2emb_gte(review_texts, device, size='base', batch_size=batch_size)
    elif llm=='gte-large-en-v1.5':
        return text2emb_gte(review_texts, device, size='large', batch_size=batch_size)
    else:
        print(llm, 'is not supported')

def train_val_test_split(labels, args):
    all_idx = np.arange(len(labels))
    sample_n = len(labels)
    train_idx, test_idx, y_train, y_test = train_test_split(all_idx, labels, stratify=labels, test_size=args['test_size'],
                                                            random_state=args['random_state'], shuffle=True)   
    train_idx, val_idx, y_train, y_val  = train_test_split(train_idx, y_train, stratify=y_train,
                                                            test_size=args['val_size'],
                                                            random_state=args['random_state'], shuffle=True)  
    train_mask, val_mask, test_mask = np.zeros(sample_n, dtype=bool), np.zeros(sample_n, dtype=bool), np.zeros(sample_n, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    for mask in [train_mask, val_mask, test_mask]:
        print(sum(mask))
        mask[labels<0] = False
        print(sum(mask))
    return train_mask, val_mask, test_mask

def get_graphs(df, connect_columns=['user', 'product_star', 'product_month'], self_loop=True, drop_values=None):
    for col in connect_columns:
        if col not in df.columns:
            print('Column', col, 'not in the input dataframe')
            return        
    review_n = df.shape[0]
    # create edges between reviews
    graphs = []
    adjs = []
    for i,col in enumerate(connect_columns):
        print('============')
        print('Constructing a graph from', col)        
        source_nodes = []
        target_nodes = []
        for user, user_df in tqdm(df.groupby(col)):
            cur_reviews = user_df.index
            if drop_values is not None:
                drop_value = drop_values[i]
                if user in drop_value:
                    continue
            if len(cur_reviews)>1:
                for source, target in combinations(cur_reviews, r=2):
                    source_nodes.append(source)
                    target_nodes.append(target)
        print(len(target_nodes))
        source_nodes = np.array(source_nodes)
        target_nodes = np.array(target_nodes)
        adj = sp.coo_matrix(([1]*len(source_nodes), (source_nodes, target_nodes)), shape=(review_n,review_n)).tocsr()
        adj += adj.transpose()
        if self_loop:
            adj += sp.eye(review_n)
        adjs.append(adj)
        g = dgl.from_scipy(adj)
        # g = dgl.graph((source_nodes, target_nodes), num_nodes=review_n) 
        # g = dgl.add_self_loop(g)
        graphs.append(g)
        print(g)
        print('is_multigraph:',g.is_multigraph)
        print('graph density:', g.num_edges()/g.num_nodes()**2)
    
    graph_homo = dgl.merge(graphs)
    print('============')
    print('Homogeneous graph by combing all previous graphs')
    print(graph_homo)
    print('Is_multigraph:', graph_homo.is_multigraph)
    print('Graph density:', graph_homo.num_edges()/graph_homo.num_nodes()**2)
    return graphs, adjs, graph_homo

def get_sentiwordnet():
    SentiWordNet = pd.read_csv('preprocess/SentiWordNet/data/SentiWordNet_3.0.0.txt', sep='\t', header=25)
    def clean_word(word):
        i = word.index('#')
        return word[:i]
    SentiWordNet_d = {}    
    for row_id, row in SentiWordNet.iterrows():
        words = row['SynsetTerms'].split(' ')
        for word in words:
           SentiWordNet_d[clean_word(word)]=row['PosScore']+row['NegScore']  
    return SentiWordNet_d

def get_fraud_features(df, review_texts):
    # used df columns: star, product, user, reviewDay, reviewMonth, reviewTime (Unix time)
    for col in ['star', 'product', 'user', 'reviewDay', 'reviewMonth', 'reviewTime']:
        if col not in df.columns:
            print('Column', col, 'not in the input dataframe')
            return

    review_n = df.shape[0]    
    other_features = {}
    
    print('Review behavioral features')
    # # rating stars
    # other_features['star'] = df['star'].values.tolist()
    # Extremity of rating
    extremity = (df['star'].values>4.9)+(df['star'].values<1.1) # 1 or 5
    other_features['extremity'] = extremity.astype(int)
    # Rating deviation of a review
    rating_deviation = np.zeros(review_n)
    # Thresholded rating deviation of review
    deviated_rating = np.zeros(review_n)
    # Early time frame: early spam reviews can greatly impact people’s sentiment on a product. delta=2 month
    early_time = np.zeros(review_n)
    for product, product_df in df.groupby('product'):
        product_df.sort_values('reviewTime', inplace=True, ascending=True)
        cur_reviews = product_df.index.values
        first_time = product_df['reviewTime'].values[0]
        early_reviews = cur_reviews[(product_df['reviewTime'].values - first_time)<= 3600*24*30*2]
        early_time[early_reviews] = 1
        rating_deviation[cur_reviews] = product_df['star'] - product_df['star'].mean()
        # deviation excluding the review itself.
        if len(cur_reviews)>1:
            deviation = np.abs(product_df['star'] - (product_df['star'].sum()-product_df['star'])/(product_df.shape[0]-1))
            deviated_reviews = cur_reviews[deviation>4*0.63] # 4 on 5-star scale
            deviated_rating[deviated_reviews] = 1
    other_features['rating_deviation'] = rating_deviation
    other_features['deviated_rating'] = deviated_rating
    other_features['early_time'] = early_time
    # Is singleton? If review is user’s sole review
    singleton = np.zeros(review_n)
    for user, user_df in df.groupby('user'):
        cur_reviews = user_df.index
        if len(cur_reviews)==1:
            singleton[cur_reviews[0]] = 1    
    other_features['singleton'] = singleton
    
    print('Review text features')
    # Percentage of ALL-capitals words
    def capital_ratio(text):
        words = text.split(' ')
        cap_n = sum([word.isupper() for word in words])
        return cap_n/len(words)
    other_features['allcap_words_ratio'] = [capital_ratio(text) for text in review_texts]
    # Percentage of capital letters
    def uppercase_ratio(s):
        if not s:  # return 0 if s is an empty string
            return 0
        return sum(1 for c in s if c.isupper()) / len(s)
    other_features['cap_letter_ratio'] = [uppercase_ratio(text) for text in review_texts]
    # Review length in words 
    other_features['length_in_words'] = [len(text.split(' ')) for text in review_texts]    
    # Ratio of 1st person pronouns (‘I’, ‘my‘, etc.)
    def keep_letters(s):
        return re.sub(r'[^a-zA-Z]', '', s)
    def first_pro_ratio(text):
        pronouns = ['i', 'me', 'we', 'us', 'my', 'mine', 'our', 'ours', 'myself', 'ourselves']
        words = np.array([keep_letters(word).lower() for word in text.split(' ')])
        return sum([sum(pronoun==words) for pronoun in pronouns])/len(words)
    other_features['1st_pronun_ratio'] = [first_pro_ratio(text) for text in review_texts]
    # Ratio of exclamation sentences containing ‘!’
    def split_into_sentences(text):
        sentence_delimiters = re.compile('.!?;"()')
        sentences = sentence_delimiters.split(text)
        return sentences
    def exclamation_ratio(text):
        sentences = split_into_sentences(text)
        exclamation_n = len(list(filter(lambda sentence: '!' in sentence, sentences)))
        return exclamation_n/len(sentences)
    other_features['excla_sentence_ratio'] = [exclamation_ratio(text) for text in review_texts]    
    # Ratio of subjective and objective words (by sentiWordNet)
    sentiwordnet = get_sentiwordnet()
    def get_words(text, lower=True):
        words = re.findall(r'\b\w+\b', text)
        return words
    def sub_obj_ratio(text):
        words = get_words(text)
        sub_score = []
        obj_score = []
        for word in words:
            if word in sentiwordnet:
                sub_score.append(sentiwordnet[word])
                obj_score.append(1-sentiwordnet[word])
            else:
                sub_score.append(0)
                obj_score.append(0)
        return np.average(sub_score), np.average(obj_score)
    sub_obj_ratios = np.array(list(map(sub_obj_ratio, review_texts)))
    other_features['subjective_word_ratio'] = sub_obj_ratios[:,0]
    other_features['objective_word_ratio'] = sub_obj_ratios[:,1]
    other_features = pd.DataFrame(other_features)
    
    print('User-related features')
    user_features = np.zeros((review_n, 13))
    def text2ngrams(texts, ngram_range=(1,2)):
        try:
            bigram_vectorizer = CountVectorizer(ngram_range=ngram_range,
                                                token_pattern=r'\b\w+\b', min_df=1)
            X = bigram_vectorizer.fit_transform(texts)
            return X
        except:
            return None
    def max_ave_similarity(X):
        sim_mat = cosine_similarity(X)
        return sim_mat.max(), sim_mat.mean()
    for user, user_df in tqdm(df.groupby('user')):
        cur_reviews = user_df.index
        # Max. number of reviews written in a day
        user_features[cur_reviews, 0] = user_df['reviewDay'].value_counts().max()
        # Max. number of reviews written in a month
        user_features[cur_reviews, 1] = user_df['reviewMonth'].value_counts().max()    
        # Ratio of positive reviews (4-5 star)
        user_features[cur_reviews, 2] = np.sum(user_df['star']>=4)/len(cur_reviews)
        # Ratio of negative reviews (1-2 star) 
        user_features[cur_reviews, 3] = np.sum(user_df['star']<=2)/len(cur_reviews)
        # Avg rating 
        user_features[cur_reviews, 4] = user_df['star'].mean()
        if len(cur_reviews)>1:
            user_features[cur_reviews, 5] = user_df['star'].std()
            # Entropy of rating distribution
            pk = [np.sum(user_df['star']==i) for i in range(5)]
            user_features[cur_reviews, 6] = scipy.stats.entropy(pk)
            # Entropy of temporal gaps (in day and month)
            for interval, feature_dim in zip([3600*24, 3600*24*30], [7,8]):
                user_time = user_df['reviewTime'].values/interval
                user_time = np.sort(user_time)
                user_gap = np.round(np.diff(user_time)).astype(int)
                MAX_GAP =1000
                pk = np.zeros(MAX_GAP)
                for gap in user_gap:
                    if gap<MAX_GAP:
                        pk[gap] +=1
                user_features[cur_reviews, feature_dim] = scipy.stats.entropy(pk)
            # Avg./Max. content similarity
            bigrams = text2ngrams(texts=[review_texts[r] for r in cur_reviews])
            if bigrams is not None:
                max_sim, ave_sim = max_ave_similarity(bigrams)
                user_features[cur_reviews, 4] = max_sim
                user_features[cur_reviews, 5] = ave_sim
            user_features[cur_reviews, 9] = max_sim
            user_features[cur_reviews, 10] = ave_sim
        # Avg rating deviation
        user_features[cur_reviews, 11] = other_features.loc[cur_reviews,'rating_deviation'].mean()
        # Avg. review length in number of words
        user_features[cur_reviews, 12] = other_features.loc[cur_reviews,'length_in_words'].mean()
        
    print('Product-related features')
    product_features = np.zeros((review_n, 13))
    for user, user_df in tqdm(df.groupby('product')):
        cur_reviews = user_df.index
        # Max. number of reviews written in a day
        product_features[cur_reviews, 0] = user_df['reviewDay'].value_counts().max()
        # Max. number of reviews written in a month
        product_features[cur_reviews, 1] = user_df['reviewMonth'].value_counts().max()    
        # Ratio of positive reviews (4-5 star)
        product_features[cur_reviews, 2] = np.sum(user_df['star']>=4)/len(cur_reviews)
        # Ratio of negative reviews (1-2 star) 
        product_features[cur_reviews, 3] = np.sum(user_df['star']<=2)/len(cur_reviews)
        # Avg rating 
        product_features[cur_reviews, 4] = user_df['star'].mean()
        if len(cur_reviews)>1:
            product_features[cur_reviews, 5] = user_df['star'].std()
            # Entropy of rating distribution
            pk = [np.sum(user_df['star']==i) for i in range(5)]
            product_features[cur_reviews, 6] = scipy.stats.entropy(pk)
            # Entropy of temporal gaps (in day and month)
            for interval, feature_dim in zip([3600*24, 3600*24*30], [7,8]):
                user_time = user_df['reviewTime'].values/interval
                user_time = np.sort(user_time)
                user_gap = np.round(np.diff(user_time)).astype(int)
                MAX_GAP =1000
                pk = np.zeros(MAX_GAP)
                for gap in user_gap:
                    if gap<MAX_GAP:
                        pk[gap] +=1
                product_features[cur_reviews, feature_dim] = scipy.stats.entropy(pk)
            # Avg./Max. content similarity
            if bigrams is not None:
                max_sim, ave_sim = max_ave_similarity(bigrams)
                user_features[cur_reviews, 4] = max_sim
                user_features[cur_reviews, 5] = ave_sim
            max_sim, ave_sim = max_ave_similarity(bigrams)
            product_features[cur_reviews, 9] = max_sim
            product_features[cur_reviews, 10] = ave_sim
        # Avg rating deviation
        product_features[cur_reviews, 11] = other_features.loc[cur_reviews,'rating_deviation'].mean()
        # Avg. review length in number of words
        product_features[cur_reviews, 12] = other_features.loc[cur_reviews,'length_in_words'].mean()
        
    other_features = np.hstack((other_features.values, user_features, product_features))
    other_features[~np.isfinite(other_features)]=0    
    return other_features

def process_dga_gnn(adjs, features, labels, train_mask, val_mask, test_mask, DATA_PATH='methods/DGA-GNN/data/processed/', data_name='amazon'):
    # 构建图
    graph_data = {
        ("user", "net_upu", "user") : adjs[0].nonzero(),
        ("user", "net_usu", "user") : adjs[1].nonzero(),
        ("user", "net_uvu", "user") : adjs[2].nonzero(),
    }
    
    graph = dgl.heterograph(graph_data)
    graph = dgl.to_bidirected(graph)
    graph.create_formats_()
    graph.ndata['feat'] = torch.FloatTensor(features)
    graph.ndata['label'] = torch.LongTensor(labels)
    graph.ndata['trn_msk'] = torch.tensor(train_mask, dtype=torch.bool) 
    graph.ndata['val_msk'] = torch.tensor(val_mask, dtype=torch.bool) 
    graph.ndata['tst_msk'] = torch.tensor(test_mask, dtype=torch.bool) 
    
    # 保存图数据
    save_graphs(DATA_PATH + data_name + '.dgldata', graph)
    return