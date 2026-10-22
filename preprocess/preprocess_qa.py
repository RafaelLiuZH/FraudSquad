import pandas as pd
import numpy as np
import torch
import dgl
import random
import os
import time
import re
import scipy
import jieba
import scipy.sparse as sp
from tqdm import tqdm
from itertools import combinations
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertModel, BertTokenizer
from dgl.data.utils import load_graphs, save_graphs

def sort_perm(key_indices, keys):
    order = np.argsort(key_indices)
    key_indices_sort = np.array(key_indices)[order]
    keys_sort = np.array(keys)[order]    
    return key_indices_sort, keys_sort

def extract_key_values(line, key_indices_sort, keys_sort):
    entry = {}
    for i, key in enumerate(keys_sort):
        start = key_indices_sort[i] + len(key) + 1
        end = key_indices_sort[i+1]
        if end>start:
            entry[key] = line[start:end].strip(' ')
    return entry

def line2entry(line, keys):
    key_indices = []
    for key in keys:
        if key in line:
            key_indices.append(line.index(key))
        else:
            key_indices.append(len(line))
    key_indices_sort, keys_sort = sort_perm(key_indices, keys)
    end =  len(line) # line.index('aid') if 'aid' in qline else len(qline)
    key_indices_sort = np.concatenate((key_indices_sort, [end]))    
    entry = extract_key_values(line, key_indices_sort, keys_sort)    
    return entry

def read_data_to_df(path, label):
    qkeys = ['qid', 'isAdopted', 'qTitle', 'qCon', 'qTags', 'qTime', 
            'giveScore', 'askerid', 'askerGrade','askerGoodRate',
             'askerAnswerNumber','askerAnswerNumber','askerName']
    akeys = ['aid', 'isBest','ahasComment','aTime','aCon', 
             'answerid','answerName', 'answerCareField','answerGrade','answerGoodRate',
            'answerNum','answerIsMaster','answerIsFamous']
    with open(path, 'r') as handle:
        raw_data = handle.readlines()
    data = []
    for line in tqdm(raw_data):
        qas = line.strip().split('\t')
        # extract qkeys
        qline = qas[0]    
        qentry = line2entry(qline, qkeys)
        if len(qas)==1:
            data.append(qentry)
        # extract akeys
        elif len(qas)>1:
            for aline in qas[1:]:  
                aentry = line2entry(aline, akeys)
                data.append({**qentry, **aentry})
    
    df = pd.DataFrame(data)
    df['label'] = label
    return df

def read_qa_data():
    df_normal = read_data_to_df('data/QA/Ordinary', 0)
    df_fraud = read_data_to_df('data/QA/Deceptive', 1)
    df = pd.concat((df_normal, df_fraud), axis=0)
    df.index = range(df.shape[0])
    return df

#####################################################

def map01(s):
    if type(s) is str:
        if '0' in s:
            return 0
        if '1' in s:
            return 1
        return 0
    else:
        return 0

def mapnum(s):
    if type(s) is str:
        try:
            return float(s)
        except:
            return 0
    return 0
    
def get_df_features(df):
    df_features = df[['isAdopted', 'giveScore', 'askerGrade', 'askerGoodRate', 'askerAnswerNumber', 'isBest',
                      'answerGrade', 'answerGoodRate', 'answerNum', 'answerIsMaster', 'answerIsFamous']]
    for col in ['isAdopted', 'isBest', 'answerIsMaster', 'answerIsFamous']:
        df_features[col] = list(map(map01, df_features[col].values))
    for col in ['giveScore', 'askerGrade', 'askerGoodRate', 'askerAnswerNumber', 'answerGrade', 'answerGoodRate', 'answerNum']:
        df_features[col] = list(map(mapnum, df_features[col].values))
    df_features = df_features.astype(int)
    
    review_n = df.shape[0]
    other_features = np.zeros((review_n, 6))
    cols = ['only_one_answer', 'no_answer', 'has_tag', 'has_desc', 'q_anonymous', 'a_anonymous']
    for qid, qid_df in tqdm(df.groupby('qid')):
        cur_reviews = qid_df.index
        if len(cur_reviews==1):
            if type(qid_df['aCon'].values[0]) is str:
                if len(qid_df['aCon'].values[0].strip())>0:
                    other_features[cur_reviews, 0] = 1 # only one answer
                    other_features[cur_reviews, 1] = 0 # no answers
                else:
                    other_features[cur_reviews, 0] = 0 # only one answer
                    other_features[cur_reviews, 1] = 1 # no answers  
    other_features[:,2] = [(type(i) is str) and (len(i)>0) for i in df['qTags'].values]
    other_features[:,3] = [(type(i) is str) and (len(i)>0) for i in df['qCon'].values]
    other_features[:,4] = [(type(i) is str) and (len(i)>0) for i in df['askerName'].values]
    other_features[:,4] = 1- other_features[:,4]
    other_features[:,5] = [(type(i) is str) and (len(i)>0) for i in df['answerName'].values]
    other_features[:,5] = 1- other_features[:,5]
    
    df_features = pd.concat((df_features,pd.DataFrame(other_features, columns=cols)), axis=1)
    return df_features

#####################################################

def preprocess_df_qa(df, df_type='qa'):
    if df_type=='qa':
        # used df columns: star, product, user, reviewTime (Unix time)
        for col in ['qTime', 'aTime', 'askerid', 'answerid']:
            if col not in df.columns:
                print('Column', col, 'not in the input dataframe')
                return    
        # 2008-12-17 08:39 (example of qTime and aTime)
        def get_day(x):
            if np.isnan(x):
                return ''
            else:
                return datetime.fromtimestamp(x).strftime('%Y-%m-%d')
        def get_month(x):
            if np.isnan(x):
                return ''
            else:
                return datetime.fromtimestamp(x).strftime('%Y-%m')     
        def get_week(x):
            if np.isnan(x):
                return ''
            else:
                dt_object = datetime.fromtimestamp(x)
                year = dt_object.isocalendar()[0]  
                week_number = dt_object.isocalendar()[1]                
                return '%s-%s'%(year, week_number)                    
        def con_str(x): 
            return str(x[0])+'_'+str(x[1])
        def get_unixtime(x):
            if type(x) is str:
                try:
                    date_obj = datetime.strptime(x, '%Y-%m-%d %H:%M')
                    timestamp = date_obj.timestamp()
                    return int(timestamp)
                except:
                    return np.nan
            return np.nan
        df['qUnixTime'] = list(map(get_unixtime, df['qTime']))            
        df['qDay'] = list(map(get_day, df['qUnixTime']))             
        df['qMonth'] = list(map(get_month, df['qUnixTime']))   
        df['qWeek'] = list(map(get_week, df['qUnixTime']))    
        df['aUnixTime'] = list(map(get_unixtime, df['aTime']))            
        df['aDay'] = list(map(get_day, df['aUnixTime']))             
        df['aMonth'] = list(map(get_month, df['aUnixTime']))    
        df['aWeek'] = list(map(get_week, df['aUnixTime']))    
        df['asker_month'] = list(map(con_str, zip(df['askerid'].values, df['qMonth'].values)))
        df['asker_week'] = list(map(con_str, zip(df['askerid'].values, df['qWeek'].values)))
        df['answer_month'] = list(map(con_str, zip(df['answerid'].values, df['aMonth'].values)))       
        df['answer_week'] = list(map(con_str, zip(df['answerid'].values, df['aWeek'].values)))       
    return df

#####################################################

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

def get_fraud_features_qa(df, review_texts, qtexts, atexts):
    # for col in ['askerid', 'answerid', 'reviewDay', 'reviewMonth', 'reviewTime']:
    #     if col not in df.columns:
    #         print('Column', col, 'not in the input dataframe')
    #         return

    review_n = df.shape[0]    
    other_features = {}
    
    print('QA behavioral features')
    # Early time frame: early spam answers can greatly impact people’s sentiment on a question. delta=3 days
    early_time = np.zeros(review_n)
    for product, product_df in tqdm(df.groupby('qid')):
        product_df.sort_values('qUnixTime', inplace=True, ascending=True)
        cur_reviews = product_df.index.values
        first_time = product_df['aUnixTime'].values[0]
        early_reviews = cur_reviews[(product_df['aUnixTime'].values - first_time)<= 3600*24*3]
        early_time[early_reviews] = 1
    other_features['early_time'] = early_time
    # Is singleton? If qa is user’s sole review
    singleton = np.zeros(review_n)
    for user, user_df in df.groupby('askerid'):
        cur_reviews = user_df.index
        if len(cur_reviews)==1:
            singleton[cur_reviews[0]] = 1    
    other_features['singleton_asker'] = singleton
    singleton = np.zeros(review_n)
    for user, user_df in df.groupby('answerid'):
        cur_reviews = user_df.index
        if len(cur_reviews)==1:
            singleton[cur_reviews[0]] = 1    
    other_features['singleton_answer'] = singleton    
    
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
    # Review length in characters 
    other_features['length_in_chars'] = [len(text) for text in review_texts]        
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
        sentence_delimiters = re.compile('.!?;"()。！？；\“\”（）')
        sentences = sentence_delimiters.split(text)
        return sentences
    def exclamation_ratio(text):
        sentences = split_into_sentences(text)
        exclamation_n = len(list(filter(lambda sentence: ('!' in sentence) or ('！' in sentence), sentences)))
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
    
    print('Asker-related features')
    user_features = np.zeros((review_n, 7))
    def text_jieba(texts):
        output_texts = []
        for text in texts:
            if (type(text) is str) and (len(text)>0):
                seg_list = list(jieba.cut(text))
                output_texts.append(' '.join(seg_list))
            else: 
                output_texts.append('')
        return output_texts
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
    def get_char_n(text):
        if type(text) is str:
            return len(text)
        return 0        
    for user, user_df in tqdm(df.groupby('askerid')):
        cur_reviews = user_df.index
        # Max. number of reviews written in a day
        user_features[cur_reviews, 0] = user_df['qDay'].value_counts().max()
        # Max. number of reviews written in a month
        user_features[cur_reviews, 1] = user_df['qMonth'].value_counts().max()    
        if len(cur_reviews)>1:
            # Entropy of temporal gaps (in day and month)
            for interval, feature_dim in zip([3600*24, 3600*24*30], [2, 3]):
                user_time = user_df['qUnixTime'].values/interval
                user_time = user_time[~np.isnan(user_time)]
                if len(user_time) <=1:
                    continue                 
                user_time = np.sort(user_time)
                user_gap = np.round(np.diff(user_time)).astype(int)
                MAX_GAP =1000
                pk = np.zeros(MAX_GAP)
                for gap in user_gap:
                    if gap<MAX_GAP:
                        pk[gap] +=1
                user_features[cur_reviews, feature_dim] = scipy.stats.entropy(pk)
            # Avg./Max. content similarity
            bigrams = text2ngrams(text_jieba([qtexts[r] for r in cur_reviews]))
            if bigrams is not None:
                max_sim, ave_sim = max_ave_similarity(bigrams)
                user_features[cur_reviews, 4] = max_sim
                user_features[cur_reviews, 5] = ave_sim
            else:
                user_features[cur_reviews, 4] = 0
                user_features[cur_reviews, 5] = 0                
            # Avg. review length in number of characters
            user_features[cur_reviews, 6] = np.average([get_char_n(qtexts[ii]) for ii in cur_reviews])
    asker_features = user_features.copy()
        
    print('Answer-related features')
    user_features = np.zeros((review_n, 7))
    for user, user_df in tqdm(df.groupby('answerid')):
        cur_reviews = user_df.index
        # Max. number of reviews written in a day
        user_features[cur_reviews, 0] = user_df['aDay'].value_counts().max()
        # Max. number of reviews written in a month
        user_features[cur_reviews, 1] = user_df['aMonth'].value_counts().max()    
        if len(cur_reviews)>1:
            # Entropy of temporal gaps (in day and month)
            for interval, feature_dim in zip([3600*24, 3600*24*30], [2, 3]):
                user_time = user_df['aUnixTime'].values/interval
                user_time = user_time[~np.isnan(user_time)]
                if len(user_time) <=1:
                    continue                
                user_time = np.sort(user_time)
                user_gap = np.round(np.diff(user_time)).astype(int)
                MAX_GAP =1000
                pk = np.zeros(MAX_GAP)
                for gap in user_gap:
                    if gap<MAX_GAP:
                        pk[gap] +=1
                user_features[cur_reviews, feature_dim] = scipy.stats.entropy(pk)
            # Avg./Max. content similarity
            bigrams = text2ngrams(texts=[atexts[r] for r in cur_reviews])
            if bigrams is not None:
                max_sim, ave_sim = max_ave_similarity(bigrams)
                user_features[cur_reviews, 4] = max_sim
                user_features[cur_reviews, 5] = ave_sim
            else:
                user_features[cur_reviews, 4] = 0
                user_features[cur_reviews, 5] = 0             
            # Avg. review length in number of characters
            user_features[cur_reviews, 6] = np.average([get_char_n(atexts[ii]) for ii in cur_reviews])
    answer_features = user_features.copy()
        
    other_features = np.hstack((other_features.values, asker_features, answer_features))
    other_features[~np.isfinite(other_features)]=0    
    return other_features