import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
from sklearn.preprocessing import normalize
from transformers import BertModel, BertTokenizer
from transformers import AutoModel, AutoTokenizer

def text2emb_bertbase(review_texts, device, size='base', return_ids=True, max_length=512):
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
            input_ids = tokenizer.encode(text, return_tensors='pt', max_length=max_length, truncation=True).to(device)
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

def text2emb_stella(review_texts, device, llm, batch_size=10, vector_dim=1024, max_length=512):
    # from https://huggingface.co/dunzhang/stella_en_400M_v5#transformers
    model_dir = 'models/'+ llm
    vector_linear_directory = f"2_Dense_{vector_dim}"
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).cuda().eval()
    # you can also use this model without the features of `use_memory_efficient_attention` and `unpad_inputs`. It can be worked in CPU.
    # model = AutoModel.from_pretrained(model_dir, trust_remote_code=True,use_memory_efficient_attention=False,unpad_inputs=False).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    vector_linear = torch.nn.Linear(in_features=model.config.hidden_size, out_features=vector_dim)
    vector_linear_dict = {
        k.replace("linear.", ""): v for k, v in
        torch.load(os.path.join(model_dir, f"{vector_linear_directory}/pytorch_model.bin")).items()
    }
    vector_linear.load_state_dict(vector_linear_dict)
    vector_linear.cuda()
    # Embed the reviews
    review_embs = []
    for i in tqdm(range(0, len(review_texts), batch_size)):
        docs = review_texts[i:(i+batch_size)]     
        # print(docs)
        with torch.no_grad():
            input_data = tokenizer(docs, padding="longest", truncation=True, max_length=max_length, return_tensors="pt")
            input_data = {k: v.cuda() for k, v in input_data.items()}
            attention_mask = input_data["attention_mask"]
            last_hidden_state = model(**input_data)[0]
            last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            docs_vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            docs_vectors = vector_linear(docs_vectors).cpu().numpy()
            review_embs.append(docs_vectors)
            
    review_embs = np.vstack(review_embs)
    review_embs = normalize(review_embs)
    return review_embs

def text2emb_sentrans(review_texts, device, model_name, batch_size=10, max_length=512):
    '''SentenceTransformer'''
    from sentence_transformers import SentenceTransformer
    if 'stella' in model_name: # =='stella_en_400M_v5':
        model = SentenceTransformer('models/'+model_name, trust_remote_code=True)
    else:
        model = SentenceTransformer('models/'+model_name)
    review_embs = []
    for i in tqdm(range(0, len(review_texts), batch_size)):
        input_texts = review_texts[i:(i+batch_size)] 
        if model_name=='jina-embeddings-v3':
            task = 'classification'
            embeddings_1 = model.encode(input_texts, normalize_embeddings=False, task=task, prompt_name=task,)
        else:
            embeddings_1 = model.encode(input_texts, normalize_embeddings=False)
        review_embs.append(embeddings_1)
    review_embs = np.vstack(review_embs)
    if model_name =='Yinka':
        n_dims = 768
        review_embs = review_embs[:, :n_dims]
    review_embs = normalize(review_embs)       
    return review_embs

def text2emb(review_texts, device, llm, batch_size=100, vector_dim=1024 ,max_length=512):
    if llm=='bert-base-uncased':
        return text2emb_bertbase(review_texts, device, size='base', return_ids=False)
    elif llm=='bert-large-uncased':
        return text2emb_bertbase(review_texts, device, size='large', return_ids=False)
    elif 'stella' in llm:
    #     return text2emb_stella(review_texts, device, llm, batch_size=batch_size, vector_dim=vector_dim)
        return text2emb_sentrans(review_texts, device, llm, batch_size=batch_size)
    elif llm in ['xiaobu-embedding-v2', 'Conan-embedding-v1']:
        return text2emb_sentrans(review_texts, device, llm, batch_size=batch_size)
    else:
        print(llm, 'is not supported')

