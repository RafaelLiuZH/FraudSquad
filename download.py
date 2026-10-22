from transformers import BertModel, BertTokenizer
import torch
import os

def download_and_save_model(model_name):
    try:
        # Create models directory if it doesn't exist
        save_path = f'models/{model_name}/'
        os.makedirs(save_path, exist_ok=True)
        
        # Load pre-trained BERT model and tokenizer
        print(f"Downloading {model_name}...")
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        
        # Save model and tokenizer
        print(f"Saving {model_name} to {save_path}...")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Successfully saved {model_name}")
        
    except Exception as e:
        print(f"Error processing {model_name}: {str(e)}")

# Download and save both models
model_names = ['bert-base-uncased', 'bert-large-uncased']
for model_name in model_names:
    download_and_save_model(model_name)
