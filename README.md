# FraudSquad
This is the author implementation of "Detecting LLM-Generated Spam Reviews by Integrating Language Model Embeddings and Graph Neural Network".

## Requirements
```
python 3.10
numpy
scipy
scikit-learn
pandas
torch==2.7.0
dgl==2.4.0+cu121 # https://www.dgl.ai/pages/start.html
``` 

## Repo structures
- `data/`: dataset files
- `models/` models of the large language models and the trained graph neural networks
- `config/`: configuration files for different datasets
- `methods/`: implementation of the FraudSquad
- `preprocess/`: functions to preprocess the dataset files
- `spam_reviews_generation.ipynb`: generating spam reviews from LLMs
- `spam_reviews_evaluation_gpt41.ipynb`: evaluating the generated spam reviews

## Usage
- Download the language model for review text embeddings: `sh download.sh`.
- Unzip `data/Amazon/amazon_2022_5core.csv.zip` and `data/QA.zip`: `cd data/Amazon && unzip amazon_2022_5core.csv.zip` and `cd data && unzip QA.zip -d QA`
- Run FraudSquad on LLM-generated spam review dataset: `python run_detect_amazon.py`. More arguments are listed in the python file. 
- Run FraudSquad on human-written spam review dataset: `python run_detect_yelp_qa.py`. More arguments are listed in the python file.

The generation and evaluation notebooks `spam_reviews_evaluation_gpt41.ipynb` and `spam_reviews_generation.ipynb` need the larger public Amazon dataset. It can be downloaded from https://amazon-reviews-2023.github.io. Both the `review` and `meta` files in the selected category are needed. Besides, it is less organized and maybe only serve for reference.