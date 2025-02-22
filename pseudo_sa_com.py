# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaForSequenceClassification, RobertaTokenizer
import json
import torch
import torch.nn.functional as F
from transformers import pipeline
import numpy as np
import requests
import time


import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

sentences = [
    "This product is amazing! I absolutely love it.",
    # "The experience was okay, but it could have been better.",
    "I am extremely disappointed with this service.",
    # "there is a shortage of capital, and we need extra financing",  
    # "growth is strong and we have plenty of liquidity", 
    # "there are doubts about our finances",
    "The sun rises in east",
    "profits are flat", "The stock price of the company is increasing rapidly"
]


models = [
    "cardiffnlp/twitter-roberta-base-sentiment-latest", #----
    # "nlptown/bert-base-multilingual-uncased-sentiment",
    # "siebert/sentiment-roberta-large-english",
    # "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    # "finiteautomata/bertweet-base-sentiment-analysis",  #------
    # "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", #------
    # "ProsusAI/finbert",
    # "LHF/finbert-regressor", #---- without tensor
    # "mr8488/distilroberta-finetuned-financial-news-sentiment-analysis", #------
]

def calculate_SA_Polarity(text,model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=False)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, force_download=False)
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze()
        
        probs = F.softmax(logits, dim=0)

        # Compute polarity score: (-1 * neg) + (0 * neutral) + (1 * pos)
        polarity = -probs[0] + probs[2]
        
        return polarity
    except Exception as e:
        print(str(e))
        return -2.5



    # calculate_SA_Polarity(text=text)


ticker = "SBI BANK"
def fetch_business_news(ticker=ticker, size=10, page=1):
    url = f"https://apibs.business-standard.com/search/?type=all&limit={size}&page={page}&keyword={ticker}"

    payload = {}
    headers = {
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-site',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36',
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    res_dict = json.loads(response.text)

    output_lst = []
    for news in res_dict['data']['news']:
        output = {'headline': news['sub_heading'], 'timestamp': news['published_date']}
        output_lst.append(output)

    return output_lst


headlines_timestamp_dict_list = fetch_business_news()
timestamps = []

for headline in headlines_timestamp_dict_list:
    timestamps.append(headline['timestamp'])

def calculate_recency_weight(timestamps):
    latest_time = max(timestamps)
    recency_weights = np.exp(-(latest_time - np.array(timestamps))/ (24*60*60*10))
    time_resp ={}
    sum_np= np.sum(recency_weights)
    for i in range(len(timestamps)):
        time_resp[timestamps[i]] = recency_weights[i]/sum_np

    return time_resp
    # return recency_weights/np.sum(recency_weights)

now = int(time.time())
timestamps.append(now)
timestamps.sort()
timestamps.reverse()

print(calculate_recency_weight(timestamps=timestamps))