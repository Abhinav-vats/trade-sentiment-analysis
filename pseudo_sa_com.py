# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaForSequenceClassification, RobertaTokenizer
import json
import torch
import torch.nn.functional as F
from transformers import pipeline
import numpy as np
import requests
import time
from datetime import datetime

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


ticker = "HDFC BANK"
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
    timestamps = []

    for news in res_dict['data']['news']:
        if news['sub_heading'] != "":
            output = {'headline': news['sub_heading'], 'timestamp': news['published_date']}
            output_lst.append(output)
            timestamps.append(news['published_date'])

    return output_lst, timestamps


headlines_timestamp_dict_list, timestamps = fetch_business_news()


def calculate_recency_weight(timestamps):
    latest_time = max(timestamps)
    recency_weights = np.exp(-(latest_time - np.array(timestamps))/ (24*60*60*10))
    time_resp ={}
    sum =0
    sum_np= np.sum(recency_weights)
    for i in range(len(timestamps)):

        val = recency_weights[i]/sum_np
        time_resp[timestamps[i]] = val
        sum +=val

    return time_resp, sum

timestamps.sort()
timestamps.reverse()



recency_dict, weight_sum = calculate_recency_weight(timestamps=timestamps)
response_final = []
sentiment_score_lst = []
sentiment_score_sum = 0
for headline in headlines_timestamp_dict_list:
    sentiment_score = calculate_SA_Polarity(headline['headline'])
    sa_score = round(float(sentiment_score)*recency_dict[headline['timestamp']],4)
    # headline['sentiment_score_weight'] = sa_score
    # headline['sentiment_score'] = round(float(sentiment_score), 2)
    # headline['weight'] = recency_dict[headline['timestamp']]
    # headline['publish_time'] = datetime.fromtimestamp(headline['timestamp']).strftime('%Y-%m-%d %H:%M:%S')

    # response_final.append(headline)
    # sentiment_score_lst.append(sa_score)
    sentiment_score_sum += sa_score


print(round(float(sentiment_score_sum), 3))
