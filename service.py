import yfinance as yf
import feedparser
from textblob import TextBlob
from nsepython import *
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import torch
import torch.nn.functional as F
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

def get_sentiment_polarity(df):
    df['polarity'] = df['title'].apply(lambda x: TextBlob(x).sentiment.polarity)
    #polarity = analysis.sentiment.polarity
    return df

def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        'name': info.get('longName', ''),
        'sector': info.get('sector', ''),
        'industry': info.get('industry', ''),
        'current_price': info.get('currentPrice', 0),
        'market_cap': info.get('marketCap', 0),
        'pe_ratio': info.get('forwardPE', 0),
        'volume': info.get('volume', 0)
    }

def fetch_business_news(ticker, size=10, page=1):#(name, sector, industry, size=10, page=1):
    # ticker = f"""news of {name}, of {sector} sector and {industry} industry, that effect the stock price"""
    # ticker = ticker.replace(' ', '%20')
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


def get_all_tickers():
    """
    Fetches all stock tickers listed on NSE using nsepython.
    """
    tickers = nse_eq_symbols()
    tickers= [s  for s in tickers]
    return tickers

def get_google_news(name, sector, industry):
    prompt = f"""news of {name}, of {sector} sector and {industry} industry, that effect the stock price"""
    prompt = prompt.replace(' ', '%20')
    url = "https://news.google.com/rss/search?q="+prompt
    news_list =[]
    feed = feedparser.parse(url)
    for item in feed.entries:
        news_list.append((item.title))

    df = pd.DataFrame(news_list, columns=['title'])
    
    #news_text = "\n".join([f"- {news}" for news in news_list])
    return df


def get_sentiment_through_news(ticker):
    stock_info = get_stock_info(ticker=ticker)
    df = get_google_news(name=stock_info['name'], sector=stock_info['sector'], industry=stock_info['industry'])
    return str(get_sentiment_polarity(df=df)[(df['polarity'] < -0.1) | (df['polarity'] > 0.1)]['polarity'].mean())

def get_sentiment_through_business_news(ticker):
    # stock_info = get_stock_info(ticker=ticker)
    headlines_timestamp_dict_list, timestamps = fetch_business_news(ticker=ticker)#name=stock_info['name'], sector=stock_info['sector'], industry=stock_info['industry'])
    timestamps.sort()
    timestamps.reverse()
    recency_dict, weight_sum = calculate_recency_weight(timestamps=timestamps)
    sentiment_score_sum = 0
    for headline in headlines_timestamp_dict_list:
        sentiment_score = calculate_SA_Polarity(headline['headline'])
        sa_score = round(float(sentiment_score)*recency_dict[headline['timestamp']],4)
        sentiment_score_sum += sa_score
    return str(round(float(sentiment_score_sum), 3))