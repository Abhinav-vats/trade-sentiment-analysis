from nsepython import *
import yfinance as yf
from yahoo_fin import news
from llama_index.llms.ollama import Ollama
llm = Ollama(model="hf.co/bartowski/gemma-2-9b-it-GGUF:IQ2_S")
print(llm.completion_to_prompt("Sky is....."))

def get_all_tickers():
    """
    Fetches all stock tickers listed on NSE using nsepython.
    """
    tickers = nse_eq_symbols()
    return tickers

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
ticker = "DOW"
stock_news = news.get_yf_rss(ticker)
print(stock_news)
#print(get_stock_info(ticker))


import requests



def get_news(query, start_date, end_date, api_key, cx):

  url = "https://www.googleapis.com/customsearch/v1"

  params = {

    "q": query, # Search query

    "cx": cx, # Custom Search Engine ID

    "key": api_key, # API key

    "num": 10, # Max results per request

    "sort": f"date:r:{start_date}:{end_date}", # Filter by date range

    "lr": "lang_en", # Language filter (English)

  }

  response = requests.get(url, params=params)

  if response.status_code == 200:

    return response.json()

  else:

    print(f"Error: {response.status_code}, {response.json()}")

    return None



# Replace with your details

API_KEY = "AIzaSyDLBq2dEepcVlTC1OZuqXbOcBJE9kUonkA"

CX = "f78ea88fa698d47de"

QUERY = "Hyundai stock news"

START_DATE = "20240101" # YYYYMMDD

END_DATE = "20241231"  # YYYYMMDD



# Fetch news articles

news_data = get_news(QUERY, START_DATE, END_DATE, API_KEY, CX)

if news_data:

  for item in news_data.get("items", []):

    print(f"Title: {item['title']}")

    print(f"Link: {item['link']}")

    print(f"Snippet: {item['snippet']}\n")

