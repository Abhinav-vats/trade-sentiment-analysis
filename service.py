import yfinance as yf
import feedparser
from textblob import TextBlob
from nsepython import *

def get_sentiment_polarity(df):
    # print(df.head())
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

def get_all_tickers():
    """
    Fetches all stock tickers listed on NSE using nsepython.
    """
    tickers = nse_eq_symbols()
    tickers= [s + ".NS" for s in tickers]
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