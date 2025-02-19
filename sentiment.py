import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import download
from finvizfinance.quote import finvizfinance
from datetime import datetime
import feedparser

# Download the necessary NLTK data
# download('vader_lexicon')


def fetch_news_by_topic(rss_url, topic):
    """
    Fetch all news from an RSS feed related to a specific topic.

    :param rss_url: URL of the RSS feed
    :param topic: Topic to filter news
    :return: List of news articles matching the topic
    """
    # Parse the RSS feed
    feed = feedparser.parse(rss_url)
    filtered_news = []

    # Filter articles based on the topic
    for entry in feed.entries:
        if topic.lower() in entry.title.lower() or topic.lower() in entry.description.lower():
            filtered_news.append({
                'title': entry.title,
                'link': entry.link,
                'description': entry.description,
                'published': entry.published
            })
    
    return filtered_news

# Initialize the Sentiment Intensity Analyzer
sid = SentimentIntensityAnalyzer()

def get_news_data(ticker):
    stock = finvizfinance(ticker)
    news_df = stock.ticker_news()
    
    # news_df['sentiment'] = news_df['Title'].apply(get_sentiment)
    # news_df['sentiment_score'] = news_df['Title'].apply(lambda x: TextBlob(x).sentiment.polarity)
    # news_df['Date'] = pd.to_datetime(news_df['Date'])
    # news_df['time_ago'] = news_df['Date'].apply(
    #     lambda x: f"{(datetime.now() - x).days}d {(datetime.now() - x).seconds//3600}h ago"
    # )
    
    return news_df

def analyze_sentiment(text):
    """
    Analyze the sentiment of a given text.
    Returns a sentiment score in the range [-1, 1].
    """
    singleText =" ".join([f" {news}" for news in text])
    sentiment = sid.polarity_scores(singleText)
    return sentiment['compound']

# Example dataset of news headlines
data = {
    'headline': [
        "Stock markets tumble as recession fears grow.",
        "The company reports record-breaking profits.",
        "Uncertainty looms ahead of the upcoming elections.",
        "New technological breakthroughs revolutionize the industry.",
        "Natural disaster causes widespread devastation."
    ]
}

def main():
    rss_feed_url = "https://news.google.com/rss/search?q=S&P"  # Example RSS feed URL
    topic = "S&P"  # Topic of interest

    news_articles = fetch_news_by_topic(rss_feed_url, topic)

    # Display the filtered news
    for news in news_articles:
        print(f"Title: {news['title']}")
        print(f"Link: {news['link']}")
        print(f"Published: {news['published']}")
        print(f"Description: {news['description']}")
        print("-" * 80)
        # Create a DataFrame
        #df = get_news_data("S&P")
        print(news_articles["Title"])

    # Add a sentiment column
    #df['sentiment'] = analyze_sentiment(df['headline'])

    # Display the DataFrame
    print(f""" Overall sentiment of the given news is: {analyze_sentiment(news_articles['Title'])}""")

if __name__ == "__main__":
    main()
