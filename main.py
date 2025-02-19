from twikit import Client
import time
from datetime import datetime
import feedparser
from langchain_community.llms import Ollama


# llm = Ollama(model="hf.co/bartowski/gemma-2-9b-it-GGUF:IQ2_S")
llm = Ollama(model  = "llama3.2:1b")


def get_google_news(name, sector, industry):
    prompt = f"""news of {name}, of {sector} sector and {industry} industry, that effect the stock price"""
    prompt = prompt.replace(' ', '%20')
    url = "https://news.google.com/rss/search?q="+prompt
    news_list =[]
    feed = feedparser.parse(url)
    for item in feed.entries:
        news_list.append(item.title)
    
    news_text = "\n".join([f"- {news}" for news in news_list])
    return news_text

def summarize_news(news_list, sentiment):
    if not news_list:
        return "No news articles to summarize."
    
    #news_text = "\n".join([f"- {news}" for news in news_list])
    prompt = f"""Filter all the {sentiment} news from the following news articles and summarize in 2-3 concise bullet points, and also highlighting the key information:
    {news_list} Provide only the bullet points, no additional text."""

    print(prompt)
    
    try:
        summary = llm.invoke(prompt)
        return summary.strip()
    except Exception as e:
        return f"Error generating summary: {str(e)}"
news = get_google_news('HDFC Bank' , 'Banking', 'Financial')
print(summarize_news(news, "Positive"))

