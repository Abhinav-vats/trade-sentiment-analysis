import yfinance as yf
from finvizfinance.quote import finvizfinance
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta
from textblob import TextBlob
from langchain_community.llms import Ollama

# Initialize Ollama with Gemma model
llm = Ollama(model="hf.co/bartowski/gemma-2-9b-it-GGUF:IQ2_S")

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .summary-box {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 5px solid;
    }
    .positive-summary {
        border-left-color: #00cc00;
    }
    .negative-summary {
        border-left-color: #ff4444;
    }
    .neutral-summary {
        border-left-color: #999999;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.1:
        return 'POSITIVE'
    elif polarity < -0.1:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'

def summarize_news(news_list, sentiment):
    if not news_list:
        return "No news articles to summarize."
    
    news_text = "\n".join([f"- {news}" for news in news_list])
    prompt = f"""Filter all the {sentiment} news from the following news articles and summarize in 2-3 concise bullet points, and also highlighting the key information:
    {news_text} Provide only the bullet points, no additional text."""

    
    try:
        summary = llm.invoke(prompt)
        return summary.strip()
    except Exception as e:
        return f"Error generating summary: {str(e)}"

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

def get_news_data(ticker):
    stock = finvizfinance(ticker)
    news_df = stock.ticker_news()
    
    news_df['sentiment'] = news_df['Title'].apply(get_sentiment)
    news_df['sentiment_score'] = news_df['Title'].apply(lambda x: TextBlob(x).sentiment.polarity)
    news_df['Date'] = pd.to_datetime(news_df['Date'])
    news_df['time_ago'] = news_df['Date'].apply(
        lambda x: f"{(datetime.now() - x).days}d {(datetime.now() - x).seconds//3600}h ago"
    )
    
    return news_df

def create_sentiment_chart(news_df):
    sentiment_counts = news_df['sentiment'].value_counts()
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="News Sentiment Distribution",
        color_discrete_map={
            'POSITIVE': '#00cc00',
            'NEGATIVE': '#ff4444',
            'NEUTRAL': '#999999'
        }
    )
    fig.update_layout(
        showlegend=True,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff'
    )
    return fig

def main():
    st.sidebar.image("https://img.icons8.com/color/48/000000/stocks.png", width=50)
    st.sidebar.title("Stock Analysis Dashboard")
    ticker = st.sidebar.text_input("Enter Stock Ticker:", value="AAPL").upper()
    
    if st.sidebar.button("Analyze"):
        try:
            st.title(f"Stock Analysis: {ticker}")
            
            # Stock Information
            info = get_stock_info(ticker)
            
            # Metrics Display
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${info['current_price']:,.2f}")
            with col2:
                st.metric("Market Cap", f"${info['market_cap']:,.0f}")
            with col3:
                st.metric("P/E Ratio", f"{info['pe_ratio']:.2f}")
            with col4:
                st.metric("Volume", f"{info['volume']:,}")
            
            # News Analysis
            news_df = get_news_data(ticker)
            
            # Sentiment Distribution
            st.plotly_chart(create_sentiment_chart(news_df), use_container_width=True)
            
            # News Summaries
            st.subheader("News Summaries")
            
            col1, col2 = st.columns(2)
            news = news_df['Title'].tolist()

            
            with col1:
                # Positive News Summary
                st.markdown("""
                    <div class="summary-box positive-summary">
                        <h3>🟢 Positive News Summary</h3>
                    </div>
                """, unsafe_allow_html=True)
                st.write(summarize_news(news, 'POSITIVE'))
                
                # Negative News Summary
                st.markdown("""
                    <div class="summary-box negative-summary">
                        <h3>🔴 Negative News Summary</h3>
                    </div>
                """, unsafe_allow_html=True)
                st.write(summarize_news(news, 'NEGATIVE'))
            
            with col2:
                # Neutral News Summary
                st.markdown("""
                    <div class="summary-box neutral-summary">
                        <h3>⚪ Neutral News Summary</h3>
                    </div>
                """, unsafe_allow_html=True)
                st.write(summarize_news(news, 'NEUTRAL'))
            
            # Detailed News Table
            st.subheader("Latest News")
            news_display = news_df[['Date', 'Title', 'sentiment', 'sentiment_score']].copy()
            news_display['sentiment'] = news_display.apply(
                lambda x: f"{'🟢' if x['sentiment'] == 'POSITIVE' else '🔴' if x['sentiment'] == 'NEGATIVE' else '⚪'} {x['sentiment']} ({x['sentiment_score']:.2f})",
                axis=1
            )
            st.dataframe(
                news_display[['Date', 'Title', 'sentiment']],
                column_config={
                    "Date": "Published",
                    "Title": "Headline",
                    "sentiment": "Sentiment (Score)"
                },
                hide_index=True
            )
            
        except Exception as e:
            st.error(f"Error in analysis: {str(e)}")

if __name__ == "__main__":
    main()