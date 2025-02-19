from langchain_ollama import OllamaLLM
import json
import pandas as pd

data = pd.read_csv('data/Sentiment_analysis_Data.csv')

llm = OllamaLLM(model="hf.co/bartowski/gemma-2-9b-it-GGUF:IQ2_S")

# text = "This is an amazing product! I love it."
# prompt = 'You are a sentiment analysis expert.  Analyze the sentiment of the provided text and provide a numerical score, between -1 and 1. Follow bellow json format only for output :{"score":<SCORE>} where <SCORE> is float value of sentiment score.Analyze the sentiment of the following text:'+text;
# summary = llm.invoke(prompt)
# sentiment  = json.loads(summary.strip())
# print(sentiment['score'])

def calculate_SA(text:str):
    # print(text)
    prompt = 'You are a sentiment analysis expert.  Analyze the sentiment of the provided text and provide a numerical score, between -1 and 1. '+'Follow bellow json format only for output :{"score":<SCORE>} where <SCORE> is float value of sentiment score between -1 and 1 where negative indicating negative sentiment while positive value indicates positive sentiment.Analyze the sentiment of the following text:'+text;
    # print(prompt)
    summary = llm.invoke(prompt)
    print(summary)
    sentiment  = json.loads(summary.strip())
    return sentiment['score']


# calculate_SA("I'm extremely disappointed with this product; it stopped working within a week.")

data['pseudo_sentiment_score'] = data['text'].apply(lambda x: calculate_SA(x))

data.to_csv('data/Sentiment_analysis_Data_updated.csv', index=False)
