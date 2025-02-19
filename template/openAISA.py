from openai import OpenAI
import json
import pandas as pd


data = pd.read_csv('data/Sentiment_analysis_Data.csv')

client = OpenAI(
  api_key=""
)
def calculate_OAI_SA(text):
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    store=True,
    temperature=0,
    messages=[
        {"role": "user", "content": 'You are a sentiment analysis expert.  Analyze the sentiment of the provided text and provide a numerical score, between -1 and 1. '+'Follow bellow json format only for output :{"score":<SCORE>} where <SCORE> is float value of sentiment score between -1 and 1 where negative indicating negative sentiment while positive value indicates positive sentiment.Analyze the sentiment of the following text: '+text}
    ]
    )
    print(completion.choices[0].message.content)
    sentiment_score = json.loads(completion.choices[0].message.content)['score']
    return sentiment_score


data['pseudo_sentiment_score'] = data['text'].apply(lambda x: calculate_OAI_SA(x))

data.to_csv('data/Sentiment_analysis_Data_updated.csv', index=False)