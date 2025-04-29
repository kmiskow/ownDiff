import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

tweets = pd.read_csv("tweets/tweet.csv", low_memory=False)
company_tweet = pd.read_csv("tweets/company_tweet.csv", low_memory=False)
tweets = tweets.merge(company_tweet, how='left', on='tweet_id')
tweets['date'] = pd.to_datetime(tweets['post_date'], unit='s', errors='coerce')
tweets = tweets.dropna(subset=['date'])

start_date = pd.to_datetime("2016-01-04")
end_date = pd.to_datetime("2019-12-30")

aapl_df = tweets[(tweets['ticker_symbol'] == 'TSLA') &
                 (tweets['date'] >= start_date) &
                 (tweets['date'] <= end_date)].copy()
aapl_df = aapl_df.dropna(subset=['body'])

BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.to(device)
model.eval()

def get_scores_batched(texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        scores = probs[:, 2] - probs[:, 0]
    return scores.cpu().numpy()

texts = aapl_df['body'].tolist()
batched_loader = DataLoader(texts, batch_size=BATCH_SIZE)
all_scores = []
for batch in tqdm(batched_loader, desc="Scoring tweets in batches"):
    batch_scores = get_scores_batched(batch)
    all_scores.extend(batch_scores)

aapl_df['score'] = all_scores
aapl_df['date_only'] = aapl_df['date'].dt.date
daily_sentiment = aapl_df.groupby('date_only')['score'].mean().reset_index()
daily_sentiment.columns = ['date', 'avg_sentiment']

os.makedirs("data", exist_ok=True)
daily_sentiment.to_csv("data/TSLA_sentiment_finbert.csv", index=False)
