import pandas as pd
import yfinance as yf

start_date = "2016-01-04"
end_date = "2019-12-31"

def download_stock_data(ticker):
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
    
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    return df

tsla_stock = download_stock_data("TSLA")
aapl_stock = download_stock_data("AAPL")

def load_sentiment_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    if 'Average Score' in df.columns:
        df.rename(columns={'Average Score': 'ts_polarity'}, inplace=True)
    
    return df

tsla_sentiment = load_sentiment_data("data/TSLA_sentiment.csv")
aapl_sentiment = load_sentiment_data("data/AAPL_sentiment.csv")

tsla_sentiment = tsla_sentiment[
    (tsla_sentiment['Date'] >= start_date) & (tsla_sentiment['Date'] <= end_date)
]
aapl_sentiment = aapl_sentiment[
    (aapl_sentiment['Date'] >= start_date) & (aapl_sentiment['Date'] <= end_date)
]

def merge_with_sentiment(stock_df, sentiment_df):
    merged = pd.merge(stock_df, sentiment_df, on='Date', how='left')
    merged.fillna(0, inplace=True)  
    column_map = {col: col.split('_')[0] for col in merged.columns if '_' in col and col.split('_')[0] in ['Open', 'Close', 'High', 'Low', 'Volume']}
    merged.rename(columns=column_map, inplace=True)
    
    return merged

tsla_merged = merge_with_sentiment(tsla_stock, tsla_sentiment)
aapl_merged = merge_with_sentiment(aapl_stock, aapl_sentiment)

tsla_merged.to_csv("TSLA_combined.csv", index=False)
aapl_merged.to_csv("AAPL_combined.csv", index=False)

print("Tesla data sample:")
print(tsla_merged.head())
print("\nApple data sample:")
print(aapl_merged.head())
