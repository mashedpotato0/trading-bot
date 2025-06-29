import os
import pandas as pd

INPUT_FOLDER = "./stock_data/train"
OUTPUT_FOLDER = "./stock_data/train_fixed"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_13_features(df):
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df = df.apply(pd.to_numeric, errors='coerce')  # 🔧 Fix type issue
    df['Returns'] = df['Close'].pct_change(fill_method=None)  # 🔕 Warning-free
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['Volatility_10'] = df['Returns'].rolling(window=10).std()
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['Price_Change'] = (df['Close'] - df['Open']) / df['Open']
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=15).mean()
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    return df

def process_all_csvs():
    for fname in os.listdir(INPUT_FOLDER):
        if fname.endswith(".csv"):
            try:
                path = os.path.join(INPUT_FOLDER, fname)
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                df = add_13_features(df)
                if df.shape[1] != 13:
                    print(f"❌ Skipped {fname}: got {df.shape[1]} features instead of 13")
                    continue
                df.to_csv(os.path.join(OUTPUT_FOLDER, fname))
                print(f"✅ Processed {fname}")
            except Exception as e:
                print(f"❌ Error processing {fname}: {e}")

if __name__ == "__main__":
    process_all_csvs()
