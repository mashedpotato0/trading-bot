import pandas as pd
import requests
from io import StringIO
import yfinance as yf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import logging
from datetime import datetime, timedelta

def fetch_and_filter_nasdaq_stocks_parallel(output_csv='./tickers.csv', start_date='2021-01-01', max_workers=50):
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    response = requests.get(url)
    if response.status_code != 200:
        print("❌ Failed to fetch Nasdaq listing.")
        return

    df = pd.read_csv(StringIO(response.text), sep='|')
    symbols = df[df['Test Issue'] == 'N']['Symbol'].dropna().tolist()
    print(f"🔍 Validating {len(symbols)} tickers in parallel...")

    valid_tickers = []
    rejected_tickers = []

    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    max_allowed_start = start_date_dt + timedelta(days=30)  # allow 30 days grace

    def check_ticker(symbol):
        try:
            logging.getLogger("yfinance").setLevel(logging.CRITICAL)
            logging.getLogger("urllib3").setLevel(logging.CRITICAL)
            warnings.simplefilter(action='ignore', category=UserWarning)

            data = yf.download(symbol, start=start_date, end="2021-01-10", progress=False, threads=False)
            if data.empty:
                return (symbol, "empty data")

            if data['Close'].dropna().empty:
                return (symbol, "no valid Close price")

            earliest_date = data.index.min()
            if earliest_date > max_allowed_start:
                return (symbol, f"earliest data {earliest_date.date()} > allowed {max_allowed_start.date()}")

            return (symbol, None)
        except Exception as e:
            return (symbol, f"exception: {e}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_ticker, symbol): symbol for symbol in symbols}
        for future in tqdm(as_completed(futures), total=len(futures)):
            symbol, reason = future.result()
            if reason is None:
                valid_tickers.append(symbol)
            else:
                rejected_tickers.append((symbol, reason))

    print(f"✅ Valid tickers: {len(valid_tickers)}")
    print(f"❌ Rejected tickers: {len(rejected_tickers)} (showing up to 10):")
    for sym, rsn in rejected_tickers[:10]:
        print(f"  {sym}: {rsn}")

    pd.DataFrame(valid_tickers, columns=['Symbol']).to_csv(output_csv, index=False)
    print(f"✅ Saved valid tickers to '{output_csv}'")

# Run the function
fetch_and_filter_nasdaq_stocks_parallel()
