import csv
import ccxt
import os
from dotenv import load_dotenv

load_dotenv()

# パラメータ
api_key = os.getenv("API_KEY")
secret = os.getenv("SECRET")
by = ccxt.bybit({
    'apiKey': api_key,
    'secret': secret,
})

# データ定義
data = ["up_slope",  "under_slope", "sum_slope"]
arr = []

with open("pred.csv", "w") as f:
  writer = csv.writer(f)
  writer.writerow([data[0], data[1], data[2]])

def write_ohlcv():
  header = ["timestamp", "open",  "high", "low", "close", "volume"]
  exchange = ccxt.binance()
  symbol = "BTC/USD"
  timeframe = "15m"
  since = exchange.milliseconds() - 900 * 1000 * 64
  limit = 64

  ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)

  with open("ohlcv.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for row in ohlcv:
      writer.writerow(row)

def read_ohclv():
  with open("ohlcv.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    print(header)
    for row in reader:
      print(row)

# def get_upper_slope():
  

read_ohclv()