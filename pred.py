import csv
from random import shuffle
import ccxt
import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import *
import lightgbm as lgb

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
  return reader

def get_upper_slope():
  high1 = 0
  high2 = 0
  tmp = 0
  ts1 = 0
  ts2 = 0

  with open("ohlcv.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
      if float(row["high"]) > high1 and abs(ts1 - float(row["timestamp"])) > 1800:
        high2 = high1
        ts2 = ts1
        high1 = float(row["high"])
        ts1 = float(row["timestamp"])
    for row in reader:
      if ts1 < float(row["timestamp"]) and abs(ts1 - float(row["timestamp"])) > 1800:
        if tmp < float(row["high"]):
          high2 = float(row["high"])
          ts2 = float(row["timestamp"])
          tmp = high2

    # print(ts1, ts2)
    if ts1 > ts2:
      up_slope = round((high1 - high2) / (ts1 - ts2), 6)
    else:
      up_slope = round((high2 - high1) / (ts2 - ts1), 6)
    
    up_slope = format_float(up_slope)
    arr.append(up_slope)

    return up_slope
  
def get_down_slope():
   low1 = 10000000
   low2 = 10000000
   tmp = 10000000
   ts1 = 0
   ts2 = 0

   with open("ohlcv.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
      if float(row["low"]) < low1 and abs(ts1 - float(row["timestamp"])) > 1800:
         low2 = low1
         ts2 = ts1
         low1 = float(row["low"])
         ts1 = float(row["timestamp"])
    for row in reader:
      if ts1 < float(row["timestamp"]) and abs(ts1 - float(row["timestamp"])) > 1800:
        if tmp > float(row["low"]):
          low2 = float(row["low"])
          ts2 = float(row["timestamp"])
          tmp = low2
    if ts1 > ts2:
       down_slope = round((low1 - low2) / (ts1 - ts2), 6)
    else:
       down_slope = round((low2 - low1) / (ts2 - ts1), 6)

    down_slope = format_float(down_slope)
    arr.append(down_slope)

    return down_slope

def format_float(v):
    s = str(v)
    if 'e' in s:
        i = s.index('e')
        exponent = int(s[i + 1:])
        mantissa = s[:i]
        if '.' not in mantissa:
            frac_len = 0
        else:
            frac_len = len(mantissa) - mantissa.index('.') - 1
        frac_len -= exponent
        if frac_len <= 0:
            s = str(int(v))
        else:
            s = f'%.{int(frac_len)}f' % v
    if '.' not in s:
        s += '.0'
    return s

def pred(arr):
  df = pd.read_csv("train.csv")  
  X = df.drop("#af1hpr", axis=1)
  y = df["#af1hpr"]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
  X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

  lgb_train = lgb.Dataset(X_train, y_train)
  lgb_eval = lgb.Dataset(X_validation, y_validation, reference=lgb_train)
  params = {
    "task": "train",
    "boosting": "gbdt",
    "objective": "regression",
    "metric": {"mse"},
    "num_leaves": 78,
    "drop_rate": 0.05,
    "learing_rate": 0.01,
    "seed": 71,
    "verbose": 0,
    "device": "cpu"
  }
  evaluation_results = {}
  model = lgb.train(params, 
                    lgb_train, 
                    num_boost_round=10000,
                    valid_sets=[lgb_train, lgb_eval],
                    valid_names=["Train", "Valid"],
                    evals_result=evaluation_results,
                    early_stopping_rounds=1000,
                    verbose_eval=100)
  va_pred = model.predict(X_validation)
  feature = arr
  pred = model.predict(feature)

  return(pred)

up_slope = get_upper_slope()
down_slope = get_down_slope()
print(up_slope)
print(down_slope)
# with open("pred.csv", "a") as f:
#   writer = csv.writer(f)
#   writer.writerow(arr)
# fea = pd.read_csv("pred.csv")  
# pred(fea)