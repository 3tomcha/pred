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
    header = next(reader)
    print(header)
    for row in reader:
      print(row)

# def get_upper_slope():

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

# read_ohclv()
with open("pred.csv", "a") as f:
  writer = csv.writer(f)
  writer.writerow(arr)
fea = pd.read_csv("pred.csv")  
pred(fea)