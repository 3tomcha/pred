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

