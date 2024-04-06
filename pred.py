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