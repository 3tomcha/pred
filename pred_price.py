import ccxt
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import *
import csv
import seaborn as sns
from sklearn.preprocessing import PowerTransformer

# パラメータ###########################################
# apikeyとsecretkeyを入れてください
by = ccxt.bybit({"apiKey":"","secret":""})

# データ定義###########################################
data = ['up_slope','under_slope','sum_slope']
arr = []

with open('pred.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow([data[0],data[1],data[2]])

# get up_slope and add arr
def get_upper_slope():
    # get__latest_kline##########################################
    kl_15m = by.public_get_kline_list({"symbol":"BTCUSD","interval":"15","from":int(time.time()-900*64),"limit":64})

    # 変数定義 and reset###########################################
    high1 = 0
    high2 = 0
    tmp = 0
    ts1 = 0
    ts2 = 0

    # get up_slope
    for i in range(0,64):
        if float(kl_15m["result"][i]['high']) > high1 and abs(ts1 - kl_15m["result"][i]['open_time'])>1800:
            high2 = high1
            ts2  =  ts1
            high1 = float(kl_15m["result"][i]['high'])
            ts1 = float(kl_15m["result"][i]['open_time'])
        
    for i in range(0,64):
        if ts1 < float(kl_15m["result"][i]['open_time']) and abs(ts1-kl_15m["result"][i]['open_time'])>1800:
            if tmp < float(kl_15m["result"][i]['high']):
                high2 = float(kl_15m["result"][i]['high'])
                ts2 = float(kl_15m["result"][i]['open_time'])
                tmp = high2
    
    if ts1 > ts2:
        up_slope = round((high1-high2)/(ts1-ts2),6)
    else:
        up_slope = round((high2-high1)/(ts2-ts1),6)

    up_slope =  format_float(up_slope)
    # add up_slope
    arr.append(up_slope)
    # return up_slope
    return(up_slope)


# get down_slope and add arr
def get_down_slope():
    # get__latest_kline##########################################
    kl_15m = by.public_get_kline_list({"symbol":"BTCUSD","interval":"15","from":int(time.time()-900*64),"limit":64})

    # 変数定義 and reset###########################################
    low1 = 10000000
    low2 = 10000000
    tmp = 10000000
    ts1 = 0
    ts2 = 0

    # get up_slope
    for i in range(0,64):
        if float(kl_15m["result"][i]['low']) < low1 and abs(ts1-kl_15m["result"][i]['open_time'])>1800:
            low2 = low1
            ts2  =  ts1
            low1 = float(kl_15m["result"][i]['low'])
            ts1 = float(kl_15m["result"][i]['open_time'])
        
    for i in range(0,64):
        if ts1 < float(kl_15m["result"][i]['open_time']) and abs(ts1-kl_15m["result"][i]['open_time'])>1800:
            if tmp > float(kl_15m["result"][i]['low']):
                low2 = float(kl_15m["result"][i]['low'])
                ts2 = float(kl_15m["result"][i]['open_time'])
                tmp = low2
    
    if ts1 > ts2:
        down_slope = round((low1-low2)/(ts1-ts2),6)
    else:
        down_slope = round((low2-low1)/(ts2-ts1),6)

    down_slope =  format_float(down_slope)
    # add down_slope
    arr.append(down_slope)
    # return down_slope
    return(down_slope)

# この関数は少数の指数表現を避けるための関数なので予測に直接は関係ありません
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
    # analyze############################################
    df = pd.read_csv('train.csv')
    X = df.drop('#af1hpr', axis=1)
    y = df['#af1hpr']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_validation, y_validation, reference=lgb_train)
    # LightGBM parameters
    params = {
            'task' : 'train',
            'boosting':'gbdt',
            'objective' : 'regression',
            'metric' : {'mse'},
            'num_leaves':78,
            'drop_rate':0.05,
            'learning_rate':0.01,
            'seed':71,
            'verbose':0,
            'device': 'cpu'
    }

    evaluation_results = {}
    model = lgb.train(params,
                    lgb_train,
                    num_boost_round=10000,
                    valid_sets=[lgb_train, lgb_eval],
                    valid_names=['Train', 'Valid'],
                    evals_result=evaluation_results,
                    early_stopping_rounds=1000,
                    verbose_eval=100)
    va_pred = model.predict(X_validation)
    feature = arr
    pred = model.predict(feature)

    return(pred)

# 特徴量を抽出　データを整理
up_slope = get_upper_slope()
down_slope = get_down_slope()
sum_slope = float(up_slope)+float(down_slope)
sum_slope = format_float(sum_slope)
arr.append(sum_slope)
with open('pred.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(arr)
fea = pd.read_csv('pred.csv')

# 機械学習で予測値を出力
pred = float(pred(fea))
if pred>0:
    print("1時間後の価格は現在の価格＋",pred,"です")
else:
     print("1時間後の価格は現在の価格",pred,"です")