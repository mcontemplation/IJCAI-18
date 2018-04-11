import pandas as pd
import xgboost as xgb
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import time
from datetime import date
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 训练集1
train = pd.read_csv('data/train1_f.csv')
# 该商户在当前时间点击次数
t = train[['shop_id','hour']]
t['shop_hour_click'] = 1
t = t.groupby(['shop_id','hour']).agg('sum').reset_index()
train = pd.merge(train,t,on=['shop_id','hour'],how='left')
# 该商户在当前时间购买次数
t = train[['shop_id','hour','is_trade']]
t = t.groupby(['shop_id','hour']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'shop_hour_buy'})
train = pd.merge(train,t,on=['shop_id','hour'],how='left')
# 该shanghu在当前时间的购买率
train['shop_hour_rate'] = train['shop_hour_buy']/train['shop_hour_click']
train[['shop_hour_click','shop_hour_buy','shop_hour_rate','shop_id','hour']].to_csv('data/shop_hour_feature1.csv',index=None)

# 训练集2
train = pd.read_csv('data/train2_f.csv')
# 该商户在当前时间点击次数
t = train[['shop_id','hour']]
t['shop_hour_click'] = 1
t = t.groupby(['shop_id','hour']).agg('sum').reset_index()
train = pd.merge(train,t,on=['shop_id','hour'],how='left')
# 该商户在当前时间购买次数
t = train[['shop_id','hour','is_trade']]
t = t.groupby(['shop_id','hour']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'shop_hour_buy'})
train = pd.merge(train,t,on=['shop_id','hour'],how='left')
# 该shanghu在当前时间的购买率
train['shop_hour_rate'] = train['shop_hour_buy']/train['shop_hour_click']
train[['shop_hour_click','shop_hour_buy','shop_hour_rate','shop_id','hour']].to_csv('data/shop_hour_feature2.csv',index=None)

# 训练集3
train = pd.read_csv('data/train3_f.csv')
# 该商户在当前时间点击次数
t = train[['shop_id','hour']]
t['shop_hour_click'] = 1
t = t.groupby(['shop_id','hour']).agg('sum').reset_index()
train = pd.merge(train,t,on=['shop_id','hour'],how='left')
# 该商户在当前时间购买次数
t = train[['shop_id','hour','is_trade']]
t = t.groupby(['shop_id','hour']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'shop_hour_buy'})
train = pd.merge(train,t,on=['shop_id','hour'],how='left')
# 该shanghu在当前时间的购买率
train['shop_hour_rate'] = train['shop_hour_buy']/train['shop_hour_click']
train[['shop_hour_click','shop_hour_buy','shop_hour_rate','shop_id','hour']].to_csv('data/shop_hour_feature3.csv',index=None)
