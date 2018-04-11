import pandas as pd
import xgboost as xgb
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import time
from datetime import date
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 从训练集1
train = pd.read_csv('data/train1_f.csv')
# 该用户点击该城市次数
t = train[['user_id','item_city_id']]
t['user_city_click'] = 1
t = t.groupby(['user_id','item_city_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_city_id'],how='left')
# 该用户购买该城市次数
t = train[['user_id','item_city_id','is_trade']]
t = t.groupby(['user_id','item_city_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_city_buy'})
train = pd.merge(train,t,on=['user_id','item_city_id'],how='left')
# 购买率
train['user_city_rate'] = train['user_city_buy']/train['user_city_click']

train[['user_city_click','user_city_buy','user_city_rate','user_id','item_city_id']].to_csv('data/user_city_feature1.csv',index=None)

# 从训练集1
train = pd.read_csv('data/train2_f.csv')
# 该用户点击该城市次数
t = train[['user_id','item_city_id']]
t['user_city_click'] = 1
t = t.groupby(['user_id','item_city_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_city_id'],how='left')
# 该用户购买该城市次数
t = train[['user_id','item_city_id','is_trade']]
t = t.groupby(['user_id','item_city_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_city_buy'})
train = pd.merge(train,t,on=['user_id','item_city_id'],how='left')
# 购买率
train['user_city_rate'] = train['user_city_buy']/train['user_city_click']

train[['user_city_click','user_city_buy','user_city_rate','user_id','item_city_id']].to_csv('data/user_city_feature2.csv',index=None)

# 从训练集1
train = pd.read_csv('data/train3_f.csv')
# 该用户点击该城市次数
t = train[['user_id','item_city_id']]
t['user_city_click'] = 1
t = t.groupby(['user_id','item_city_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_city_id'],how='left')
# 该用户购买该城市次数
t = train[['user_id','item_city_id','is_trade']]
t = t.groupby(['user_id','item_city_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_city_buy'})
train = pd.merge(train,t,on=['user_id','item_city_id'],how='left')
# 购买率
train['user_city_rate'] = train['user_city_buy']/train['user_city_click']

train[['user_city_click','user_city_buy','user_city_rate','user_id','item_city_id']].to_csv('data/user_city_feature3.csv',index=None)