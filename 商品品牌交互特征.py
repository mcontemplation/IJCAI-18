import pandas as pd
import xgboost as xgb
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import time
from datetime import date
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 从训练集1中提取商品品牌交互特征
train = pd.read_csv('data/train1_f.csv')
# 点击该商品品牌次数
t = train[['item_id','item_brand_id']]
t['item_brand_click'] = 1
t = t.groupby(['item_id','item_brand_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['item_id','item_brand_id'],how='left')
# 购买该shang品牌次数
t = train[['item_id','item_brand_id','is_trade']]
t = t.groupby(['item_id','item_brand_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'item_brand_buy'})
train = pd.merge(train,t,on=['item_id','item_brand_id'],how='left')
# 购买该shangpin品牌次数占该pinpai总购买比率
t = train[['item_brand_id','is_trade']]
t = t.groupby('item_brand_id').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'brand_buy_total'})
train = pd.merge(train,t,on='item_brand_id',how='left')
train['item_brand_rate'] = train['item_brand_buy']/train['brand_buy_total']
train[['item_brand_rate','item_id','item_brand_id']].to_csv('data/item_brand_feature1.csv',index=None)
# 'item_brand_click','item_brand_buy',
# 从训练集1中提取商品品牌交互特征
train = pd.read_csv('data/train2_f.csv')
# 点击该商品品牌次数
t = train[['item_id','item_brand_id']]
t['item_brand_click'] = 1
t = t.groupby(['item_id','item_brand_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['item_id','item_brand_id'],how='left')
# 购买该shang品牌次数
t = train[['item_id','item_brand_id','is_trade']]
t = t.groupby(['item_id','item_brand_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'item_brand_buy'})
train = pd.merge(train,t,on=['item_id','item_brand_id'],how='left')
# 购买该shangpin品牌次数占该pinpai总购买比率
t = train[['item_brand_id','is_trade']]
t = t.groupby('item_brand_id').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'brand_buy_total'})
train = pd.merge(train,t,on='item_brand_id',how='left')
train['item_brand_rate'] = train['item_brand_buy']/train['brand_buy_total']
train[['item_brand_rate','item_id','item_brand_id']].to_csv('data/item_brand_feature2.csv',index=None)
# 'item_brand_click','item_brand_buy',
# 从训练集1中提取商品品牌交互特征
train = pd.read_csv('data/train3_f.csv')
# 点击该商品品牌次数
t = train[['item_id','item_brand_id']]
t['item_brand_click'] = 1
t = t.groupby(['item_id','item_brand_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['item_id','item_brand_id'],how='left')
# 购买该shang品牌次数
t = train[['item_id','item_brand_id','is_trade']]
t = t.groupby(['item_id','item_brand_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'item_brand_buy'})
train = pd.merge(train,t,on=['item_id','item_brand_id'],how='left')
# 购买该shangpin品牌次数占该pinpai总购买比率
t = train[['item_brand_id','is_trade']]
t = t.groupby('item_brand_id').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'brand_buy_total'})
train = pd.merge(train,t,on='item_brand_id',how='left')
train['item_brand_rate'] = train['item_brand_buy']/train['brand_buy_total']
train[['item_brand_rate','item_id','item_brand_id']].to_csv('data/item_brand_feature3.csv',index=None)
# 'item_brand_click','item_brand_buy',