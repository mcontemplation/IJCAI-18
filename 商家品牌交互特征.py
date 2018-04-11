import pandas as pd
import xgboost as xgb
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import time
from datetime import date
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# 从训练集1中提取商家品牌交互特征
train = pd.read_csv('data/train1_f.csv')
s = pd.read_csv('data/shop_feature1.csv')
s = s[['shop_id','shop_click_total','shop_click_buy_total']]
u = s.drop_duplicates()
# 商家被点击该品牌次数
t = train[['shop_id','item_brand_id']]
t['shop_brand_click'] = 1
t = t.groupby(['shop_id','item_brand_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['shop_id','item_brand_id'],how='left')
# 商家被购买该品牌次数
t = train[['shop_id','item_brand_id','is_trade']]
t = t.groupby(['shop_id','item_brand_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'shop_brand_buy'})
train = pd.merge(train,t,on=['shop_id','item_brand_id'],how='left')
# 商家被购买该品牌次数占该商家总购买比率
train = pd.merge(train,u[['shop_id','shop_click_buy_total']],on='shop_id',how='left')
train['shop_brand_rate'] = train['shop_brand_buy']/train['shop_click_buy_total']
# 商家被点击该品牌次数占该商家总点击比率
train = pd.merge(train,u[['shop_id','shop_click_total']],on='shop_id',how='left')
train['shop_brand_crate'] = train['shop_brand_click']/train['shop_click_total']
train[['shop_brand_click','shop_brand_buy','shop_brand_rate','shop_brand_crate','shop_id','item_brand_id']].to_csv('data/shop_brand_feature1.csv',index=None)


# 从训练集2中提取商家品牌交互特征
train = pd.read_csv('data/train2_f.csv')
s = pd.read_csv('data/shop_feature2.csv')
s = s[['shop_id','shop_click_total','shop_click_buy_total']]
u = s.drop_duplicates()
# 商家被点击该品牌次数
t = train[['shop_id','item_brand_id']]
t['shop_brand_click'] = 1
t = t.groupby(['shop_id','item_brand_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['shop_id','item_brand_id'],how='left')
# 商家被购买该品牌次数
t = train[['shop_id','item_brand_id','is_trade']]
t = t.groupby(['shop_id','item_brand_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'shop_brand_buy'})
train = pd.merge(train,t,on=['shop_id','item_brand_id'],how='left')
# 商家被购买该品牌次数占该商家总购买比率
train = pd.merge(train,u[['shop_id','shop_click_buy_total']],on='shop_id',how='left')
train['shop_brand_rate'] = train['shop_brand_buy']/train['shop_click_buy_total']
# 商家被点击该品牌次数占该商家总点击比率
train = pd.merge(train,u[['shop_id','shop_click_total']],on='shop_id',how='left')
train['shop_brand_crate'] = train['shop_brand_click']/train['shop_click_total']
train[['shop_brand_click','shop_brand_buy','shop_brand_rate','shop_brand_crate','shop_id','item_brand_id']].to_csv('data/shop_brand_feature2.csv',index=None)



# 从训练集1中提取商家品牌交互特征
train = pd.read_csv('data/train3_f.csv')
s = pd.read_csv('data/shop_feature3.csv')
s = s[['shop_id','shop_click_total','shop_click_buy_total']]
u = s.drop_duplicates()
# 商家被点击该品牌次数
t = train[['shop_id','item_brand_id']]
t['shop_brand_click'] = 1
t = t.groupby(['shop_id','item_brand_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['shop_id','item_brand_id'],how='left')
# 商家被购买该品牌次数
t = train[['shop_id','item_brand_id','is_trade']]
t = t.groupby(['shop_id','item_brand_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'shop_brand_buy'})
train = pd.merge(train,t,on=['shop_id','item_brand_id'],how='left')
# 商家被购买该品牌次数占该商家总购买比率
train = pd.merge(train,u[['shop_id','shop_click_buy_total']],on='shop_id',how='left')
train['shop_brand_rate'] = train['shop_brand_buy']/train['shop_click_buy_total']
# 商家被点击该品牌次数占该商家总点击比率
train = pd.merge(train,u[['shop_id','shop_click_total']],on='shop_id',how='left')
train['shop_brand_crate'] = train['shop_brand_click']/train['shop_click_total']
train[['shop_brand_click','shop_brand_buy','shop_brand_rate','shop_brand_crate','shop_id','item_brand_id']].to_csv('data/shop_brand_feature3.csv',index=None)
