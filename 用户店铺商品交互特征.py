import pandas as pd
import xgboost as xgb
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import time
from datetime import date
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 从训练集1中提取
train = pd.read_csv('data/train1_f.csv')
u = pd.read_csv('data/user_item_feature1.csv')#user_item_click_total,user_item_click_buy_total
u.drop_duplicates()
s = pd.read_csv('data/shop_feature1.csv')#shop_click_total,shop_click_buy_total
s.drop_duplicates()
i = pd.read_csv('data/item_feature1.csv')#item_click,item_buy
i.drop_duplicates()

# 用户在该店铺点击该商品次数
t = train[['user_id','shop_id','item_id']]
t['u_s_i_click'] = 1
t = t.groupby(['user_id','shop_id','item_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','shop_id','item_id'],how='left')
# 用户在该店铺购买该商品次数
t = train[['user_id','shop_id','item_id','is_trade']]
t = t.groupby(['user_id','shop_id','item_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'u_s_i_buy'})
train = pd.merge(train,t,on=['user_id','shop_id','item_id'],how='left')
# 用户在该店铺点击该商品次数占用户点击该商品比
t = train[['user_id','shop_id','item_id','u_s_i_click']]
t = pd.merge(t,u[['user_item_click_total','user_id','item_id']],on=['user_id','item_id'],how='left')
t['u_s_i_uclick_rate'] = t['u_s_i_click']/t['user_item_click_total']
train = pd.merge(train,t[['user_id','shop_id','item_id','u_s_i_uclick_rate']],on=['user_id','shop_id','item_id'],how='left')

train[['u_s_i_click','u_s_i_buy','u_s_i_uclick_rate','user_id','shop_id','item_id']].to_csv('data/user_shop_item_feature1.csv',index=None)

# 从训练集2中提取
train = pd.read_csv('data/train2_f.csv')
u = pd.read_csv('data/user_item_feature2.csv')#user_item_click_total,user_item_click_buy_total
u.drop_duplicates()
s = pd.read_csv('data/shop_feature2.csv')#shop_click_total,shop_click_buy_total
s.drop_duplicates()
i = pd.read_csv('data/item_feature2.csv')#item_click,item_buy
i.drop_duplicates()

# 用户在该店铺点击该商品次数
t = train[['user_id','shop_id','item_id']]
t['u_s_i_click'] = 1
t = t.groupby(['user_id','shop_id','item_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','shop_id','item_id'],how='left')
# 用户在该店铺购买该商品次数
t = train[['user_id','shop_id','item_id','is_trade']]
t = t.groupby(['user_id','shop_id','item_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'u_s_i_buy'})
train = pd.merge(train,t,on=['user_id','shop_id','item_id'],how='left')
# 用户在该店铺点击该商品次数占用户点击该商品比
t = train[['user_id','shop_id','item_id','u_s_i_click']]
t = pd.merge(t,u[['user_item_click_total','user_id','item_id']],on=['user_id','item_id'],how='left')
t['u_s_i_uclick_rate'] = t['u_s_i_click']/t['user_item_click_total']
train = pd.merge(train,t[['user_id','shop_id','item_id','u_s_i_uclick_rate']],on=['user_id','shop_id','item_id'],how='left')

train[['u_s_i_click','u_s_i_buy','u_s_i_uclick_rate','user_id','shop_id','item_id']].to_csv('data/user_shop_item_feature2.csv',index=None)


# 从训练集3中提取
train = pd.read_csv('data/train3_f.csv')
u = pd.read_csv('data/user_item_feature3.csv')#user_item_click_total,user_item_click_buy_total
u.drop_duplicates()
s = pd.read_csv('data/shop_feature3.csv')#shop_click_total,shop_click_buy_total
s.drop_duplicates()
i = pd.read_csv('data/item_feature3.csv')#item_click,item_buy
i.drop_duplicates()

# 用户在该店铺点击该商品次数
t = train[['user_id','shop_id','item_id']]
t['u_s_i_click'] = 1
t = t.groupby(['user_id','shop_id','item_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','shop_id','item_id'],how='left')
# 用户在该店铺购买该商品次数
t = train[['user_id','shop_id','item_id','is_trade']]
t = t.groupby(['user_id','shop_id','item_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'u_s_i_buy'})
train = pd.merge(train,t,on=['user_id','shop_id','item_id'],how='left')
# 用户在该店铺点击该商品次数占用户点击该商品比
t = train[['user_id','shop_id','item_id','u_s_i_click']]
t = pd.merge(t,u[['user_item_click_total','user_id','item_id']],on=['user_id','item_id'],how='left')
t['u_s_i_uclick_rate'] = t['u_s_i_click']/t['user_item_click_total']
train = pd.merge(train,t[['user_id','shop_id','item_id','u_s_i_uclick_rate']],on=['user_id','shop_id','item_id'],how='left')

train[['u_s_i_click','u_s_i_buy','u_s_i_uclick_rate','user_id','shop_id','item_id']].to_csv('data/user_shop_item_feature3.csv',index=None)
