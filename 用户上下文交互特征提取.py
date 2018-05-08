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
# 用户点击该页的次数
t = train[['user_id','context_page_id']]
t['user_context_click'] = 1
t = t.groupby(['user_id','context_page_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','context_page_id'],how='left')
# 用户在该页购买的次数
t = train[['user_id','context_page_id','is_trade']]
t = t.groupby(['user_id','context_page_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_context_buy'})
train = pd.merge(train,t,on=['user_id','context_page_id'],how='left')
# 用户在该页的购买率
train['user_context_rate'] = train['user_context_buy']/train['user_context_click']
train[['user_context_click','user_context_buy','user_context_rate','user_id','context_page_id']].to_csv('data/user_context_feature1.csv',index=None)


# 训练集2
train = pd.read_csv('data/train2_f.csv')
# 用户点击该页的次数
t = train[['user_id','context_page_id']]
t['user_context_click'] = 1
t = t.groupby(['user_id','context_page_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','context_page_id'],how='left')
# 用户在该页购买的次数
t = train[['user_id','context_page_id','is_trade']]
t = t.groupby(['user_id','context_page_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_context_buy'})
train = pd.merge(train,t,on=['user_id','context_page_id'],how='left')
# 用户在该页的购买率
train['user_context_rate'] = train['user_context_buy']/train['user_context_click']
train[['user_context_click','user_context_buy','user_context_rate','user_id','context_page_id']].to_csv('data/user_context_feature2.csv',index=None)

# 训练集3
train = pd.read_csv('data/train3_f.csv')
# 用户点击该页的次数
t = train[['user_id','context_page_id']]
t['user_context_click'] = 1
t = t.groupby(['user_id','context_page_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','context_page_id'],how='left')
# 用户在该页购买的次数
t = train[['user_id','context_page_id','is_trade']]
t = t.groupby(['user_id','context_page_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_context_buy'})
train = pd.merge(train,t,on=['user_id','context_page_id'],how='left')
# 用户在该页的购买率
train['user_context_rate'] = train['user_context_buy']/train['user_context_click']
train[['user_context_click','user_context_buy','user_context_rate','user_id','context_page_id']].to_csv('data/user_context_feature3.csv',index=None)



# 训练集4
train = pd.read_csv('data/train4_f.csv')
# 用户点击该页的次数
t = train[['user_id','context_page_id']]
t['user_context_click'] = 1
t = t.groupby(['user_id','context_page_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','context_page_id'],how='left')
# 用户在该页购买的次数
t = train[['user_id','context_page_id','is_trade']]
t = t.groupby(['user_id','context_page_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_context_buy'})
train = pd.merge(train,t,on=['user_id','context_page_id'],how='left')
# 用户在该页的购买率
train['user_context_rate'] = train['user_context_buy']/train['user_context_click']
train[['user_context_click','user_context_buy','user_context_rate','user_id','context_page_id']].to_csv('data/user_context_feature4.csv',index=None)



# # 训练集4
# train = pd.read_csv('data/train5_f.csv')
# # 用户点击该页的次数
# t = train[['user_id','context_page_id']]
# t['user_context_click'] = 1
# t = t.groupby(['user_id','context_page_id']).agg('sum').reset_index()
# train = pd.merge(train,t,on=['user_id','context_page_id'],how='left')
# # 用户在该页购买的次数
# t = train[['user_id','context_page_id','is_trade']]
# t = t.groupby(['user_id','context_page_id']).agg('sum').reset_index()
# t = t.rename(columns={'is_trade':'user_context_buy'})
# train = pd.merge(train,t,on=['user_id','context_page_id'],how='left')
# # 用户在该页的购买率
# train['user_context_rate'] = train['user_context_buy']/train['user_context_click']
# train[['user_context_click','user_context_buy','user_context_rate','user_id','context_page_id']].to_csv('data/user_context_feature5.csv',index=None)

# colormap = plt.cm.RdBu
# plt.figure(figsize=(14,12))
# plt.title('Pearson Correlation of Features', y=1.05, size=15)
# data = train[['user_context_click','user_context_buy','user_context_rate','is_trade']]
# sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0,square=True,cmap=colormap,linecolor='white',annot=True)
# plt.show()
