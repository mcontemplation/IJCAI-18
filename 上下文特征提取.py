import pandas as pd
import xgboost as xgb
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import time
from datetime import date
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 获取训练集一中上下文相关特征
train1_f = pd.read_csv('data/train1_f.csv')
tra = train1_f
# 展示在该页的被点击的次数
t = tra[['context_page_id']]
t['page_click']=1
t = t.groupby('context_page_id').agg('sum').reset_index()
tra = pd.merge(tra,t,on='context_page_id',how='left')
# 展示在该页被购买的次数
t = tra[['context_page_id','is_trade']]
t = t.groupby('context_page_id').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'page_buy'})
tra = pd.merge(tra,t,on='context_page_id',how='left')
# 展示在该页的购买率
tra['page_buy_rate'] = tra['page_buy']/tra['page_click']
# 展示在该页



tra[['page_click','page_buy','page_buy_rate','context_page_id']].to_csv('data/context_feature1.csv',index=None)


# 获取训练集二中上下文相关特征
train1_f = pd.read_csv('data/train2_f.csv')
tra = train1_f
# 展示在该页的被点击的次数
t = tra[['context_page_id']]
t['page_click']=1
t = t.groupby('context_page_id').agg('sum').reset_index()
tra = pd.merge(tra,t,on='context_page_id',how='left')
# 展示在该页被购买的次数
t = tra[['context_page_id','is_trade']]
t = t.groupby('context_page_id').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'page_buy'})
tra = pd.merge(tra,t,on='context_page_id',how='left')
# 展示在该页的购买率
tra['page_buy_rate'] = tra['page_buy']/tra['page_click']



tra[['page_click','page_buy','page_buy_rate','context_page_id']].to_csv('data/context_feature2.csv',index=None)


# 获取训练集三中上下文相关特征
train1_f = pd.read_csv('data/train3_f.csv')
tra = train1_f
# 展示在该页的被点击的次数
t = tra[['context_page_id']]
t['page_click']=1
t = t.groupby('context_page_id').agg('sum').reset_index()
tra = pd.merge(tra,t,on='context_page_id',how='left')
# 展示在该页被购买的次数
t = tra[['context_page_id','is_trade']]
t = t.groupby('context_page_id').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'page_buy'})
tra = pd.merge(tra,t,on='context_page_id',how='left')
# 展示在该页的购买率
tra['page_buy_rate'] = tra['page_buy']/tra['page_click']
# 展示在该页



tra[['page_click','page_buy','page_buy_rate','context_page_id']].to_csv('data/context_feature3.csv',index=None)
