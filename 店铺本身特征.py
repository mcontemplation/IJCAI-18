import pandas as pd
import xgboost as xgb
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import time
from datetime import date
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 1
tra = pd.read_csv('data/train1_f.csv')

# 该评论等级
t = tra[['shop_review_num_level']]
t['review_click']=1
t = t.groupby('shop_review_num_level').agg('sum').reset_index()
tra = pd.merge(tra,t,on='shop_review_num_level',how='left')
# 该用户星级购买数
t = tra[['shop_review_num_level','is_trade']]
t = t.groupby('shop_review_num_level').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'review_buy'})
tra = pd.merge(tra,t,on='shop_review_num_level',how='left')
# 该用户星级购买率
tra['review_buy_rate'] = tra['review_buy']/tra['review_click']
tra[['review_buy_rate','review_click','review_buy','shop_review_num_level']].to_csv('data/review_feature1.csv',index=None)
# 该店铺星际
t = tra[['shop_star_level']]
t['shop_star_click']=1
t = t.groupby('shop_star_level').agg('sum').reset_index()
tra = pd.merge(tra,t,on='shop_star_level',how='left')
# 该用户星级购买数
t = tra[['shop_star_level','is_trade']]
t = t.groupby('shop_star_level').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'shop_star_buy'})
tra = pd.merge(tra,t,on='shop_star_level',how='left')
# 该用户星级购买率
tra['shop_star_buy_rate'] = tra['shop_star_buy']/tra['shop_star_click']
tra[['shop_star_buy_rate','shop_star_click','shop_star_buy','shop_star_level']].to_csv('data/shop_star_feature1.csv',index=None)


# 2
tra = pd.read_csv('data/train2_f.csv')
# 该评论等级
t = tra[['shop_review_num_level']]
t['review_click']=1
t = t.groupby('shop_review_num_level').agg('sum').reset_index()
tra = pd.merge(tra,t,on='shop_review_num_level',how='left')
# 该用户星级购买数
t = tra[['shop_review_num_level','is_trade']]
t = t.groupby('shop_review_num_level').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'review_buy'})
tra = pd.merge(tra,t,on='shop_review_num_level',how='left')
# 该用户星级购买率
tra['review_buy_rate'] = tra['review_buy']/tra['review_click']
tra[['review_buy_rate','review_click','review_buy','shop_review_num_level']].to_csv('data/review_feature2.csv',index=None)
# 该店铺星际
t = tra[['shop_star_level']]
t['shop_star_click']=1
t = t.groupby('shop_star_level').agg('sum').reset_index()
tra = pd.merge(tra,t,on='shop_star_level',how='left')
# 该用户星级购买数
t = tra[['shop_star_level','is_trade']]
t = t.groupby('shop_star_level').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'shop_star_buy'})
tra = pd.merge(tra,t,on='shop_star_level',how='left')
# 该用户星级购买率
tra['shop_star_buy_rate'] = tra['shop_star_buy']/tra['shop_star_click']
tra[['shop_star_buy_rate','shop_star_click','shop_star_buy','shop_star_level']].to_csv('data/shop_star_feature2.csv',index=None)


# 3
tra = pd.read_csv('data/train3_f.csv')
# 该评论等级
t = tra[['shop_review_num_level']]
t['review_click']=1
t = t.groupby('shop_review_num_level').agg('sum').reset_index()
tra = pd.merge(tra,t,on='shop_review_num_level',how='left')
# 该用户星级购买数
t = tra[['shop_review_num_level','is_trade']]
t = t.groupby('shop_review_num_level').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'review_buy'})
tra = pd.merge(tra,t,on='shop_review_num_level',how='left')
# 该用户星级购买率
tra['review_buy_rate'] = tra['review_buy']/tra['review_click']
tra[['review_buy_rate','review_click','review_buy','shop_review_num_level']].to_csv('data/review_feature3.csv',index=None)

# 该店铺星际
t = tra[['shop_star_level']]
t['shop_star_click']=1
t = t.groupby('shop_star_level').agg('sum').reset_index()
tra = pd.merge(tra,t,on='shop_star_level',how='left')
# 该用户星级购买数
t = tra[['shop_star_level','is_trade']]
t = t.groupby('shop_star_level').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'shop_star_buy'})
tra = pd.merge(tra,t,on='shop_star_level',how='left')
# 该用户星级购买率
tra['shop_star_buy_rate'] = tra['shop_star_buy']/tra['shop_star_click']
tra[['shop_star_buy_rate','shop_star_click','shop_star_buy','shop_star_level']].to_csv('data/shop_star_feature3.csv',index=None)



# 4
tra = pd.read_csv('data/train4_f.csv')
# 该评论等级
t = tra[['shop_review_num_level']]
t['review_click']=1
t = t.groupby('shop_review_num_level').agg('sum').reset_index()
tra = pd.merge(tra,t,on='shop_review_num_level',how='left')
# 该用户星级购买数
t = tra[['shop_review_num_level','is_trade']]
t = t.groupby('shop_review_num_level').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'review_buy'})
tra = pd.merge(tra,t,on='shop_review_num_level',how='left')
# 该用户星级购买率
tra['review_buy_rate'] = tra['review_buy']/tra['review_click']
tra[['review_buy_rate','review_click','review_buy','shop_review_num_level']].to_csv('data/review_feature4.csv',index=None)

# 该店铺星际
t = tra[['shop_star_level']]
t['shop_star_click']=1
t = t.groupby('shop_star_level').agg('sum').reset_index()
tra = pd.merge(tra,t,on='shop_star_level',how='left')
# 该用户星级购买数
t = tra[['shop_star_level','is_trade']]
t = t.groupby('shop_star_level').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'shop_star_buy'})
tra = pd.merge(tra,t,on='shop_star_level',how='left')
# 该用户星级购买率
tra['shop_star_buy_rate'] = tra['shop_star_buy']/tra['shop_star_click']
tra[['shop_star_buy_rate','shop_star_click','shop_star_buy','shop_star_level']].to_csv('data/shop_star_feature4.csv',index=None)



# # 5
# tra = pd.read_csv('data/train5_f.csv')
# # 该评论等级
# t = tra[['shop_review_num_level']]
# t['review_click']=1
# t = t.groupby('shop_review_num_level').agg('sum').reset_index()
# tra = pd.merge(tra,t,on='shop_review_num_level',how='left')
# # 该用户星级购买数
# t = tra[['shop_review_num_level','is_trade']]
# t = t.groupby('shop_review_num_level').agg('sum').reset_index()
# t = t.rename(columns={'is_trade':'review_buy'})
# tra = pd.merge(tra,t,on='shop_review_num_level',how='left')
# # 该用户星级购买率
# tra['review_buy_rate'] = tra['review_buy']/tra['review_click']
# tra[['review_buy_rate','review_click','review_buy','shop_review_num_level']].to_csv('data/review_feature5.csv',index=None)
#
# # 该店铺星际
# t = tra[['shop_star_level']]
# t['shop_star_click']=1
# t = t.groupby('shop_star_level').agg('sum').reset_index()
# tra = pd.merge(tra,t,on='shop_star_level',how='left')
# # 该用户星级购买数
# t = tra[['shop_star_level','is_trade']]
# t = t.groupby('shop_star_level').agg('sum').reset_index()
# t = t.rename(columns={'is_trade':'shop_star_buy'})
# tra = pd.merge(tra,t,on='shop_star_level',how='left')
# # 该用户星级购买率
# tra['shop_star_buy_rate'] = tra['shop_star_buy']/tra['shop_star_click']
# tra[['shop_star_buy_rate','shop_star_click','shop_star_buy','shop_star_level']].to_csv('data/shop_star_feature5.csv',index=None)