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
u = pd.read_csv('data/user_feature1.csv')
u = u.drop_duplicates(subset='user_id')
# 该用户点击该类目次数
t = train[['user_id','cate2']]
t['user_cate_click'] = 1
t = t.groupby(['user_id','cate2']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','cate2'],how='left')
# 该用户购买该类目次数
t = train[['user_id','cate2','is_trade']]
t = t.groupby(['user_id','cate2']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_cate_buy'})
train = pd.merge(train,t,on=['user_id','cate2'],how='left')
# 该用户购买该类别率
train['user_cate_rate'] = train['user_cate_buy']/train['user_cate_click']
# 该用户点击该类目占该用户点击比
train = pd.merge(train,u[['user_id','user_click_total','user_click_buy_total']],on='user_id',how='left')
train['user_cate_rate'] = train['user_cate_click']/train['user_click_total']
# 该用户购买该类目占该用户购买比
train['user_cate_brate'] = train['user_cate_buy']/train['user_click_buy_total']
train[['user_cate_click','user_cate_buy','user_cate_rate','user_cate_rate','user_cate_brate','user_id','cate2']].to_csv('data/user_cate_feature1.csv',index=None)

# 该星级
# 该年龄点击该类目次数
t = train[['user_star_level','cate2']]
t['star_cate_click'] = 1
t = t.groupby(['user_star_level','cate2']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_star_level','cate2'],how='left')
# 该年龄购买该类目次数
t = train[['user_star_level','cate2','is_trade']]
t = t.groupby(['user_star_level','cate2']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'star_cate_buy'})
train = pd.merge(train,t,on=['user_star_level','cate2'],how='left')
# 该年龄购买该类别率
train['star_cate_rate'] = train['star_cate_buy']/train['star_cate_click']
# # 该年龄点击该类目占该年龄点击比
#
# train['user_cate_rate'] = train['user_cate_click']/train['user_click_total']
# # 该年龄购买该类目占该年龄购买比
# train['user_cate_brate'] = train['user_cate_buy']/train['user_click_buy_total']
train[['star_cate_rate','user_star_level','cate2']].to_csv('data/star_cate_feature1.csv',index=None)
# 'star_cate_click','star_cate_buy',



# 从训练集2
train = pd.read_csv('data/train2_f.csv')
u = pd.read_csv('data/user_feature2.csv')
u = u.drop_duplicates(subset='user_id')
# 该用户点击该类目次数
t = train[['user_id','cate2']]
t['user_cate_click'] = 1
t = t.groupby(['user_id','cate2']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','cate2'],how='left')
# 该用户购买该类目次数
t = train[['user_id','cate2','is_trade']]
t = t.groupby(['user_id','cate2']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_cate_buy'})
train = pd.merge(train,t,on=['user_id','cate2'],how='left')
# 该用户购买该类别率
train['user_cate_rate'] = train['user_cate_buy']/train['user_cate_click']
# 该用户点击该类目占该用户点击比
train = pd.merge(train,u[['user_id','user_click_total','user_click_buy_total']],on='user_id',how='left')
train['user_cate_rate'] = train['user_cate_click']/train['user_click_total']
# 该用户购买该类目占该用户购买比
train['user_cate_brate'] = train['user_cate_buy']/train['user_click_buy_total']

train[['user_cate_click','user_cate_buy','user_cate_rate','user_cate_rate','user_cate_brate','user_id','cate2']].to_csv('data/user_cate_feature2.csv',index=None)

# 该星级
# 该年龄点击该类目次数
t = train[['user_star_level','cate2']]
t['star_cate_click'] = 1
t = t.groupby(['user_star_level','cate2']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_star_level','cate2'],how='left')
# 该年龄购买该类目次数
t = train[['user_star_level','cate2','is_trade']]
t = t.groupby(['user_star_level','cate2']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'star_cate_buy'})
train = pd.merge(train,t,on=['user_star_level','cate2'],how='left')
# 该年龄购买该类别率
train['star_cate_rate'] = train['star_cate_buy']/train['star_cate_click']
# # 该年龄点击该类目占该年龄点击比
#
# train['user_cate_rate'] = train['user_cate_click']/train['user_click_total']
# # 该年龄购买该类目占该年龄购买比
# train['user_cate_brate'] = train['user_cate_buy']/train['user_click_buy_total']
train[['star_cate_rate','user_star_level','cate2']].to_csv('data/star_cate_feature2.csv',index=None)
# 'star_cate_click','star_cate_buy',





# 从训练集3
train = pd.read_csv('data/train3_f.csv')
u = pd.read_csv('data/user_feature3.csv')
u = u.drop_duplicates(subset='user_id')
# 该用户点击该类目次数
t = train[['user_id','cate2']]
t['user_cate_click'] = 1
t = t.groupby(['user_id','cate2']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','cate2'],how='left')
# 该用户购买该类目次数
t = train[['user_id','cate2','is_trade']]
t = t.groupby(['user_id','cate2']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_cate_buy'})
train = pd.merge(train,t,on=['user_id','cate2'],how='left')
# 该用户购买该类别率
train['user_cate_rate'] = train['user_cate_buy']/train['user_cate_click']
# 该用户点击该类目占该用户点击比
train = pd.merge(train,u[['user_id','user_click_total','user_click_buy_total']],on='user_id',how='left')
train['user_cate_rate'] = train['user_cate_click']/train['user_click_total']
# 该用户购买该类目占该用户购买比
train['user_cate_brate'] = train['user_cate_buy']/train['user_click_buy_total']

train[['user_cate_click','user_cate_buy','user_cate_rate','user_cate_rate','user_cate_brate','user_id','cate2']].to_csv('data/user_cate_feature3.csv',index=None)

# 该星级
# 该年龄点击该类目次数
t = train[['user_star_level','cate2']]
t['star_cate_click'] = 1
t = t.groupby(['user_star_level','cate2']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_star_level','cate2'],how='left')
# 该年龄购买该类目次数
t = train[['user_star_level','cate2','is_trade']]
t = t.groupby(['user_star_level','cate2']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'star_cate_buy'})
train = pd.merge(train,t,on=['user_star_level','cate2'],how='left')
# 该年龄购买该类别率
train['star_cate_rate'] = train['star_cate_buy']/train['star_cate_click']
# # 该年龄点击该类目占该年龄点击比
#
# train['user_cate_rate'] = train['user_cate_click']/train['user_click_total']
# # 该年龄购买该类目占该年龄购买比
# train['user_cate_brate'] = train['user_cate_buy']/train['user_click_buy_total']
train[['star_cate_rate','user_star_level','cate2']].to_csv('data/star_cate_feature3.csv',index=None)
# 'star_cate_click','star_cate_buy',




# 从训练集4
train = pd.read_csv('data/train4_f.csv')
u = pd.read_csv('data/user_feature4.csv')
u = u.drop_duplicates(subset='user_id')
# 该用户点击该类目次数
t = train[['user_id','cate2']]
t['user_cate_click'] = 1
t = t.groupby(['user_id','cate2']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','cate2'],how='left')
# 该用户购买该类目次数
t = train[['user_id','cate2','is_trade']]
t = t.groupby(['user_id','cate2']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_cate_buy'})
train = pd.merge(train,t,on=['user_id','cate2'],how='left')
# 该用户购买该类别率
train['user_cate_rate'] = train['user_cate_buy']/train['user_cate_click']
# 该用户点击该类目占该用户点击比
train = pd.merge(train,u[['user_id','user_click_total','user_click_buy_total']],on='user_id',how='left')
train['user_cate_rate'] = train['user_cate_click']/train['user_click_total']
# 该用户购买该类目占该用户购买比
train['user_cate_brate'] = train['user_cate_buy']/train['user_click_buy_total']

train[['user_cate_click','user_cate_buy','user_cate_rate','user_cate_rate','user_cate_brate','user_id','cate2']].to_csv('data/user_cate_feature4.csv',index=None)

# 该星级
# 该年龄点击该类目次数
t = train[['user_star_level','cate2']]
t['star_cate_click'] = 1
t = t.groupby(['user_star_level','cate2']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_star_level','cate2'],how='left')
# 该年龄购买该类目次数
t = train[['user_star_level','cate2','is_trade']]
t = t.groupby(['user_star_level','cate2']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'star_cate_buy'})
train = pd.merge(train,t,on=['user_star_level','cate2'],how='left')
# 该年龄购买该类别率
train['star_cate_rate'] = train['star_cate_buy']/train['star_cate_click']
# # 该年龄点击该类目占该年龄点击比
#
# train['user_cate_rate'] = train['user_cate_click']/train['user_click_total']
# # 该年龄购买该类目占该年龄购买比
# train['user_cate_brate'] = train['user_cate_buy']/train['user_click_buy_total']
train[['star_cate_rate','user_star_level','cate2']].to_csv('data/star_cate_feature4.csv',index=None)
# 'star_cate_click','star_cate_buy',




# # 从训练集5
# train = pd.read_csv('data/train5_f.csv')
# u = pd.read_csv('data/user_feature5.csv')
# u = u.drop_duplicates(subset='user_id')
# # 该用户点击该类目次数
# t = train[['user_id','cate2']]
# t['user_cate_click'] = 1
# t = t.groupby(['user_id','cate2']).agg('sum').reset_index()
# train = pd.merge(train,t,on=['user_id','cate2'],how='left')
# # 该用户购买该类目次数
# t = train[['user_id','cate2','is_trade']]
# t = t.groupby(['user_id','cate2']).agg('sum').reset_index()
# t = t.rename(columns={'is_trade':'user_cate_buy'})
# train = pd.merge(train,t,on=['user_id','cate2'],how='left')
# # 该用户购买该类别率
# train['user_cate_rate'] = train['user_cate_buy']/train['user_cate_click']
# # 该用户点击该类目占该用户点击比
# train = pd.merge(train,u[['user_id','user_click_total','user_click_buy_total']],on='user_id',how='left')
# train['user_cate_rate'] = train['user_cate_click']/train['user_click_total']
# # 该用户购买该类目占该用户购买比
# train['user_cate_brate'] = train['user_cate_buy']/train['user_click_buy_total']
#
# train[['user_cate_click','user_cate_buy','user_cate_rate','user_cate_rate','user_cate_brate','user_id','cate2']].to_csv('data/user_cate_feature5.csv',index=None)
#
# # 该星级
# # 该年龄点击该类目次数
# t = train[['user_star_level','cate2']]
# t['star_cate_click'] = 1
# t = t.groupby(['user_star_level','cate2']).agg('sum').reset_index()
# train = pd.merge(train,t,on=['user_star_level','cate2'],how='left')
# # 该年龄购买该类目次数
# t = train[['user_star_level','cate2','is_trade']]
# t = t.groupby(['user_star_level','cate2']).agg('sum').reset_index()
# t = t.rename(columns={'is_trade':'star_cate_buy'})
# train = pd.merge(train,t,on=['user_star_level','cate2'],how='left')
# # 该年龄购买该类别率
# train['star_cate_rate'] = train['star_cate_buy']/train['star_cate_click']
# # # 该年龄点击该类目占该年龄点击比
# #
# # train['user_cate_rate'] = train['user_cate_click']/train['user_click_total']
# # # 该年龄购买该类目占该年龄购买比
# # train['user_cate_brate'] = train['user_cate_buy']/train['user_click_buy_total']
# train[['star_cate_rate','user_star_level','cate2']].to_csv('data/star_cate_feature5.csv',index=None)
