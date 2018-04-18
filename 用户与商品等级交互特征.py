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
u = pd.read_csv('data/user_feature1.csv')
#用户点击该价格等级的商品数量
t = train[['user_id','item_price_level']]
t['user_price_ctotal'] = 1
t = t.groupby(['user_id','item_price_level']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_price_level'],how='left')
# 用户购买该价格等级商品的数量
t = train[['user_id','item_price_level','is_trade']]
t = t.groupby(['user_id','item_price_level']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_price_btotal'})
train = pd.merge(train,t,on=['user_id','item_price_level'],how='left')
# 用户购买该价格等级商品占用户购买比
u = u[['user_id','user_click_total','user_click_buy_total']]
u = u.drop_duplicates(subset='user_id')
train = pd.merge(train,u,on='user_id',how='left')
train['user_price_crate'] = train['user_price_ctotal']/train['user_click_total']
train['user_price_brate'] = train['user_price_btotal']/train['user_click_buy_total']
train[['user_price_ctotal','user_price_btotal','user_price_crate','user_price_brate','user_id','item_price_level']].to_csv('data/user_price_feature1.csv',index=None)


#用户点击该收藏等级的商品数量
t = train[['user_id','item_collected_level']]
t['user_collected_ctotal'] = 1
t = t.groupby(['user_id','item_collected_level']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_collected_level'],how='left')
# 用户购买该收藏等级商品的数量
t = train[['user_id','item_collected_level','is_trade']]
t = t.groupby(['user_id','item_collected_level']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_collected_btotal'})
train = pd.merge(train,t,on=['user_id','item_collected_level'],how='left')
# 用户购买该价格等级商品占用户购买比
# print(train[['user_collected_ctotal']])
train['user_collected_crate'] = train['user_collected_ctotal']/train['user_click_total']
train['user_collected_brate'] = train['user_collected_btotal']/train['user_click_buy_total']
train[['user_collected_ctotal','user_collected_crate','user_collected_brate','user_id','item_collected_level']].to_csv('data/user_collected_feature1.csv',index=None)
# 'user_collected_btotal'

# 该星级
# 该年龄点击该类目次数
t = train[['user_star_level','item_price_level']]
t['star_price_click'] = 1
t = t.groupby(['user_star_level','item_price_level']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_star_level','item_price_level'],how='left')
# 该年龄购买该类目次数
t = train[['user_star_level','item_price_level','is_trade']]
t = t.groupby(['user_star_level','item_price_level']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'star_price_buy'})
train = pd.merge(train,t,on=['user_star_level','item_price_level'],how='left')
# 该年龄购买该类别率
train['star_price_rate'] = train['star_price_buy']/train['star_price_click']

train[['star_price_rate','user_star_level','item_price_level']].to_csv('data/star_price_feature1.csv',index=None)


# 该星级与商品销量
# 该年龄点击该类目次数
t = train[['user_star_level','item_sales_level']]
t['star_sale_click'] = 1
t = t.groupby(['user_star_level','item_sales_level']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_star_level','item_sales_level'],how='left')
# 该年龄购买该类目次数
t = train[['user_star_level','item_sales_level','is_trade']]
t = t.groupby(['user_star_level','item_sales_level']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'star_sale_buy'})
train = pd.merge(train,t,on=['user_star_level','item_sales_level'],how='left')
# 该年龄购买该类别率
train['star_sale_rate'] = train['star_sale_buy']/train['star_sale_click']
train[['star_sale_rate','user_star_level','item_sales_level']].to_csv('data/star_sale_feature1.csv',index=None)



# 训练集2
train = pd.read_csv('data/train2_f.csv')
u = pd.read_csv('data/user_feature2.csv')
#用户点击该价格等级的商品数量
t = train[['user_id','item_price_level']]
t['user_price_ctotal'] = 1
t = t.groupby(['user_id','item_price_level']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_price_level'],how='left')
# 用户购买该价格等级商品的数量
t = train[['user_id','item_price_level','is_trade']]
t = t.groupby(['user_id','item_price_level']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_price_btotal'})
train = pd.merge(train,t,on=['user_id','item_price_level'],how='left')
# 用户购买该价格等级商品占用户购买比
u = u[['user_id','user_click_total','user_click_buy_total']]
u = u.drop_duplicates(subset='user_id')
train = pd.merge(train,u,on='user_id',how='left')
train['user_price_crate'] = train['user_price_ctotal']/train['user_click_total']
train['user_price_brate'] = train['user_price_btotal']/train['user_click_buy_total']
train[['user_price_ctotal','user_price_btotal','user_price_crate','user_price_brate','user_id','item_price_level']].to_csv('data/user_price_feature2.csv',index=None)

#用户点击该收藏等级的商品数量
train = pd.read_csv('data/train2_f.csv')
u = pd.read_csv('data/user_feature2.csv')
t = train[['user_id','item_collected_level']]
t['user_collected_ctotal'] = 1
t = t.groupby(['user_id','item_collected_level']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_collected_level'],how='left')
# 用户购买该价格等级商品的数量
t = train[['user_id','item_collected_level','is_trade']]
t = t.groupby(['user_id','item_collected_level']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_collected_btotal'})
train = pd.merge(train,t,on=['user_id','item_collected_level'],how='left')
# 用户购买该价格等级商品占用户购买比
u = u[['user_id','user_click_total','user_click_buy_total']]
u = u.drop_duplicates(subset='user_id')
train = pd.merge(train,u,on=['user_id'],how='left')
# print(train[['user_collected_ctotal']])
train['user_collected_crate'] = train['user_collected_ctotal']/train['user_click_total']
train['user_collected_brate'] = train['user_collected_btotal']/train['user_click_buy_total']
train[['user_collected_ctotal','user_collected_crate','user_collected_brate','user_id','item_collected_level']].to_csv('data/user_collected_feature2.csv',index=None)
# ,'user_collected_btotal'


# 该星级
# 该年龄点击该类目次数
t = train[['user_star_level','item_price_level']]
t['star_price_click'] = 1
t = t.groupby(['user_star_level','item_price_level']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_star_level','item_price_level'],how='left')
# 该年龄购买该类目次数
t = train[['user_star_level','item_price_level','is_trade']]
t = t.groupby(['user_star_level','item_price_level']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'star_price_buy'})
train = pd.merge(train,t,on=['user_star_level','item_price_level'],how='left')
# 该年龄购买该类别率
train['star_price_rate'] = train['star_price_buy']/train['star_price_click']

train[['star_price_rate','user_star_level','item_price_level']].to_csv('data/star_price_feature2.csv',index=None)


# 该星级与商品销量
# 该年龄点击该类目次数
t = train[['user_star_level','item_sales_level']]
t['star_sale_click'] = 1
t = t.groupby(['user_star_level','item_sales_level']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_star_level','item_sales_level'],how='left')
# 该年龄购买该类目次数
t = train[['user_star_level','item_sales_level','is_trade']]
t = t.groupby(['user_star_level','item_sales_level']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'star_sale_buy'})
train = pd.merge(train,t,on=['user_star_level','item_sales_level'],how='left')
# 该年龄购买该类别率
train['star_sale_rate'] = train['star_sale_buy']/train['star_sale_click']
train[['star_sale_rate','user_star_level','item_sales_level']].to_csv('data/star_sale_feature2.csv',index=None)




# 训练集3
train = pd.read_csv('data/train3_f.csv')
u = pd.read_csv('data/user_feature3.csv')
#用户点击该价格等级的商品数量
t = train[['user_id','item_price_level']]
t['user_price_ctotal'] = 1
t = t.groupby(['user_id','item_price_level']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_price_level'],how='left')
# 用户购买该价格等级商品的数量
t = train[['user_id','item_price_level','is_trade']]
t = t.groupby(['user_id','item_price_level']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_price_btotal'})
train = pd.merge(train,t,on=['user_id','item_price_level'],how='left')
# 用户购买该价格等级商品占用户购买比
u = u[['user_id','user_click_total','user_click_buy_total']]
u = u.drop_duplicates(subset='user_id')
train = pd.merge(train,u,on='user_id',how='left')
train['user_price_crate'] = train['user_price_ctotal']/train['user_click_total']
train['user_price_brate'] = train['user_price_btotal']/train['user_click_buy_total']
train[['user_price_ctotal','user_price_btotal','user_price_crate','user_price_brate','user_id','item_price_level']].to_csv('data/user_price_feature3.csv',index=None)

#用户点击该收藏等级的商品数量
train = pd.read_csv('data/train3_f.csv')
u = pd.read_csv('data/user_feature3.csv')
t = train[['user_id','item_collected_level']]
t['user_collected_ctotal'] = 1
t = t.groupby(['user_id','item_collected_level']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_collected_level'],how='left')
# 用户购买该价格等级商品的数量
t = train[['user_id','item_collected_level','is_trade']]
t = t.groupby(['user_id','item_collected_level']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_collected_btotal'})
train = pd.merge(train,t,on=['user_id','item_collected_level'],how='left')
# 用户购买该价格等级商品占用户购买比
u = u[['user_id','user_click_total','user_click_buy_total']]
u = u.drop_duplicates(subset='user_id')
train = pd.merge(train,u,on=['user_id'],how='left')
# print(train[['user_collected_ctotal']])
train['user_collected_crate'] = train['user_collected_ctotal']/train['user_click_total']
train['user_collected_brate'] = train['user_collected_btotal']/train['user_click_buy_total']
train[['user_collected_ctotal','user_collected_crate','user_collected_brate','user_id','item_collected_level']].to_csv('data/user_collected_feature3.csv',index=None)
# 'user_collected_btotal'

# 该星级
# 该年龄点击该类目次数
t = train[['user_star_level','item_price_level']]
t['star_price_click'] = 1
t = t.groupby(['user_star_level','item_price_level']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_star_level','item_price_level'],how='left')
# 该年龄购买该类目次数
t = train[['user_star_level','item_price_level','is_trade']]
t = t.groupby(['user_star_level','item_price_level']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'star_price_buy'})
train = pd.merge(train,t,on=['user_star_level','item_price_level'],how='left')
# 该年龄购买该类别率
train['star_price_rate'] = train['star_price_buy']/train['star_price_click']

train[['star_price_rate','user_star_level','item_price_level']].to_csv('data/star_price_feature3.csv',index=None)


# 该星级与商品销量
# 该年龄点击该类目次数
t = train[['user_star_level','item_sales_level']]
t['star_sale_click'] = 1
t = t.groupby(['user_star_level','item_sales_level']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_star_level','item_sales_level'],how='left')
# 该年龄购买该类目次数
t = train[['user_star_level','item_sales_level','is_trade']]
t = t.groupby(['user_star_level','item_sales_level']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'star_sale_buy'})
train = pd.merge(train,t,on=['user_star_level','item_sales_level'],how='left')
# 该年龄购买该类别率
train['star_sale_rate'] = train['star_sale_buy']/train['star_sale_click']
train[['star_sale_rate','user_star_level','item_sales_level']].to_csv('data/star_sale_feature3.csv',index=None)



