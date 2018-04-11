import pandas as pd
import xgboost as xgb
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import time
from datetime import date
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

train1_f = pd.read_csv('data/train1_f.csv')
train1_p = pd.read_csv('data/train1_f.csv')

# 提取训练集一的用户相关特征
# 用户点击次数
tra = train1_f
tra_p = train1_p
t = tra[['user_id']]
t['user_click_total'] = 1
t = t.groupby('user_id')['user_click_total'].agg('sum').reset_index()
tra = pd.merge(tra,t,on='user_id',how='left')

# 用户点击且购买次数
t = tra[['user_id','is_trade']]
t = t.groupby('user_id')['is_trade'].agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_click_buy_total'})
tra = pd.merge(tra,t,on='user_id',how='left')

# 用户购买占点击比重
tra['user_click_buy_rate'] = tra['user_click_buy_total']/tra['user_click_total']

# 用户点击不同商家的数量
t = tra[['user_id','shop_id']]
t = t.drop_duplicates()
t['user_click_difshop_total']=1
t = t.groupby('user_id')['user_click_difshop_total'].agg('sum').reset_index()
tra = pd.merge(tra,t,on='user_id',how='left')

# 用户点不同商家数量占所有不同商家比重
t = tra[['shop_id','user_click_difshop_total','user_id']]
t['a']=1
num = t.groupby('shop_id').agg('sum').reset_index().shape[0]
t['user_click_shop_rate'] = t['user_click_difshop_total']/num
t.drop(['shop_id','user_click_difshop_total','a'],axis=1,inplace=True)
tra = pd.merge(tra,t,on='user_id',how='left')

# 用户点击最多/最少/平均商家次数
t = tra[['user_id','shop_id']]
t['click']=1
t = t.groupby(['user_id','shop_id']).agg('sum').reset_index()
t1 = t.groupby(['user_id'])['click'].agg('max').reset_index()
t2 = t.groupby(['user_id'])['click'].agg('mean').reset_index()
t3 = t.groupby(['user_id'])['click'].agg('min').reset_index()
t1.rename(columns={'click':'user_click_max'},inplace=True)
t2.rename(columns={'click':'user_click_mean'},inplace=True)
t3.rename(columns={'click':'user_click_min'},inplace=True)
tra = pd.merge(tra,t1,on='user_id',how='left')
tra = pd.merge(tra,t2,on='user_id',how='left')
tra = pd.merge(tra,t3,on='user_id',how='left')

# 用户上午点击次数
t = tra[['user_id','is_am']]
t = t.groupby('user_id')['is_am'].agg(sum).reset_index()
t = t.rename(columns={'is_am':'user_click_amtotal'})
tra = pd.merge(tra,t,on='user_id',how='left')

# # 用户购买最大价格最小等级商品
# t = tra[['user_id','item_price_level']]
# t = t.drop_duplicates()
# t = t.groupby('user_id').agg('max').reset_index()
# t = t.rename(columns={'item_price_level':'user_max_price'})
# tra = pd.merge(tra,t,on='user_id',how='left')
# t = tra[['user_id','item_price_level']]
# t = t.drop_duplicates()
# t = t.groupby('user_id').agg('min').reset_index()
# t = t.rename(columns={'item_price_level':'user_min_price'})
# tra = pd.merge(tra,t,on='user_id',how='left')


tra[['user_click_total','user_click_buy_total','user_click_buy_rate','user_click_difshop_total','user_click_shop_rate','user_click_max','user_click_mean','user_click_min',
'user_click_amtotal','user_id']].to_csv('data/user_feature1.csv',index=None)


# 提取训练集二的用户相关特征
train2_f = pd.read_csv('data/train2_f.csv')
# 用户点击次数
tra = train2_f
t = tra[['user_id']]
t['user_click_total'] = 1
t = t.groupby('user_id')['user_click_total'].agg('sum').reset_index()
tra = pd.merge(tra,t,on='user_id',how='left')

# 用户点击且购买次数
t = tra[['user_id','is_trade']]
t = t.groupby('user_id')['is_trade'].agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_click_buy_total'})
tra = pd.merge(tra,t,on='user_id',how='left')

# 用户购买占点击比重
tra['user_click_buy_rate'] = tra['user_click_buy_total']/tra['user_click_total']

# 用户点击不同商家的数量
t = tra[['user_id','shop_id']]
t = t.drop_duplicates()
t['user_click_difshop_total']=1
t = t.groupby('user_id')['user_click_difshop_total'].agg('sum').reset_index()
tra = pd.merge(tra,t,on='user_id',how='left')

# 用户点不同商家数量占所有不同商家比重
t = tra[['shop_id','user_click_difshop_total','user_id']]
t['a']=1
num = t.groupby('shop_id').agg('sum').reset_index().shape[0]
t['user_click_shop_rate'] = t['user_click_difshop_total']/num
t.drop(['shop_id','user_click_difshop_total','a'],axis=1,inplace=True)
tra = pd.merge(tra,t,on='user_id',how='left')

# 用户点击最多/最少/平均商家次数
t = tra[['user_id','shop_id']]
t['click']=1
t = t.groupby(['user_id','shop_id']).agg('sum').reset_index()
t1 = t.groupby(['user_id'])['click'].agg('max').reset_index()
t2 = t.groupby(['user_id'])['click'].agg('mean').reset_index()
t3 = t.groupby(['user_id'])['click'].agg('min').reset_index()
t1.rename(columns={'click':'user_click_max'},inplace=True)
t2.rename(columns={'click':'user_click_mean'},inplace=True)
t3.rename(columns={'click':'user_click_min'},inplace=True)
tra = pd.merge(tra,t1,on='user_id',how='left')
tra = pd.merge(tra,t2,on='user_id',how='left')
tra = pd.merge(tra,t3,on='user_id',how='left')

# 用户上午点击次数
t = tra[['user_id','is_am']]
t = t.groupby('user_id')['is_am'].agg(sum).reset_index()
t = t.rename(columns={'is_am':'user_click_amtotal'})
tra = pd.merge(tra,t,on='user_id',how='left')

# # 用户购买最大价格最小等级商品
# t = tra[['user_id','item_price_level']]
# t = t.drop_duplicates()
# t = t.groupby('user_id').agg('max').reset_index()
# t = t.rename(columns={'item_price_level':'user_max_price'})
# tra = pd.merge(tra,t,on='user_id',how='left')
# t = tra[['user_id','item_price_level']]
# t = t.drop_duplicates()
# t = t.groupby('user_id').agg('min').reset_index()
# t = t.rename(columns={'item_price_level':'user_min_price'})
# tra = pd.merge(tra,t,on='user_id',how='left')

tra[['user_click_total','user_click_buy_total','user_click_buy_rate','user_click_difshop_total','user_click_shop_rate','user_click_max','user_click_mean','user_click_min',
'user_click_amtotal','user_id']].to_csv('data/user_feature2.csv',index=None)




# 提取训测试集三的用户相关特征
train3_f = pd.read_csv('data/train3_f.csv')
# 用户点击次数
tra = train3_f
t = tra[['user_id']]
t['user_click_total'] = 1
t = t.groupby('user_id')['user_click_total'].agg('sum').reset_index()
tra = pd.merge(tra,t,on='user_id',how='left')

# 用户点击且购买次数
t = tra[['user_id','is_trade']]
t = t.groupby('user_id')['is_trade'].agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_click_buy_total'})
tra = pd.merge(tra,t,on='user_id',how='left')

# 用户购买占点击比重
tra['user_click_buy_rate'] = tra['user_click_buy_total']/tra['user_click_total']

# 用户点击不同商家的数量
t = tra[['user_id','shop_id']]
t['user_click_difshop_total']=1
t = t.groupby('user_id')['user_click_difshop_total'].agg('sum').reset_index()
tra = pd.merge(tra,t,on='user_id',how='left')

# 用户点不同商家数量占所有不同商家比重
t = tra[['shop_id','user_click_difshop_total','user_id']]
t['a']=1
num = t.groupby('shop_id').agg('sum').reset_index().shape[0]
t['user_click_shop_rate'] = t['user_click_difshop_total']/num
t.drop(['shop_id','user_click_difshop_total','a'],axis=1,inplace=True)
tra = pd.merge(tra,t,on='user_id',how='left')

# 用户点击最多/最少/平均商家次数
t = tra[['user_id','shop_id']]
t['click']=1
t = t.groupby(['user_id','shop_id']).agg('sum').reset_index()
t1 = t.groupby(['user_id'])['click'].agg('max').reset_index()
t2 = t.groupby(['user_id'])['click'].agg('mean').reset_index()
t3 = t.groupby(['user_id'])['click'].agg('min').reset_index()
t1.rename(columns={'click':'user_click_max'},inplace=True)
t2.rename(columns={'click':'user_click_mean'},inplace=True)
t3.rename(columns={'click':'user_click_min'},inplace=True)
tra = pd.merge(tra,t1,on='user_id',how='left')
tra = pd.merge(tra,t2,on='user_id',how='left')
tra = pd.merge(tra,t3,on='user_id',how='left')

# 用户上午点击次数
t = tra[['user_id','is_am']]
t = t.groupby('user_id')['is_am'].agg(sum).reset_index()
t = t.rename(columns={'is_am':'user_click_amtotal'})
tra = pd.merge(tra,t,on='user_id',how='left')

# # 用户购买最大价格最小等级商品
# t = tra[['user_id','item_price_level']]
# t = t.drop_duplicates()
# t = t.groupby('user_id').agg('max').reset_index()
# t = t.rename(columns={'item_price_level':'user_max_price'})
# tra = pd.merge(tra,t,on='user_id',how='left')
# t = tra[['user_id','item_price_level']]
# t = t.drop_duplicates()
# t = t.groupby('user_id').agg('min').reset_index()
# t = t.rename(columns={'item_price_level':'user_min_price'})
# tra = pd.merge(tra,t,on='user_id',how='left')

tra[['user_click_total','user_click_buy_total','user_click_buy_rate','user_click_difshop_total','user_click_shop_rate','user_click_max','user_click_mean','user_click_min',
'user_click_amtotal','user_id']].to_csv('data/user_feature3.csv',index=None)
