import pandas as pd
import xgboost as xgb
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import time
from datetime import date
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import gc

train1_f = pd.read_csv('data/train1_f.csv')
# train1_p = pd.read_csv('data/train1_f.csv')

# 提取训练集一的用户相关特征
# 用户点击次数
tra = train1_f
# tra_p = train1_p
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
del t
gc.collect()

# 用户点击不同商家的数量
t = tra[['user_id','shop_id']]
t = t.drop_duplicates()
t['user_click_difshop_total']=1
t = t.groupby('user_id')['user_click_difshop_total'].agg('sum').reset_index()
t = t.drop_duplicates(subset=['user_id'])
tra = pd.merge(tra,t,on='user_id',how='left')
del t
gc.collect()
# 用户点不同商家数量占所有不同商家比重
t = tra[['shop_id','user_click_difshop_total','user_id']]
t['a']=1
num = t.groupby('shop_id').agg('sum').reset_index().shape[0]
t['user_click_shop_rate'] = t['user_click_difshop_total']/num
t.drop(['shop_id','user_click_difshop_total','a'],axis=1,inplace=True)
t = t.drop_duplicates(subset=['user_id'])
tra = pd.merge(tra,t,on='user_id',how='left')
del t
gc.collect()
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
del t
gc.collect()
tra[['user_click_total','user_click_buy_total','user_click_buy_rate','user_click_difshop_total','user_click_shop_rate','user_click_max','user_click_mean','user_click_min',
'user_id']].to_csv('data/user_feature1.csv',index=None)
# 用户在该时间点击数量
t = tra[['user_id','timeduan']]
t['user_time_click'] = 1
t = t.groupby(['user_id','timeduan']).agg('sum').reset_index()
tra = pd.merge(tra,t,on=['user_id','timeduan'],how='left')
# goumai
t = tra[['user_id','timeduan','is_trade']]
t = t.groupby(['user_id','timeduan']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_time_buy'})
tra = pd.merge(tra,t,on=['user_id','timeduan'],how='left')
# rate
tra['user_time_rate'] = tra['user_time_buy']/tra['user_time_click']
tra[['user_time_click','user_time_buy','user_time_rate','user_id','timeduan']].to_csv('data/user_time_feature1.csv',index=None)
del t
gc.collect()
# 改nianling购买平均时间
t = tra[tra.is_trade==1][['user_age_level','hour','is_trade']]
t = t.groupby(['user_age_level','hour']).agg('sum').reset_index()
t['time_sum'] = t['hour']*t['is_trade']
t = t.groupby('user_age_level').agg('sum').reset_index()
t['mean_time'] = t['time_sum']/t['is_trade']
tra = pd.merge(tra,t,on=['user_age_level'],how='left')
tra[['mean_time','user_age_level']].to_csv('data/age_time_feature1.csv',index=None)
import gc
del tra,train1_f
gc.collect()




# 提取训练集二的用户相关特征
train2_f = pd.read_csv('data/train2_f.csv')
# 用户点击次数
tra = train2_f
t = tra[['user_id']]
t['user_click_total'] = 1
t = t.groupby('user_id')['user_click_total'].agg('sum').reset_index()
tra = pd.merge(tra,t,on='user_id',how='left')
del t
gc.collect()
# 用户点击且购买次数
t = tra[['user_id','is_trade']]
t = t.groupby('user_id')['is_trade'].agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_click_buy_total'})
tra = pd.merge(tra,t,on='user_id',how='left')
del t
gc.collect()
# 用户购买占点击比重
tra['user_click_buy_rate'] = tra['user_click_buy_total']/tra['user_click_total']

# 用户点击不同商家的数量
t = tra[['user_id','shop_id']]
t = t.drop_duplicates()
t['user_click_difshop_total']=1
t = t.groupby('user_id')['user_click_difshop_total'].agg('sum').reset_index()
tra = pd.merge(tra,t,on='user_id',how='left')
del t
gc.collect()
# 用户点不同商家数量占所有不同商家比重
t = tra[['shop_id','user_click_difshop_total','user_id']]
t['a']=1
num = t.groupby('shop_id').agg('sum').reset_index().shape[0]
t['user_click_shop_rate'] = t['user_click_difshop_total']/num
t.drop(['shop_id','user_click_difshop_total','a'],axis=1,inplace=True)
t = t.drop_duplicates(subset=['user_id'])
tra = pd.merge(tra,t,on='user_id',how='left')
del t
gc.collect()
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
del t
gc.collect()

# 用户购买的种类数目
t = tra[['user_id','cate2','is_trade']]
t = t[t.is_trade==1]
t = t.drop_duplicates(subset=['user_id','cate2'])
t = t.groupby(['user_id'])['is_trade'].agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_cate2_total'})
tra = pd.merge(tra,t,on=['user_id'],how='left')
# 用户购买的品牌数目
t = tra[['user_id','item_brand_id','is_trade']]
t = t[t.is_trade==1]
t = t.drop_duplicates(subset=['user_id','item_brand_id'])
t = t.groupby(['user_id'])['is_trade'].agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_brand_total'})
tra = pd.merge(tra,t,on=['user_id'],how='left')
del t
gc.collect()
tra[['user_click_total','user_click_buy_total','user_click_buy_rate','user_click_difshop_total','user_click_shop_rate','user_click_max','user_click_mean','user_click_min',
'user_id']].to_csv('data/user_feature2.csv',index=None)
# 'user_cate2_total','user_brand_total','user_click_amtotal','user_buy_amtotal''user_buy_pmtotal'
# 用户在该时间点击数量
t = tra[['user_id','timeduan']]
t['user_time_click'] = 1
t = t.groupby(['user_id','timeduan']).agg('sum').reset_index()
tra = pd.merge(tra,t,on=['user_id','timeduan'],how='left')
# goumai
t = tra[['user_id','timeduan','is_trade']]
t = t.groupby(['user_id','timeduan']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_time_buy'})
tra = pd.merge(tra,t,on=['user_id','timeduan'],how='left')
# rate
tra['user_time_rate'] = tra['user_time_buy']/tra['user_time_click']
tra[['user_time_click','user_time_buy','user_time_rate','user_id','timeduan']].to_csv('data/user_time_feature2.csv',index=None)
del t
gc.collect()
# 改nianling购买平均时间
t = tra[tra.is_trade==1][['user_age_level','hour','is_trade']]
t = t.groupby(['user_age_level','hour']).agg('sum').reset_index()
t['time_sum'] = t['hour']*t['is_trade']
t = t.groupby('user_age_level').agg('sum').reset_index()
t['mean_time'] = t['time_sum']/t['is_trade']
tra = pd.merge(tra,t,on=['user_age_level'],how='left')
tra[['mean_time','user_age_level']].to_csv('data/age_time_feature2.csv',index=None)
del tra,train2_f
gc.collect()




# 提取训测试集三的用户相关特征
train3_f = pd.read_csv('data/train3_f.csv')
# 用户点击次数
tra = train3_f
t = tra[['user_id']]
t['user_click_total'] = 1
t = t.groupby('user_id')['user_click_total'].agg('sum').reset_index()
tra = pd.merge(tra,t,on='user_id',how='left')
del t
gc.collect()
# 用户点击且购买次数
t = tra[['user_id','is_trade']]
t = t.groupby('user_id')['is_trade'].agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_click_buy_total'})
tra = pd.merge(tra,t,on='user_id',how='left')
del t
gc.collect()
# 用户购买占点击比重
tra['user_click_buy_rate'] = tra['user_click_buy_total']/tra['user_click_total']

# 用户点击不同商家的数量
t = tra[['user_id','shop_id']]
t['user_click_difshop_total']=1
t = t.groupby('user_id')['user_click_difshop_total'].agg('sum').reset_index()
tra = pd.merge(tra,t,on='user_id',how='left')
del t
gc.collect()
# 用户点不同商家数量占所有不同商家比重
t = tra[['shop_id','user_click_difshop_total','user_id']]
t['a']=1
num = t.groupby('shop_id').agg('sum').reset_index().shape[0]
t['user_click_shop_rate'] = t['user_click_difshop_total']/num
t.drop(['shop_id','user_click_difshop_total','a'],axis=1,inplace=True)
t = t.drop_duplicates(subset=['user_id'])
tra = pd.merge(tra,t,on='user_id',how='left')
del t
gc.collect()
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
del t
gc.collect()
# 用户购买的种类数目
t = tra[['user_id','cate2','is_trade']]
t = t[t.is_trade==1]
t = t.drop_duplicates(subset=['user_id','cate2'])
t = t.groupby(['user_id'])['is_trade'].agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_cate2_total'})
tra = pd.merge(tra,t,on=['user_id'],how='left')
del t
gc.collect()
# 用户购买的品牌数目
t = tra[['user_id','item_brand_id','is_trade']]
t = t[t.is_trade==1]
t = t.drop_duplicates(subset=['user_id','item_brand_id'])
t = t.groupby(['user_id'])['is_trade'].agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_brand_total'})
tra = pd.merge(tra,t,on=['user_id'],how='left')
del t
gc.collect()
tra[['user_click_total','user_click_buy_total','user_click_buy_rate','user_click_difshop_total','user_click_shop_rate','user_click_max','user_click_mean','user_click_min',
'user_id']].to_csv('data/user_feature3.csv',index=None)
# ,'user_cate2_total','user_brand_total','user_click_amtotal',,'user_buy_amtotal','user_buy_pmtotal',
# 用户在该时间点击数量
t = tra[['user_id','timeduan']]
t['user_time_click'] = 1
t = t.groupby(['user_id','timeduan']).agg('sum').reset_index()
tra = pd.merge(tra,t,on=['user_id','timeduan'],how='left')
del t
gc.collect()
# goumai
t = tra[['user_id','timeduan','is_trade']]
t = t.groupby(['user_id','timeduan']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_time_buy'})
tra = pd.merge(tra,t,on=['user_id','timeduan'],how='left')
del t
gc.collect()
# rate
tra['user_time_rate'] = tra['user_time_buy']/tra['user_time_click']
tra[['user_time_click','user_time_buy','user_time_rate','user_id','timeduan']].to_csv('data/user_time_feature3.csv',index=None)

# 改nianling购买平均时间
t = tra[tra.is_trade==1][['user_age_level','hour','is_trade']]
t = t.groupby(['user_age_level','hour']).agg('sum').reset_index()
t['time_sum'] = t['hour']*t['is_trade']
t = t.groupby('user_age_level').agg('sum').reset_index()
t['mean_time'] = t['time_sum']/t['is_trade']
tra = pd.merge(tra,t,on=['user_age_level'],how='left')
tra[['mean_time','user_age_level']].to_csv('data/age_time_feature3.csv',index=None)
del tra
gc.collect()







# 提取训测试集三的用户相关特征
train3_f = pd.read_csv('data/train4_f.csv')
# 用户点击次数
tra = train3_f
t = tra[['user_id']]
t['user_click_total'] = 1
t = t.groupby('user_id')['user_click_total'].agg('sum').reset_index()
tra = pd.merge(tra,t,on='user_id',how='left')
del t
gc.collect()
# 用户点击且购买次数
t = tra[['user_id','is_trade']]
t = t.groupby('user_id')['is_trade'].agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_click_buy_total'})
tra = pd.merge(tra,t,on='user_id',how='left')
del t
gc.collect()
# 用户购买占点击比重
tra['user_click_buy_rate'] = tra['user_click_buy_total']/tra['user_click_total']

# 用户点击不同商家的数量
t = tra[['user_id','shop_id']]
t['user_click_difshop_total']=1
t = t.groupby('user_id')['user_click_difshop_total'].agg('sum').reset_index()
tra = pd.merge(tra,t,on='user_id',how='left')
del t
gc.collect()
# 用户点不同商家数量占所有不同商家比重
t = tra[['shop_id','user_click_difshop_total','user_id']]
t['a']=1
num = t.groupby('shop_id').agg('sum').reset_index().shape[0]
t['user_click_shop_rate'] = t['user_click_difshop_total']/num
t.drop(['shop_id','user_click_difshop_total','a'],axis=1,inplace=True)
t = t.drop_duplicates(subset=['user_id'])
tra = pd.merge(tra,t,on='user_id',how='left')
del t
gc.collect()
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
del t
gc.collect()
# 用户购买的种类数目
t = tra[['user_id','cate2','is_trade']]
t = t[t.is_trade==1]
t = t.drop_duplicates(subset=['user_id','cate2'])
t = t.groupby(['user_id'])['is_trade'].agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_cate2_total'})
tra = pd.merge(tra,t,on=['user_id'],how='left')
del t
gc.collect()
# 用户购买的品牌数目
t = tra[['user_id','item_brand_id','is_trade']]
t = t[t.is_trade==1]
t = t.drop_duplicates(subset=['user_id','item_brand_id'])
t = t.groupby(['user_id'])['is_trade'].agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_brand_total'})
tra = pd.merge(tra,t,on=['user_id'],how='left')
del t
gc.collect()
tra[['user_click_total','user_click_buy_total','user_click_buy_rate','user_click_difshop_total','user_click_shop_rate','user_click_max','user_click_mean','user_click_min',
'user_id']].to_csv('data/user_feature4.csv',index=None)
# ,'user_cate2_total','user_brand_total','user_click_amtotal',,'user_buy_amtotal','user_buy_pmtotal',
# 用户在该时间点击数量
t = tra[['user_id','timeduan']]
t['user_time_click'] = 1
t = t.groupby(['user_id','timeduan']).agg('sum').reset_index()
tra = pd.merge(tra,t,on=['user_id','timeduan'],how='left')
# goumai
t = tra[['user_id','timeduan','is_trade']]
t = t.groupby(['user_id','timeduan']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_time_buy'})
tra = pd.merge(tra,t,on=['user_id','timeduan'],how='left')
# rate
tra['user_time_rate'] = tra['user_time_buy']/tra['user_time_click']
tra[['user_time_click','user_time_buy','user_time_rate','user_id','timeduan']].to_csv('data/user_time_feature4.csv',index=None)
del t
gc.collect()
# 改nianling购买平均时间
t = tra[tra.is_trade==1][['user_age_level','hour','is_trade']]
t = t.groupby(['user_age_level','hour']).agg('sum').reset_index()
t['time_sum'] = t['hour']*t['is_trade']
t = t.groupby('user_age_level').agg('sum').reset_index()
t['mean_time'] = t['time_sum']/t['is_trade']
tra = pd.merge(tra,t,on=['user_age_level'],how='left')
tra[['mean_time','user_age_level']].to_csv('data/age_time_feature4.csv',index=None)




# # 提取训测试集三的用户相关特征
# train3_f = pd.read_csv('data/train5_f.csv')
# # 用户点击次数
# tra = train3_f
# t = tra[['user_id']]
# t['user_click_total'] = 1
# t = t.groupby('user_id')['user_click_total'].agg('sum').reset_index()
# tra = pd.merge(tra,t,on='user_id',how='left')
#
# # 用户点击且购买次数
# t = tra[['user_id','is_trade']]
# t = t.groupby('user_id')['is_trade'].agg('sum').reset_index()
# t = t.rename(columns={'is_trade':'user_click_buy_total'})
# tra = pd.merge(tra,t,on='user_id',how='left')
#
# # 用户购买占点击比重
# tra['user_click_buy_rate'] = tra['user_click_buy_total']/tra['user_click_total']
#
# # 用户点击不同商家的数量
# t = tra[['user_id','shop_id']]
# t['user_click_difshop_total']=1
# t = t.groupby('user_id')['user_click_difshop_total'].agg('sum').reset_index()
# tra = pd.merge(tra,t,on='user_id',how='left')
#
# # 用户点不同商家数量占所有不同商家比重
# t = tra[['shop_id','user_click_difshop_total','user_id']]
# t['a']=1
# num = t.groupby('shop_id').agg('sum').reset_index().shape[0]
# t['user_click_shop_rate'] = t['user_click_difshop_total']/num
# t.drop(['shop_id','user_click_difshop_total','a'],axis=1,inplace=True)
# tra = pd.merge(tra,t,on='user_id',how='left')
#
# # 用户点击最多/最少/平均商家次数
# t = tra[['user_id','shop_id']]
# t['click']=1
# t = t.groupby(['user_id','shop_id']).agg('sum').reset_index()
# t1 = t.groupby(['user_id'])['click'].agg('max').reset_index()
# t2 = t.groupby(['user_id'])['click'].agg('mean').reset_index()
# t3 = t.groupby(['user_id'])['click'].agg('min').reset_index()
# t1.rename(columns={'click':'user_click_max'},inplace=True)
# t2.rename(columns={'click':'user_click_mean'},inplace=True)
# t3.rename(columns={'click':'user_click_min'},inplace=True)
# tra = pd.merge(tra,t1,on='user_id',how='left')
# tra = pd.merge(tra,t2,on='user_id',how='left')
# tra = pd.merge(tra,t3,on='user_id',how='left')
#
# # 用户购买的种类数目
# t = tra[['user_id','cate2','is_trade']]
# t = t[t.is_trade==1]
# t = t.drop_duplicates(subset=['user_id','cate2'])
# t = t.groupby(['user_id'])['is_trade'].agg('sum').reset_index()
# t = t.rename(columns={'is_trade':'user_cate2_total'})
# tra = pd.merge(tra,t,on=['user_id'],how='left')
# # 用户购买的品牌数目
# t = tra[['user_id','item_brand_id','is_trade']]
# t = t[t.is_trade==1]
# t = t.drop_duplicates(subset=['user_id','item_brand_id'])
# t = t.groupby(['user_id'])['is_trade'].agg('sum').reset_index()
# t = t.rename(columns={'is_trade':'user_brand_total'})
# tra = pd.merge(tra,t,on=['user_id'],how='left')
#
# tra[['user_click_total','user_click_buy_total','user_click_buy_rate','user_click_difshop_total','user_click_shop_rate','user_click_max','user_click_mean','user_click_min',
# 'user_id']].to_csv('data/user_feature5.csv',index=None)
# # ,'user_cate2_total','user_brand_total','user_click_amtotal',,'user_buy_amtotal','user_buy_pmtotal',
# # 用户在该时间点击数量
# t = tra[['user_id','timeduan']]
# t['user_time_click'] = 1
# t = t.groupby(['user_id','timeduan']).agg('sum').reset_index()
# tra = pd.merge(tra,t,on=['user_id','timeduan'],how='left')
# # goumai
# t = tra[['user_id','timeduan','is_trade']]
# t = t.groupby(['user_id','timeduan']).agg('sum').reset_index()
# t = t.rename(columns={'is_trade':'user_time_buy'})
# tra = pd.merge(tra,t,on=['user_id','timeduan'],how='left')
# # rate
# tra['user_time_rate'] = tra['user_time_buy']/tra['user_time_click']
# tra[['user_time_click','user_time_buy','user_time_rate','user_id','timeduan']].to_csv('data/user_time_feature5.csv',index=None)
#
# # 改nianling购买平均时间
# t = tra[tra.is_trade==1][['user_age_level','hour','is_trade']]
# t = t.groupby(['user_age_level','hour']).agg('sum').reset_index()
# t['time_sum'] = t['hour']*t['is_trade']
# t = t.groupby('user_age_level').agg('sum').reset_index()
# t['mean_time'] = t['time_sum']/t['is_trade']
# tra = pd.merge(tra,t,on=['user_age_level'],how='left')
# tra[['mean_time','user_age_level']].to_csv('data/age_time_feature5.csv',index=None)