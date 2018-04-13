import pandas as pd
import xgboost as xgb
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import time
from datetime import date
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 从训练集一中获取店铺相关的特征
train1_f = pd.read_csv('data/train1_f.csv')
store = train1_f
# 该店铺被点击次数
t = store[['shop_id']]
t['shop_click_total'] = 1
t = t.groupby('shop_id').agg('sum').reset_index()
store = pd.merge(store,t,on='shop_id',how='left')
# 该店铺被购买次数
t = store[['shop_id','is_trade']]
t = t.groupby('shop_id').agg('sum').reset_index()
t.rename(columns={'is_trade':'shop_click_buy_total'},inplace=True)
store = pd.merge(store,t,on='shop_id',how='left')
# 该店铺购买率
store['shop_click_buy_rate'] = store['shop_click_buy_total']/store['shop_click_total']
# 店铺最早消费时间，最晚消费时间
t = store[['shop_id','hour','is_trade']]
t = t[t.is_trade==1][['shop_id','hour']]
t1 = t.groupby('shop_id')['hour'].agg('min').reset_index()
t2 = t.groupby('shop_id')['hour'].agg('max').reset_index()
t1 = t1.rename(columns={'hour':'min_sale_hour'})
t2 = t2.rename(columns={'hour':'max_sale_hour'})
store = pd.merge(store,t1,on='shop_id',how='left')
store = pd.merge(store,t2,on='shop_id',how='left')
# 购买该店铺的最小年龄等级，最大，平均年龄等级
t = store[['shop_id','user_age_level','is_trade']]
t = t[t.is_trade==1][['shop_id','user_age_level']]
t[['user_age_level']]=t[['user_age_level']].astype(int)
t1 = t.groupby('shop_id').agg('min').reset_index()
t2 = t.groupby('shop_id').agg('mean').reset_index()
t3 = t.groupby('shop_id').agg('max').reset_index()
t1 = t1.rename(columns={'user_age_level':'min_age'})
t2 = t2.rename(columns={'user_age_level':'mean_age'})
t3 = t3.rename(columns={'user_age_level':'max_age'})
store = pd.merge(store,t1,on='shop_id',how='left')
store = pd.merge(store,t2,on='shop_id',how='left')
store = pd.merge(store,t3,on='shop_id',how='left')

#该店铺上午售出数量 下午售出数量
t = store[['shop_id','hour','is_trade']]
t1 = t[(t.is_trade==1)&(t.hour)<=12][['shop_id']]
t1['sale_am']=1
t1 = t1.groupby('shop_id').agg('sum').reset_index()

t2 = t[(t.is_trade==1)&(t.hour)>12][['shop_id']]
t2['sale_pm']=1
t2 = t2.groupby('shop_id').agg('sum').reset_index()
store = pd.merge(store,t1,on='shop_id',how='left')
store = pd.merge(store,t2,on='shop_id',how='left')

# 该店铺被多少不同用户购买
t = store[['user_id','shop_id','is_trade']]
t = t[t.is_trade==1]
t = t.drop_duplicates(subset=['user_id','shop_id'])
t = t.groupby(['shop_id'])['is_trade'].agg('sum').reset_index()
t = t.rename(columns={'is_trade':'shop_difuser_total'})
store = pd.merge(store,t,on=['shop_id'],how='left')
# 该店铺售卖种类
t = store[['shop_id','cate2','is_trade']]
t = t[t.is_trade==1]
t = t.drop_duplicates(subset=['shop_id','cate2'])
t = t.groupby(['shop_id'])['is_trade'].agg('sum').reset_index()
t = t.rename(columns={'is_trade':'shop_cate2_btotal'})
store = pd.merge(store,t,on=['shop_id'],how='left')

store[['shop_click_buy_rate','shop_click_buy_total','shop_click_total','min_sale_hour','max_sale_hour','min_age','mean_age','max_age',
       'sale_am','sale_pm','shop_cate2_btotal','shop_id']].to_csv('data/shop_feature1.csv',index=None)
# 'shop_difuser_total',


# 从训练集二中获取店铺相关的特征
train2_f = pd.read_csv('data/train2_f.csv')
store = train2_f
# 该店铺被点击次数
t = store[['shop_id']]
t['shop_click_total'] = 1
t = t.groupby('shop_id').agg('sum').reset_index()
store = pd.merge(store,t,on='shop_id',how='left')
# 该店铺被购买次数
t = store[['shop_id','is_trade']]
t = t.groupby('shop_id').agg('sum').reset_index()
t.rename(columns={'is_trade':'shop_click_buy_total'},inplace=True)
store = pd.merge(store,t,on='shop_id',how='left')
# 该店铺购买率
store['shop_click_buy_rate'] = store['shop_click_buy_total']/store['shop_click_total']
# 店铺最早消费时间，最晚消费时间
t = store[['shop_id','hour','is_trade']]
t = t[t.is_trade==1][['shop_id','hour']]
t1 = t.groupby('shop_id')['hour'].agg('min').reset_index()
t2 = t.groupby('shop_id')['hour'].agg('max').reset_index()
t1 = t1.rename(columns={'hour':'min_sale_hour'})
t2 = t2.rename(columns={'hour':'max_sale_hour'})
store = pd.merge(store,t1,on='shop_id',how='left')
store = pd.merge(store,t2,on='shop_id',how='left')
# 购买该店铺的最小年龄等级，最大，平均年龄等级
t = store[['shop_id','user_age_level','is_trade']]
t = t[t.is_trade==1][['shop_id','user_age_level']]
t[['user_age_level']]=t[['user_age_level']].astype(int)
t1 = t.groupby('shop_id').agg('min').reset_index()
t2 = t.groupby('shop_id').agg('mean').reset_index()
t3 = t.groupby('shop_id').agg('max').reset_index()
t1 = t1.rename(columns={'user_age_level':'min_age'})
t2 = t2.rename(columns={'user_age_level':'mean_age'})
t3 = t3.rename(columns={'user_age_level':'max_age'})
store = pd.merge(store,t1,on='shop_id',how='left')
store = pd.merge(store,t2,on='shop_id',how='left')
store = pd.merge(store,t3,on='shop_id',how='left')

#该店铺上午售出数量 下午售出数量
t = store[['shop_id','hour','is_trade']]
t1 = t[(t.is_trade==1)&(t.hour)<=12][['shop_id']]
t1['sale_am']=1
t1 = t1.groupby('shop_id').agg('sum').reset_index()

t2 = t[(t.is_trade==1)&(t.hour)>12][['shop_id']]
t2['sale_pm']=1
t2 = t2.groupby('shop_id').agg('sum').reset_index()
store = pd.merge(store,t1,on='shop_id',how='left')
store = pd.merge(store,t2,on='shop_id',how='left')

# 该店铺被多少不同用户购买
t = store[['user_id','shop_id','is_trade']]
t = t[t.is_trade==1]
t = t.drop_duplicates(subset=['user_id','shop_id'])
t = t.groupby(['shop_id'])['is_trade'].agg('sum').reset_index()
t = t.rename(columns={'is_trade':'shop_difuser_total'})
store = pd.merge(store,t,on=['shop_id'],how='left')
# 该店铺售卖种类
t = store[['shop_id','cate2','is_trade']]
t = t[t.is_trade==1]
t = t.drop_duplicates(subset=['shop_id','cate2'])
t = t.groupby(['shop_id'])['is_trade'].agg('sum').reset_index()
t = t.rename(columns={'is_trade':'shop_cate2_btotal'})
store = pd.merge(store,t,on=['shop_id'],how='left')

store[['shop_click_buy_rate','shop_click_buy_total','shop_click_total','min_sale_hour','max_sale_hour','min_age','mean_age','max_age',
       'sale_am','sale_pm','shop_cate2_btotal','shop_id']].to_csv('data/shop_feature2.csv',index=None)
# 'shop_difuser_total',
# 从训练集三中获取店铺相关的特征
train3_f = pd.read_csv('data/train3_f.csv')
store = train3_f
# 该店铺被点击次数
t = store[['shop_id']]
t['shop_click_total'] = 1
t = t.groupby('shop_id').agg('sum').reset_index()
store = pd.merge(store,t,on='shop_id',how='left')
# 该店铺被购买次数
t = store[['shop_id','is_trade']]
t = t.groupby('shop_id').agg('sum').reset_index()
t.rename(columns={'is_trade':'shop_click_buy_total'},inplace=True)
store = pd.merge(store,t,on='shop_id',how='left')
# 该店铺购买率
store['shop_click_buy_rate'] = store['shop_click_buy_total']/store['shop_click_total']
# 店铺最早消费时间，最晚消费时间
t = store[['shop_id','hour','is_trade']]
t = t[t.is_trade==1][['shop_id','hour']]
t1 = t.groupby('shop_id')['hour'].agg('min').reset_index()
t2 = t.groupby('shop_id')['hour'].agg('max').reset_index()
t1 = t1.rename(columns={'hour':'min_sale_hour'})
t2 = t2.rename(columns={'hour':'max_sale_hour'})
store = pd.merge(store,t1,on='shop_id',how='left')
store = pd.merge(store,t2,on='shop_id',how='left')
# 购买该店铺的最小年龄等级，最大，平均年龄等级
t = store[['shop_id','user_age_level','is_trade']]
t = t[t.is_trade==1][['shop_id','user_age_level']]
t[['user_age_level']]=t[['user_age_level']].astype(int)
t1 = t.groupby('shop_id').agg('min').reset_index()
t2 = t.groupby('shop_id').agg('mean').reset_index()
t3 = t.groupby('shop_id').agg('max').reset_index()
t1 = t1.rename(columns={'user_age_level':'min_age'})
t2 = t2.rename(columns={'user_age_level':'mean_age'})
t3 = t3.rename(columns={'user_age_level':'max_age'})
store = pd.merge(store,t1,on='shop_id',how='left')
store = pd.merge(store,t2,on='shop_id',how='left')
store = pd.merge(store,t3,on='shop_id',how='left')

#该店铺上午售出数量 下午售出数量
t = store[['shop_id','hour','is_trade']]
t1 = t[(t.is_trade==1)&(t.hour)<=12][['shop_id']]
t1['sale_am']=1
t1 = t1.groupby('shop_id').agg('sum').reset_index()

t2 = t[(t.is_trade==1)&(t.hour)>12][['shop_id']]
t2['sale_pm']=1
t2 = t2.groupby('shop_id').agg('sum').reset_index()
store = pd.merge(store,t1,on='shop_id',how='left')
store = pd.merge(store,t2,on='shop_id',how='left')

# 该店铺被多少不同用户购买
t = store[['user_id','shop_id','is_trade']]
t = t[t.is_trade==1]
t = t.drop_duplicates(subset=['user_id','shop_id'])
t = t.groupby(['shop_id'])['is_trade'].agg('sum').reset_index()
t = t.rename(columns={'is_trade':'shop_difuser_total'})
store = pd.merge(store,t,on=['shop_id'],how='left')
# 该店铺售卖种类
t = store[['shop_id','cate2','is_trade']]
t = t[t.is_trade==1]
t = t.drop_duplicates(subset=['shop_id','cate2'])
t = t.groupby(['shop_id'])['is_trade'].agg('sum').reset_index()
t = t.rename(columns={'is_trade':'shop_cate2_btotal'})
store = pd.merge(store,t,on=['shop_id'],how='left')

store[['shop_click_buy_rate','shop_click_buy_total','shop_click_total','min_sale_hour','max_sale_hour','mean_age','max_age',
       'sale_am','sale_pm','shop_cate2_btotal','shop_id']].to_csv('data/shop_feature3.csv',index=None)
# ,'shop_difuser_total'
# ,'min_age'