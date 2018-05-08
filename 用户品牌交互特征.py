import pandas as pd
import xgboost as xgb
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import time
from datetime import date
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# 从训练集1中提取用户品牌交互特征
train = pd.read_csv('data/train1_f.csv')
u = pd.read_csv('data/user_feature1.csv')
u = u[['user_id','user_click_total','user_click_buy_total']]
u = u.drop_duplicates()
# 用户点击该品牌次数
t = train[['user_id','item_brand_id']]
t['user_brand_click'] = 1
t = t.groupby(['user_id','item_brand_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_brand_id'],how='left')
# 用户购买该品牌次数
t = train[['user_id','item_brand_id','is_trade']]
t = t.groupby(['user_id','item_brand_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_brand_buy'})
train = pd.merge(train,t,on=['user_id','item_brand_id'],how='left')
# 用户购买该品牌次数占该用户总购买比率
train = pd.merge(train,u[['user_id','user_click_buy_total']],on='user_id',how='left')
train['user_brand_rate'] = train['user_brand_buy']/train['user_click_buy_total']
# 用户点击该品牌次数占该用户总点击比率
train = pd.merge(train,u[['user_id','user_click_total']],on='user_id',how='left')
train['user_brand_crate'] = train['user_brand_click']/train['user_click_total']
train[['user_brand_click','user_brand_buy','user_brand_rate','user_brand_crate','user_id','item_brand_id']].to_csv('data/user_brand_feature1.csv',index=None)

#该职业购买该品牌次数
t = train[['user_occupation_id','item_brand_id','is_trade']]
t = t.groupby(['user_occupation_id','item_brand_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'occupation_brand_buy'})
train = pd.merge(train,t,on=['user_occupation_id','item_brand_id'],how='left')
# 该职业点击该品牌次数
t = train[['user_occupation_id','item_brand_id']]
t['occupation_brand_click'] = 1
t = t.groupby(['user_occupation_id','item_brand_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_occupation_id','item_brand_id'],how='left')
# 该职业购买率
train['occupation_brand_rate'] = train['occupation_brand_buy']/train['occupation_brand_click']
train[['occupation_brand_buy','occupation_brand_click','occupation_brand_rate','user_occupation_id','item_brand_id']].to_csv('data/occupation_brand_feature1.csv',index=None)

# 该星级
t = train[['user_star_level','item_brand_id','is_trade']]
t = t.groupby(['user_star_level','item_brand_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'star_brand_buy'})
train = pd.merge(train,t,on=['user_star_level','item_brand_id'],how='left')
# 该职业点击该品牌次数
t = train[['user_star_level','item_brand_id']]
t['star_brand_click'] = 1
t = t.groupby(['user_star_level','item_brand_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_star_level','item_brand_id'],how='left')
# 该职业购买率
train['star_brand_rate'] = train['star_brand_buy']/train['star_brand_click']
train[['star_brand_buy','star_brand_click','star_brand_rate','user_star_level','item_brand_id']].to_csv('data/star_brand_feature1.csv',index=None)

# 职业星级年龄
t = train[['user_star_level','user_age_level','user_occupation_id','item_brand_id','is_trade']]
t = t.groupby(['user_star_level','user_age_level','user_occupation_id','item_brand_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'occupation_star_age_brand_buy'})
train = pd.merge(train,t,on=['user_star_level','user_age_level','user_occupation_id','item_brand_id'],how='left')
# 该职业点击该品牌次数
t = train[['user_star_level','user_age_level','user_occupation_id','item_brand_id']]
t['occupation_star_age_brand_click'] = 1
t = t.groupby(['user_star_level','user_age_level','user_occupation_id','item_brand_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_star_level','user_age_level','user_occupation_id','item_brand_id'],how='left')
# 该职业购买率
train['occupation_star_age_brand_rate'] = train['occupation_star_age_brand_buy']/train['occupation_star_age_brand_click']
train[['occupation_star_age_brand_buy','occupation_star_age_brand_click','occupation_star_age_brand_rate','user_star_level','user_age_level','user_occupation_id','item_brand_id']].to_csv('data/occupation_star_age_brand_feature1.csv',index=None)

# 该年龄
t = train[['user_age_level','item_brand_id','is_trade']]
t = t.groupby(['user_age_level','item_brand_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'age_brand_buy'})
train = pd.merge(train,t,on=['user_age_level','item_brand_id'],how='left')
# 该职业点击该品牌次数
t = train[['user_age_level','item_brand_id']]
t['age_brand_click'] = 1
t = t.groupby(['user_age_level','item_brand_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_age_level','item_brand_id'],how='left')
# 该职业购买率
train['age_brand_rate'] = train['age_brand_buy']/train['age_brand_click']
train[['age_brand_buy','age_brand_click','age_brand_rate','user_age_level','item_brand_id']].to_csv('data/age_brand_feature1.csv',index=None)




# 从训练集2中提取用户品牌交互特征
train = pd.read_csv('data/train2_f.csv')
u = pd.read_csv('data/user_feature2.csv')
u = u[['user_id','user_click_total','user_click_buy_total']]
u = u.drop_duplicates()
# 用户点击该品牌次数
t = train[['user_id','item_brand_id']]
t['user_brand_click'] = 1
t = t.groupby(['user_id','item_brand_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_brand_id'],how='left')
# 用户购买该品牌次数
t = train[['user_id','item_brand_id','is_trade']]
t = t.groupby(['user_id','item_brand_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_brand_buy'})
train = pd.merge(train,t,on=['user_id','item_brand_id'],how='left')
# 用户购买该品牌次数占该用户总购买比率
u = u.drop_duplicates()
train = pd.merge(train,u[['user_id','user_click_buy_total']],on='user_id',how='left')
train['user_brand_rate'] = train['user_brand_buy']/train['user_click_buy_total']
# 用户点击该品牌次数占该用户总点击比率
train = pd.merge(train,u[['user_id','user_click_total']],on='user_id',how='left')
train['user_brand_crate'] = train['user_brand_click']/train['user_click_total']
train[['user_brand_click','user_brand_buy','user_brand_rate','user_brand_crate','user_id','item_brand_id']].to_csv('data/user_brand_feature2.csv',index=None)

#该职业购买该品牌次数
t = train[['user_occupation_id','item_brand_id','is_trade']]
t = t.groupby(['user_occupation_id','item_brand_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'occupation_brand_buy'})
train = pd.merge(train,t,on=['user_occupation_id','item_brand_id'],how='left')
# 该职业点击该品牌次数
t = train[['user_occupation_id','item_brand_id']]
t['occupation_brand_click'] = 1
t = t.groupby(['user_occupation_id','item_brand_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_occupation_id','item_brand_id'],how='left')
# 该职业购买率
train['occupation_brand_rate'] = train['occupation_brand_buy']/train['occupation_brand_click']
train[['occupation_brand_buy','occupation_brand_click','occupation_brand_rate','user_occupation_id','item_brand_id']].to_csv('data/occupation_brand_feature2.csv',index=None)

# 该星级
t = train[['user_star_level','item_brand_id','is_trade']]
t = t.groupby(['user_star_level','item_brand_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'star_brand_buy'})
train = pd.merge(train,t,on=['user_star_level','item_brand_id'],how='left')
# 该职业点击该品牌次数
t = train[['user_star_level','item_brand_id']]
t['star_brand_click'] = 1
t = t.groupby(['user_star_level','item_brand_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_star_level','item_brand_id'],how='left')
# 该职业购买率
train['star_brand_rate'] = train['star_brand_buy']/train['star_brand_click']
train[['star_brand_buy','star_brand_click','star_brand_rate','user_star_level','item_brand_id']].to_csv('data/star_brand_feature2.csv',index=None)
# 职业星级年龄
t = train[['user_star_level','user_age_level','user_occupation_id','item_brand_id','is_trade']]
t = t.groupby(['user_star_level','user_age_level','user_occupation_id','item_brand_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'occupation_star_age_brand_buy'})
train = pd.merge(train,t,on=['user_star_level','user_age_level','user_occupation_id','item_brand_id'],how='left')
# 该职业点击该品牌次数
t = train[['user_star_level','user_age_level','user_occupation_id','item_brand_id']]
t['occupation_star_age_brand_click'] = 1
t = t.groupby(['user_star_level','user_age_level','user_occupation_id','item_brand_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_star_level','user_age_level','user_occupation_id','item_brand_id'],how='left')
# 该职业购买率
train['occupation_star_age_brand_rate'] = train['occupation_star_age_brand_buy']/train['occupation_star_age_brand_click']
train[['occupation_star_age_brand_buy','occupation_star_age_brand_click','occupation_star_age_brand_rate','user_star_level','user_age_level','user_occupation_id','item_brand_id']].to_csv('data/occupation_star_age_brand_feature2.csv',index=None)

# 该年龄
t = train[['user_age_level','item_brand_id','is_trade']]
t = t.groupby(['user_age_level','item_brand_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'age_brand_buy'})
train = pd.merge(train,t,on=['user_age_level','item_brand_id'],how='left')
# 该职业点击该品牌次数
t = train[['user_age_level','item_brand_id']]
t['age_brand_click'] = 1
t = t.groupby(['user_age_level','item_brand_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_age_level','item_brand_id'],how='left')
# 该职业购买率
train['age_brand_rate'] = train['age_brand_buy']/train['age_brand_click']
train[['age_brand_buy','age_brand_click','age_brand_rate','user_age_level','item_brand_id']].to_csv('data/age_brand_feature2.csv',index=None)


# 从训练集1中提取用户品牌交互特征
train = pd.read_csv('data/train3_f.csv')
u = pd.read_csv('data/user_feature3.csv')
u = u[['user_id','user_click_total','user_click_buy_total']]
u = u.drop_duplicates()
# 用户点击该品牌次数
t = train[['user_id','item_brand_id']]
t['user_brand_click'] = 1
t = t.groupby(['user_id','item_brand_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_brand_id'],how='left')
# 用户购买该品牌次数
t = train[['user_id','item_brand_id','is_trade']]
t = t.groupby(['user_id','item_brand_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_brand_buy'})
train = pd.merge(train,t,on=['user_id','item_brand_id'],how='left')
# 用户购买该品牌次数占该用户总购买比率
u = u.drop_duplicates()
train = pd.merge(train,u[['user_id','user_click_buy_total']],on='user_id',how='left')
train['user_brand_rate'] = train['user_brand_buy']/train['user_click_buy_total']
# 用户点击该品牌次数占该用户总点击比率
train = pd.merge(train,u[['user_id','user_click_total']],on='user_id',how='left')
train['user_brand_crate'] = train['user_brand_click']/train['user_click_total']
train = train.drop_duplicates()
train[['user_brand_click','user_brand_buy','user_brand_rate','user_brand_crate','user_id','item_brand_id']].to_csv('data/user_brand_feature3.csv',index=None)



# 从训练集1中提取用户品牌交互特征
train = pd.read_csv('data/train4_f.csv')
u = pd.read_csv('data/user_feature4.csv')
u = u[['user_id','user_click_total','user_click_buy_total']]
u = u.drop_duplicates()
# 用户点击该品牌次数
t = train[['user_id','item_brand_id']]
t['user_brand_click'] = 1
t = t.groupby(['user_id','item_brand_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_brand_id'],how='left')
# 用户购买该品牌次数
t = train[['user_id','item_brand_id','is_trade']]
t = t.groupby(['user_id','item_brand_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'user_brand_buy'})
train = pd.merge(train,t,on=['user_id','item_brand_id'],how='left')
# 用户购买该品牌次数占该用户总购买比率
u = u.drop_duplicates()
train = pd.merge(train,u[['user_id','user_click_buy_total']],on='user_id',how='left')
train['user_brand_rate'] = train['user_brand_buy']/train['user_click_buy_total']
# 用户点击该品牌次数占该用户总点击比率
train = pd.merge(train,u[['user_id','user_click_total']],on='user_id',how='left')
train['user_brand_crate'] = train['user_brand_click']/train['user_click_total']
train = train.drop_duplicates()
train[['user_brand_click','user_brand_buy','user_brand_rate','user_brand_crate','user_id','item_brand_id']].to_csv('data/user_brand_feature4.csv',index=None)





# # 从训练集1中提取用户品牌交互特征
# train = pd.read_csv('data/train5_f.csv')
# u = pd.read_csv('data/user_feature5.csv')
# u = u[['user_id','user_click_total','user_click_buy_total']]
# u = u.drop_duplicates()
# # 用户点击该品牌次数
# t = train[['user_id','item_brand_id']]
# t['user_brand_click'] = 1
# t = t.groupby(['user_id','item_brand_id']).agg('sum').reset_index()
# train = pd.merge(train,t,on=['user_id','item_brand_id'],how='left')
# # 用户购买该品牌次数
# t = train[['user_id','item_brand_id','is_trade']]
# t = t.groupby(['user_id','item_brand_id']).agg('sum').reset_index()
# t = t.rename(columns={'is_trade':'user_brand_buy'})
# train = pd.merge(train,t,on=['user_id','item_brand_id'],how='left')
# # 用户购买该品牌次数占该用户总购买比率
# u = u.drop_duplicates()
# train = pd.merge(train,u[['user_id','user_click_buy_total']],on='user_id',how='left')
# train['user_brand_rate'] = train['user_brand_buy']/train['user_click_buy_total']
# # 用户点击该品牌次数占该用户总点击比率
# train = pd.merge(train,u[['user_id','user_click_total']],on='user_id',how='left')
# train['user_brand_crate'] = train['user_brand_click']/train['user_click_total']
# train = train.drop_duplicates()
# train[['user_brand_click','user_brand_buy','user_brand_rate','user_brand_crate','user_id','item_brand_id']].to_csv('data/user_brand_feature5.csv',index=None)