import pandas as pd
import xgboost as xgb
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import time
from datetime import date
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 从训练集一中获取用户店铺交互特征
train1_f = pd.read_csv('data/train1_f.csv')
shop_fea = pd.read_csv('data/shop_feature1.csv')
shop_fea.drop_duplicates(inplace=True)
user_fea = pd.read_csv('data/user_feature1.csv')
user_fea.drop_duplicates(inplace=True)
train1_f = pd.merge(train1_f,shop_fea,on='shop_id',how='left')
train1_f = pd.merge(train1_f,user_fea,on='user_id',how='left')
user_shop = train1_f
# 该用户点击该店铺次数
t = user_shop[['user_id','shop_id']]
t['user_shop_click_total'] = 1
t = t.groupby(['user_id','shop_id'])['user_shop_click_total'].agg('sum').reset_index()
user_shop = pd.merge(user_shop,t,on=['user_id','shop_id'],how='left')

# 用户购买该店铺次数
t = user_shop[['user_id','shop_id','is_trade']]
t = t.groupby(['user_id','shop_id']).agg('sum').reset_index()
t.rename(columns={'is_trade':'user_shop_click_buy_total'},inplace=True)
user_shop = pd.merge(user_shop,t,on=['user_id','shop_id'],how='left')
# 用户购买该店铺率
user_shop['user_shop_click_buy_rate'] = user_shop['user_shop_click_buy_total']/user_shop['user_shop_click_total']

# 用户购买该店铺占店铺售卖比重
user_shop['user_shop_click_buy_rate']=user_shop['user_shop_click_buy_total']/user_shop['shop_click_buy_total']

# 用户购买该店铺占用户购买比重
user_shop['shop_user_click_buy_rate']=user_shop['user_shop_click_buy_total']/user_shop['user_click_buy_total']

user_shop[['user_shop_click_buy_rate','user_shop_click_buy_total','user_shop_click_total','user_shop_click_buy_rate','shop_user_click_buy_rate','user_id','shop_id']].to_csv('data/user_shop_feature1.csv',index=None)



# 从训练集二中获取用户店铺交互特征
train1_f = pd.read_csv('data/train2_f.csv')
shop_fea = pd.read_csv('data/shop_feature2.csv')
shop_fea.drop_duplicates(inplace=True)
user_fea = pd.read_csv('data/user_feature2.csv')
user_fea.drop_duplicates(inplace=True)
train1_f = pd.merge(train1_f,shop_fea,on='shop_id',how='left')
train1_f = pd.merge(train1_f,user_fea,on='user_id',how='left')
user_shop = train1_f
# 该用户点击该店铺次数
t = user_shop[['user_id','shop_id']]
t['user_shop_click_total'] = 1
t = t.groupby(['user_id','shop_id'])['user_shop_click_total'].agg('sum').reset_index()
user_shop = pd.merge(user_shop,t,on=['user_id','shop_id'],how='left')

# 用户购买该店铺次数
t = user_shop[['user_id','shop_id','is_trade']]
t = t.groupby(['user_id','shop_id']).agg('sum').reset_index()
t.rename(columns={'is_trade':'user_shop_click_buy_total'},inplace=True)
user_shop = pd.merge(user_shop,t,on=['user_id','shop_id'],how='left')
# 用户购买该店铺率
user_shop['user_shop_click_buy_rate'] = user_shop['user_shop_click_buy_total']/user_shop['user_shop_click_total']

# 用户购买该店铺占店铺售卖比重
user_shop['user_shop_click_buy_rate']=user_shop['user_shop_click_buy_total']/user_shop['shop_click_buy_total']

# 用户购买该店铺占用户购买比重
user_shop['shop_user_click_buy_rate']=user_shop['user_shop_click_buy_total']/user_shop['user_click_buy_total']

user_shop[['user_shop_click_buy_rate','user_shop_click_buy_total','user_shop_click_total','user_shop_click_buy_rate','shop_user_click_buy_rate','user_id','shop_id']].to_csv('data/user_shop_feature2.csv',index=None)



# 从训练集三中获取用户店铺交互特征
train1_f = pd.read_csv('data/train3_f.csv')
shop_fea = pd.read_csv('data/shop_feature3.csv')
shop_fea.drop_duplicates(inplace=True)
user_fea = pd.read_csv('data/user_feature3.csv')
user_fea.drop_duplicates(inplace=True)
train1_f = pd.merge(train1_f,shop_fea,on='shop_id',how='left')
train1_f = pd.merge(train1_f,user_fea,on='user_id',how='left')
user_shop = train1_f
# 该用户点击该店铺次数
t = user_shop[['user_id','shop_id']]
t['user_shop_click_total'] = 1
t = t.groupby(['user_id','shop_id'])['user_shop_click_total'].agg('sum').reset_index()
user_shop = pd.merge(user_shop,t,on=['user_id','shop_id'],how='left')

# 用户购买该店铺次数
t = user_shop[['user_id','shop_id','is_trade']]
t = t.groupby(['user_id','shop_id']).agg('sum').reset_index()
t.rename(columns={'is_trade':'user_shop_click_buy_total'},inplace=True)
user_shop = pd.merge(user_shop,t,on=['user_id','shop_id'],how='left')
# 用户购买该店铺率
user_shop['user_shop_click_buy_rate'] = user_shop['user_shop_click_buy_total']/user_shop['user_shop_click_total']

# 用户购买该店铺占店铺售卖比重
user_shop['user_shop_click_buy_rate']=user_shop['user_shop_click_buy_total']/user_shop['shop_click_buy_total']

# 用户购买该店铺占用户购买比重
user_shop['shop_user_click_buy_rate']=user_shop['user_shop_click_buy_total']/user_shop['user_click_buy_total']

user_shop[['user_shop_click_buy_rate','user_shop_click_buy_total','user_shop_click_total','user_shop_click_buy_rate','shop_user_click_buy_rate','user_id','shop_id']].to_csv('data/user_shop_feature3.csv',index=None)




# 从训练集4中获取用户店铺交互特征
train1_f = pd.read_csv('data/train4_f.csv')
shop_fea = pd.read_csv('data/shop_feature4.csv')
shop_fea.drop_duplicates(inplace=True)
user_fea = pd.read_csv('data/user_feature4.csv')
user_fea.drop_duplicates(inplace=True)
train1_f = pd.merge(train1_f,shop_fea,on='shop_id',how='left')
train1_f = pd.merge(train1_f,user_fea,on='user_id',how='left')
user_shop = train1_f
# 该用户点击该店铺次数
t = user_shop[['user_id','shop_id']]
t['user_shop_click_total'] = 1
t = t.groupby(['user_id','shop_id'])['user_shop_click_total'].agg('sum').reset_index()
user_shop = pd.merge(user_shop,t,on=['user_id','shop_id'],how='left')

# 用户购买该店铺次数
t = user_shop[['user_id','shop_id','is_trade']]
t = t.groupby(['user_id','shop_id']).agg('sum').reset_index()
t.rename(columns={'is_trade':'user_shop_click_buy_total'},inplace=True)
user_shop = pd.merge(user_shop,t,on=['user_id','shop_id'],how='left')
# 用户购买该店铺率
user_shop['user_shop_click_buy_rate'] = user_shop['user_shop_click_buy_total']/user_shop['user_shop_click_total']

# 用户购买该店铺占店铺售卖比重
user_shop['user_shop_click_buy_rate']=user_shop['user_shop_click_buy_total']/user_shop['shop_click_buy_total']

# 用户购买该店铺占用户购买比重
user_shop['shop_user_click_buy_rate']=user_shop['user_shop_click_buy_total']/user_shop['user_click_buy_total']

user_shop[['user_shop_click_buy_rate','user_shop_click_buy_total','user_shop_click_total','user_shop_click_buy_rate','shop_user_click_buy_rate','user_id','shop_id']].to_csv('data/user_shop_feature4.csv',index=None)




# # 从训练集4中获取用户店铺交互特征
# train1_f = pd.read_csv('data/train5_f.csv')
# shop_fea = pd.read_csv('data/shop_feature5.csv')
# shop_fea.drop_duplicates(inplace=True)
# user_fea = pd.read_csv('data/user_feature5.csv')
# user_fea.drop_duplicates(inplace=True)
# train1_f = pd.merge(train1_f,shop_fea,on='shop_id',how='left')
# train1_f = pd.merge(train1_f,user_fea,on='user_id',how='left')
# user_shop = train1_f
# # 该用户点击该店铺次数
# t = user_shop[['user_id','shop_id']]
# t['user_shop_click_total'] = 1
# t = t.groupby(['user_id','shop_id'])['user_shop_click_total'].agg('sum').reset_index()
# user_shop = pd.merge(user_shop,t,on=['user_id','shop_id'],how='left')
#
# # 用户购买该店铺次数
# t = user_shop[['user_id','shop_id','is_trade']]
# t = t.groupby(['user_id','shop_id']).agg('sum').reset_index()
# t.rename(columns={'is_trade':'user_shop_click_buy_total'},inplace=True)
# user_shop = pd.merge(user_shop,t,on=['user_id','shop_id'],how='left')
# # 用户购买该店铺率
# user_shop['user_shop_click_buy_rate'] = user_shop['user_shop_click_buy_total']/user_shop['user_shop_click_total']
#
# # 用户购买该店铺占店铺售卖比重
# user_shop['user_shop_click_buy_rate']=user_shop['user_shop_click_buy_total']/user_shop['shop_click_buy_total']
#
# # 用户购买该店铺占用户购买比重
# user_shop['shop_user_click_buy_rate']=user_shop['user_shop_click_buy_total']/user_shop['user_click_buy_total']
#
# user_shop[['user_shop_click_buy_rate','user_shop_click_buy_total','user_shop_click_total','user_shop_click_buy_rate','shop_user_click_buy_rate','user_id','shop_id']].to_csv('data/user_shop_feature5.csv',index=None)