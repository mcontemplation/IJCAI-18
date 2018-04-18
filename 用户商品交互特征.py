import pandas as pd
import xgboost as xgb
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import time
from datetime import date
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# 从训练集一中获取用户商品相关特征
train1_f = pd.read_csv('data/train1_f.csv')
user_commodity = train1_f
# 用户点击该商品次数
t = user_commodity[['user_id','item_id']]
t['user_item_click_total'] = 1
t = t.groupby(['user_id','item_id']).agg(sum).reset_index()
user_commodity = pd.merge(user_commodity,t,on=['user_id','item_id'],how='left')

# 用户购买该商品次数
t = user_commodity[['user_id','item_id','is_trade']]
t = t.groupby(['user_id','item_id']).agg(sum).reset_index()
t.rename(columns={'is_trade':'user_item_click_buy_total'},inplace=True)
user_commodity = pd.merge(user_commodity,t,on=['user_id','item_id'],how='left')

# 用户购买该商品率
user_commodity['user_item_click_buy_rate'] = user_commodity['user_item_click_buy_total']/user_commodity['user_item_click_total']
# 该用户在多少家不同的店铺点击过该商品
t = user_commodity[['user_id','item_id','shop_id']]
t = t.drop_duplicates()
t['user_item_diff'] = 1
t = t.groupby(['user_id','item_id'])['user_item_diff'].agg('sum').reset_index()
user_commodity = pd.merge(user_commodity,t,on=['item_id','user_id'],how='left')
user_commodity[[
                'user_item_click_buy_rate','user_item_click_buy_total','user_item_click_total','user_item_diff',
                'user_id','item_id']].to_csv('data/user_item_feature1.csv',index=None)


# 该等级点击该商品次数
t = user_commodity[['user_star_level','item_id']]
t['star_item_click_total'] = 1
t = t.groupby(['user_star_level','item_id']).agg(sum).reset_index()
user_commodity = pd.merge(user_commodity,t,on=['user_star_level','item_id'],how='left')

# 该等级购买该商品次数
t = user_commodity[['user_star_level','item_id','is_trade']]
t = t.groupby(['user_star_level','item_id']).agg(sum).reset_index()
t.rename(columns={'is_trade':'star_item_click_buy_total'},inplace=True)
user_commodity = pd.merge(user_commodity,t,on=['user_star_level','item_id'],how='left')

# 该等级购买该商品率
user_commodity['star_item_click_buy_rate'] = user_commodity['star_item_click_buy_total']/user_commodity['star_item_click_total']
user_commodity[['star_item_click_buy_rate','star_item_click_buy_total','star_item_click_total','user_star_level','item_id']].to_csv('data/star_item_feature1.csv',index=None)



# 该年龄星级点击该商品次数
t = user_commodity[['item_id','user_age_level','user_star_level']]
t['age_star_click_total'] = 1
t = t.groupby(['user_age_level','user_star_level','item_id']).agg('sum').reset_index()
user_commodity = pd.merge(user_commodity,t,on=['user_age_level','user_star_level','item_id'],how='left')
# 该年龄星级购买该商品数
t = user_commodity[['item_id','user_age_level','user_star_level','is_trade']]
t = t.groupby(['user_age_level','user_star_level','item_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'age_star_click_buy_total'})
user_commodity = pd.merge(user_commodity,t,on=['user_age_level','user_star_level','item_id'],how='left')
# 该年龄星级购买该商品率
user_commodity['age_star_click_buy_rate'] = user_commodity['age_star_click_buy_total']/user_commodity['age_star_click_total']
user_commodity[['age_star_click_total','age_star_click_buy_total','age_star_click_buy_rate','item_id','user_age_level','user_star_level']].to_csv('data/age_star_item_feature1.csv',index=None)




#从训练集二中获取用户商品相关特征
train2_f = pd.read_csv('data/train2_f.csv')
user_commodity = train2_f
# 用户点击该商品次数
t = user_commodity[['user_id','item_id']]
t['user_item_click_total'] = 1
t = t.groupby(['user_id','item_id']).agg(sum).reset_index()
user_commodity = pd.merge(user_commodity,t,on=['user_id','item_id'],how='left')

# 用户购买该商品次数
t = user_commodity[['user_id','item_id','is_trade']]
t = t.groupby(['user_id','item_id']).agg(sum).reset_index()
t.rename(columns={'is_trade':'user_item_click_buy_total'},inplace=True)
user_commodity = pd.merge(user_commodity,t,on=['user_id','item_id'],how='left')

# 用户购买该商品率
user_commodity['user_item_click_buy_rate'] = user_commodity['user_item_click_buy_total']/user_commodity['user_item_click_total']

# 该用户在多少家不同的店铺点击过该商品
t = user_commodity[['user_id','item_id','shop_id']]
t = t.drop_duplicates()
t['user_item_diff'] = 1
t = t.groupby(['user_id','item_id'])['user_item_diff'].agg('sum').reset_index()
user_commodity = pd.merge(user_commodity,t,on=['item_id','user_id'],how='left')
user_commodity[[
                'user_item_click_buy_rate','user_item_click_buy_total','user_item_click_total','user_item_diff',
                'user_id','item_id']].to_csv('data/user_item_feature2.csv',index=None)

# 该等级点击该商品次数
t = user_commodity[['user_star_level','item_id']]
t['star_item_click_total'] = 1
t = t.groupby(['user_star_level','item_id']).agg(sum).reset_index()
user_commodity = pd.merge(user_commodity,t,on=['user_star_level','item_id'],how='left')

# 该等级购买该商品次数
t = user_commodity[['user_star_level','item_id','is_trade']]
t = t.groupby(['user_star_level','item_id']).agg(sum).reset_index()
t.rename(columns={'is_trade':'star_item_click_buy_total'},inplace=True)
user_commodity = pd.merge(user_commodity,t,on=['user_star_level','item_id'],how='left')

# 该等级购买该商品率
user_commodity['star_item_click_buy_rate'] = user_commodity['star_item_click_buy_total']/user_commodity['star_item_click_total']
user_commodity[['star_item_click_buy_rate','star_item_click_buy_total','star_item_click_total','user_star_level','item_id']].to_csv('data/star_item_feature2.csv',index=None)


# 该年龄星级点击该商品次数
t = user_commodity[['item_id','user_age_level','user_star_level']]
t['age_star_click_total'] = 1
t = t.groupby(['user_age_level','user_star_level','item_id']).agg('sum').reset_index()
user_commodity = pd.merge(user_commodity,t,on=['user_age_level','user_star_level','item_id'],how='left')
# 该年龄星级购买该商品数
t = user_commodity[['item_id','user_age_level','user_star_level','is_trade']]
t = t.groupby(['user_age_level','user_star_level','item_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'age_star_click_buy_total'})
user_commodity = pd.merge(user_commodity,t,on=['user_age_level','user_star_level','item_id'],how='left')
# 该年龄星级购买该商品率
user_commodity['age_star_click_buy_rate'] = user_commodity['age_star_click_buy_total']/user_commodity['age_star_click_total']
user_commodity[['age_star_click_total','age_star_click_buy_total','age_star_click_buy_rate','item_id','user_age_level','user_star_level']].to_csv('data/age_star_item_feature2.csv',index=None)




#从训练集三中获取用户商品相关特征
train2_f = pd.read_csv('data/train3_f.csv')
user_commodity = train2_f
# 用户点击该商品次数
t = user_commodity[['user_id','item_id']]
t['user_item_click_total'] = 1
t = t.groupby(['user_id','item_id']).agg(sum).reset_index()
user_commodity = pd.merge(user_commodity,t,on=['user_id','item_id'],how='left')

# 用户购买该商品次数
t = user_commodity[['user_id','item_id','is_trade']]
t = t.groupby(['user_id','item_id']).agg(sum).reset_index()
t.rename(columns={'is_trade':'user_item_click_buy_total'},inplace=True)
user_commodity = pd.merge(user_commodity,t,on=['user_id','item_id'],how='left')

# 用户购买该商品率
user_commodity['user_item_click_buy_rate'] = user_commodity['user_item_click_buy_total']/user_commodity['user_item_click_total']
# 该用户在多少家不同的店铺点击过该商品
t = user_commodity[['user_id','item_id','shop_id']]
t = t.drop_duplicates()
t['user_item_diff'] = 1
t = t.groupby(['user_id','item_id'])['user_item_diff'].agg('sum').reset_index()
user_commodity = pd.merge(user_commodity,t,on=['item_id','user_id'],how='left')

user_commodity[[ 'user_item_click_buy_rate','user_item_click_buy_total','user_item_click_total','user_item_diff',
                'user_id','item_id']].to_csv('data/user_item_feature3.csv',index=None)

# 该等级点击该商品次数
t = user_commodity[['user_star_level','item_id']]
t['star_item_click_total'] = 1
t = t.groupby(['user_star_level','item_id']).agg(sum).reset_index()
user_commodity = pd.merge(user_commodity,t,on=['user_star_level','item_id'],how='left')

# 该等级购买该商品次数
t = user_commodity[['user_star_level','item_id','is_trade']]
t = t.groupby(['user_star_level','item_id']).agg(sum).reset_index()
t.rename(columns={'is_trade':'star_item_click_buy_total'},inplace=True)
user_commodity = pd.merge(user_commodity,t,on=['user_star_level','item_id'],how='left')

# 该等级购买该商品率
user_commodity['star_item_click_buy_rate'] = user_commodity['star_item_click_buy_total']/user_commodity['star_item_click_total']
user_commodity[['star_item_click_buy_rate','star_item_click_buy_total','star_item_click_total','user_star_level','item_id']].to_csv('data/star_item_feature3.csv',index=None)


# 该年龄星级点击该商品次数
t = user_commodity[['item_id','user_age_level','user_star_level']]
t['age_star_click_total'] = 1
t = t.groupby(['user_age_level','user_star_level','item_id']).agg('sum').reset_index()
user_commodity = pd.merge(user_commodity,t,on=['user_age_level','user_star_level','item_id'],how='left')
# 该年龄星级购买该商品数
t = user_commodity[['item_id','user_age_level','user_star_level','is_trade']]
t = t.groupby(['user_age_level','user_star_level','item_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'age_star_click_buy_total'})
user_commodity = pd.merge(user_commodity,t,on=['user_age_level','user_star_level','item_id'],how='left')
# 该年龄星级购买该商品率
user_commodity['age_star_click_buy_rate'] = user_commodity['age_star_click_buy_total']/user_commodity['age_star_click_total']
user_commodity[['age_star_click_total','age_star_click_buy_total','age_star_click_buy_rate','item_id','user_age_level','user_star_level']].to_csv('data/age_star_item_feature3.csv',index=None)






