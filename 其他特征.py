import pandas as pd
import xgboost as xgb
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import time
from datetime import date
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
def get_bnum(s):
    all,now = s.split('-')
    times = all.split(':')
    # gaps = []
    num = 0
    for d in times:
        if d<now:
            num+=1  # gaps.append(d)
    return num
def get_anum(s):
    all, now = s.split('-')
    times = all.split(':')
    # gaps = []
    num = 0
    for d in times:
        if d > now:
            num += 1  # gaps.append(d)
    return num
def get_nnum(s):
    all, now = s.split('-')
    times = all.split(':')
    # gaps = []
    num = 0
    for d in times:
        if d == now:
            num += 1  # gaps.append(d)
    return num
# 训练预测集一上提取
train = pd.read_csv('data/train1_p.csv')
# 用户呃外特征
# 用户当天点击次数
t = train[['user_id']]
t['user_today_click'] = 1
t = t.groupby('user_id').agg('sum').reset_index()
train = pd.merge(train,t,on='user_id',how='left')

# 用户当天点击商家数目
t = train[['user_id','shop_id']]
t.drop_duplicates(inplace=True)
t['user_today_num']=1
t = t.groupby('user_id')['user_today_num'].agg('sum').reset_index()
train = pd.merge(train,t,on='user_id',how='left')
# 用户当天点击不同商品数目
t = train[['user_id','item_id']]
t.drop_duplicates(inplace=True)
t['user_today_item_num']=1
t = t.groupby('user_id')['user_today_item_num'].agg('sum').reset_index()
train = pd.merge(train,t,on='user_id',how='left')
# 用户当天点击种类数目
t = train[['user_id','cate2']]
t = t.drop_duplicates()
t['user_cate_today_sum'] = 1
t = t.groupby('user_id')['user_cate_today_sum'].agg('sum').reset_index()
train = pd.merge(train,t,on='user_id',how='left')
# 用户当天点击品牌数目
t = train[['user_id','item_brand_id']]
t = t.drop_duplicates()
t['user_brand_today_sum'] = 1
t = t.groupby('user_id')['user_brand_today_sum'].agg('sum').reset_index()
train = pd.merge(train,t,on='user_id',how='left')
# 用户上午点击次数
t = train[['user_id','is_am']]
t = t.groupby('user_id').agg('sum').reset_index()
t = t.rename(columns={'is_am':'user_am_click'})
train = pd.merge(train,t,on='user_id',how='left')
# 用户下午点击次数
train['user_pm_click'] = train['user_today_click']-train['user_am_click']
# 用户下午点击次数占总的百分比
train['user_pm_rate'] = train['user_pm_click']/train['user_today_click']
train[['user_today_click','user_today_num','user_today_item_num','user_am_click','user_pm_click','user_pm_rate',
       'user_cate_today_sum','user_brand_today_sum','user_id']].to_csv('data/other_user_feature1.csv',index=None)

# 商品呃外特征
# 该商品当天被点击次数
t = train[['item_id']]
t['item_today_click'] = 1
t = t.groupby('item_id').agg('sum').reset_index()
train = pd.merge(train,t,on='item_id',how='left')
# 该商品当天被多少不同的用户点击
t = train[['item_id','user_id']]
t = t.drop_duplicates()
t['item_diffuser_click'] = 1
t = t.groupby(['item_id'])['item_diffuser_click'].agg('sum').reset_index()
# # 该商品的点击率
# train['item_today_rate'] = train['item_today_click']/train.shape[0]

train = pd.merge(train,t,on='item_id',how='left')
train[['item_today_click','item_diffuser_click','item_id']].to_csv('data/other_item_feature1.csv',index=None)
# ,'item_today_rate'
# ,'item_diffuser_click'
# 用户商品交互二外特征
# 用户当天点击不同商品数量
t = train[['user_id','item_id']]
t['user_today_click_diff'] = 1
t = t.groupby(['user_id','item_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_id'],how='left')
# 用户在多少不同商家点击该商品
t = train[['user_id','item_id','shop_id']]
t = t.drop_duplicates()
t['user_item_ns'] = 1
t = t.groupby(['user_id','item_id'])['user_item_ns'].agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_id'],how='left')
# 用户点击该商品数量占用户当天总点击的比值
t = train[['user_id','item_id']]
t['user_item_today'] = 1
t = t.groupby(['user_id','item_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_id'],how='left')
train['user_item_rate'] = train['user_item_today']/train['user_today_click']
# 用户点击该商品占该商品当天点击的比值
train['item_user_rate'] = train['user_item_today']/train['item_today_click']
train[['user_today_click_diff','user_item_today','user_item_rate','user_item_ns','item_user_rate','item_id','user_id']].to_csv('data/other_user_item_feature1.csv',index=None)

# 商家呃外特征
# 商家当天被点击次数
t  = train[['shop_id']]
t['shop_today_beclick'] = 1
t = t.groupby('shop_id').agg('sum').reset_index()
train = pd.merge(train,t,on='shop_id',how='left')
# 商家当天被多少不同的用户点击
t  = train[['shop_id','user_id']]
t = t.drop_duplicates()
t['shop_diffuser_click'] = 1
t = t.groupby(['shop_id'])['shop_diffuser_click'].agg('sum').reset_index()
train = pd.merge(train,t,on='shop_id',how='left')
train[['shop_today_beclick','shop_diffuser_click','shop_id']].to_csv('data/other_shop_feature1.csv',index=None)
# ,
# 用户商家交互呃外特征
# 用户在该商家点击的商品数目
t = train[['user_id','shop_id','item_id']]
t = t.drop_duplicates()
t['user_shop_itnum'] = 1
t = t.groupby(['user_id','shop_id'])['user_shop_itnum'].agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','shop_id'],how='left')
# 用户点击该商家次数占用户当天点击比值
t = train[['user_id','shop_id']]
t['user_shop_today'] = 1
t = t.groupby(['user_id','shop_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','shop_id'],how='left')
train['user_shop_rate'] = train['user_shop_today']/train['user_today_click']
# 用户点击该商家次数占商家当天点击比值
train['shop_user_rate'] = train['user_shop_today']/train['shop_today_beclick']
train[['user_shop_rate','user_shop_today','shop_user_rate','user_shop_itnum','user_id','shop_id']].to_csv('data/other_user_shop_feature1.csv',index=None)

# 商品店铺的交互呃外特征
t = train[['shop_id','item_id']]
# 该商品在该店铺被点击的次数
t['item_shop_click'] = 1
t = t.groupby(['shop_id','item_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['shop_id','item_id'],how='left')
# 该商品在该店铺被点击的次数占该店铺点击比
train['item_shop_rate'] = train['item_shop_click']/train['shop_today_beclick']
# 该商品在该店铺被点击的次数占该商品点击比
train['shop_item_rate'] = train['item_shop_click']/train['item_today_click']
train[['item_shop_click','item_shop_rate','shop_item_rate','item_id','shop_id']].to_csv('data/other_item_shop_feature1.csv',index=None)

# 品牌呃外特征
# 该品牌当天被点击次数
t = train[['item_brand_id']]
t['brand_today_click'] =1
t = t.groupby('item_brand_id').agg('sum').reset_index()
train = pd.merge(train,t,on='item_brand_id',how='left')
train[['item_brand_id','brand_today_click']].to_csv('data/other_brand_feature1.csv',index=None)

# 用户品牌交互特征
t = train[['item_brand_id','user_id']]
# 该用户当天点击该品牌次数
t['user_brand_today_click'] = 1
t = t.groupby(['user_id','item_brand_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_brand_id'],how='left')
# 该用户点击该品牌次数占用户当天点击比
train['user_brand_today_rate'] = train['user_brand_today_click']/train['user_today_click']
# 该用户点击该品牌占该品牌当天点击比
train['brand_user_today_rate'] = train['user_brand_today_click']/train['brand_today_click']
train[['item_brand_id','user_id','user_brand_today_click','user_brand_today_rate','brand_user_today_rate']].to_csv('data/other_user_brand_feature1.csv',index=None)
#
# 商户品牌交互特征
# t = train[['shop_id','item_brand_id']]
# # 该商户的该品牌当天被点击次数
# t['shop_brand_today_click'] = 1
# t = t.groupby(['shop_id','item_brand_id']).agg('sum').reset_index()
# train = pd.merge(train,t,on=['shop_id','item_brand_id'],how='left')
# # 该商户该品牌被点击次数占该商户被点击次数比
# train['shop_brand_today_rate'] = train['shop_brand_today_click']/train['shop_today_beclick']
# # 该商户该品牌被点击次数占该品牌被点击次数比
# train['brand_shop_today_rate'] = train['shop_brand_today_click']/train['brand_today_click']
# train[['item_brand_id','shop_id','shop_brand_today_click','shop_brand_today_rate','brand_shop_today_rate']].to_csv('data/other_shop_brand_feature1.csv',index=None)

# 用户在该店铺点击该商品次数
t = train[['user_id','shop_id','item_id']]
t['u_s_i_tclick'] = 1
t = t.groupby(['user_id','shop_id','item_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','shop_id','item_id'],how='left')
# 用户在该店铺点击该商品次数占用户点击该商品比
train['u_s_i_tclick_rate'] = train['u_s_i_tclick']/train['user_item_today']
# 用户在该店铺点击该商品次数占商户被点击该商品比
train['u_i_s_tclick_rate'] = train['u_s_i_tclick']/train['item_shop_click']
train[['u_s_i_tclick','u_s_i_tclick_rate','user_id','u_i_s_tclick_rate','shop_id','item_id']].to_csv('data/other_user_shop_item_feature1.csv',index=None)

# 用户商品时间特征提取
t = train[['user_id','item_id','hour']]
h = train[['user_id','item_id','hour']]
h = h.sort_values('hour')
h['hour'] = h['hour'].astype(str)
h = h.groupby(['user_id','item_id'])['hour'].agg(lambda x:':'.join(x)).reset_index()
h = h.rename(columns={'hour':'time'})
t = pd.merge(t,h,on=['user_id','item_id'],how='left')
t['hour'] = t['hour'].astype(str)
t['time_list'] = t['time']+'-'+t['hour']
t['before'] = t['time_list'].apply(get_bnum)
t['after'] = t['time_list'].apply(get_anum)
t['now'] = t['time_list'].apply(get_nnum)
t['b/a'] = t['before']/t['after']
t['hour'] = t['hour'].astype(int)
t = t.drop_duplicates(subset=['user_id','item_id','hour'])
train = pd.merge(train,t,on=['user_id','item_id','hour'],how='left')
train[['before','after','now','user_id','b/a','item_id','hour']].to_csv('data/other_user_item_hour_feature1.csv',index=None)

# 用户当天点击该价格等级的数目
t = train[['user_id','item_price_level']]
t['user_price_ctoday'] = 1
t = t.groupby(['user_id','item_price_level']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_price_level'],how='left')
# 用户点击该价格等级商品占用户点击比
train['user_price_tcrate'] = train['user_price_ctoday']/train['user_today_click']
# # 用户点击改价格等级商品占该价格等级点击比
# t = train[['item_price_level']]
# t['price_num'] = 1
# t = t.groupby('item_price_level').agg('sum').reset_index()
# train = pd.merge(train,t,on='item_price_level',how='left')
# train['price_user_tcrate'] = train['user_price_ctoday']/train['price_num']

train[['user_price_ctoday','user_price_tcrate','user_id','item_price_level']].to_csv('data/other_user_price_feature1.csv',index=None)
# ,'price_user_tcrate'
# 用户当天点击该收藏等级的数目
t = train[['user_id','item_collected_level']]
t['user_collected_ctoday'] = 1
t = t.groupby(['user_id','item_collected_level']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_collected_level'],how='left')
# 用户点击该价格等级商品占用户点击比
train['user_collected_tcrate'] = train['user_collected_ctoday']/train['user_today_click']
# 用户点击该价格等级商品占价格等级点击比
t = train[['item_collected_level']]
t['collected_num'] = 1
t = t.groupby('item_collected_level').agg('sum').reset_index()
train = pd.merge(train,t,on='item_collected_level',how='left')
train['collected_user_tcrate'] = train['user_collected_ctoday']/train['collected_num']

train[['user_collected_ctoday','user_collected_tcrate','collected_user_tcrate','user_id','item_collected_level']].to_csv('data/other_user_collected_feature1.csv',index=None)

# 用户当天点击该sale等级的数目
# t = train[['user_id','item_sales_level']]
# t['user_sales_ctoday'] = 1
# t = t.groupby(['user_id','item_sales_level']).agg('sum').reset_index()
# train = pd.merge(train,t,on=['user_id','item_sales_level'],how='left')
# # 用户点击该价格等级商品占用户点击比
# train['user_sales_tcrate'] = train['user_sales_ctoday']/train['user_today_click']
# # 用户点击该价格等级商品占价格等级点击比
# t = train[['item_sales_level']]
# t['sales_num'] = 1
# t = t.groupby('item_sales_level').agg('sum').reset_index()
# train = pd.merge(train,t,on='item_sales_level',how='left')
# train['sales_user_tcrate'] = train['user_sales_ctoday']/train['sales_num']
# train[['user_sales_ctoday','user_sales_tcrate','user_id','sales_user_tcrate','item_sales_level']].to_csv('data/other_user_sales_feature1.csv',index=None)

# 用户时间特征提取
t = train[['user_id','hour']]
h = train[['user_id','hour']]
h = h.sort_values('hour')
h['hour'] = h['hour'].astype(str)
h = h.groupby(['user_id'])['hour'].agg(lambda x:':'.join(x)).reset_index()
h = h.rename(columns={'hour':'time'})
t = pd.merge(t,h,on=['user_id'],how='left')
t['hour'] = t['hour'].astype(str)
t['time_list'] = t['time']+'-'+t['hour']
t['ubefore'] = t['time_list'].apply(get_bnum)
t['uafter'] = t['time_list'].apply(get_anum)
t['unow'] = t['time_list'].apply(get_nnum)
t['ub/ua'] = t['ubefore']/t['uafter']
t['hour'] = t['hour'].astype(int)
t = t.drop_duplicates(subset=['user_id','hour'])
train = pd.merge(train,t,on=['user_id','hour'],how='left')
train['unow_rate'] = train['unow']/train['user_today_click']
train[['ubefore','uafter','unow','unow_rate','ub/ua','user_id','hour']].to_csv('data/other_user_hour_feature1.csv',index=None)

# 商户时间提取
t = train[['shop_id','hour']]
# 该商户在该时段被点击次数
t['shop_hour_tclick'] = 1
t = t.groupby(['shop_id','hour']).agg('sum').reset_index()
train = pd.merge(train,t,on=['shop_id','hour'],how='left')
# 该商户在该时段被点击率
train['shop_hour_trate'] = train['shop_hour_tclick']/train['shop_today_beclick']
train[['shop_hour_tclick','shop_hour_trate','shop_id','hour']].to_csv('data/other_shop_hour_feature1.csv',index=None)

# 用户类目提取
t = train[['user_id','cate2']]
# 用户点击该类目次数
t['user_cate_tclick'] = 1
t = t.groupby(['user_id','cate2']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','cate2'],how='left')
# 用户点击该类目占用户当天点击次数比
train['user_cate_trate'] = train['user_cate_tclick']/train['user_today_click']

train[['user_cate_tclick','user_cate_trate','user_id','cate2']].to_csv('data/other_user_cate_feature1.csv',index=None)

# # 用户在一个小时点击次数
# t1 =train[['user_id','hour']]
# t1['hourclick'] = 1
# t1 = t1.groupby(['user_id','hour'])['hourclick'].agg('sum').reset_index()
# train = pd.merge(train,t1,on=['user_id','hour'],how='left')
# train[['user_id','hour','hourclick']].to_csv('data/other_user_hourclick_feature1.csv',index=None)

# 该年龄等级当天点击时间与平均的距离的大小
t = pd.read_csv('data/age_time_feature1.csv')
t1 = train[['user_age_level','hour']]
t = t.drop_duplicates(subset=['user_age_level'])
t1 = t1.drop_duplicates(subset=['user_age_level','hour'])
t1 = pd.merge(t1,t,on=['user_age_level'],how='left')
t1['distance'] = abs(t1['hour']-t1['mean_time'])
train = pd.merge(train,t1,on=['user_age_level','hour'],how='left')
train[['distance','mean_time','user_age_level','hour']].to_csv('data/other_age_time_feature1.csv',index=None)







# 训练预测集2上提取
train = pd.read_csv('data/train2_p.csv')
# 用户呃外特征
# 用户当天点击次数
t = train[['user_id']]
t['user_today_click'] = 1
t = t.groupby('user_id').agg('sum').reset_index()
train = pd.merge(train,t,on='user_id',how='left')
# 用户当天点击种类数目
t = train[['user_id','cate2']]
t = t.drop_duplicates()
t['user_cate_today_sum'] = 1
t = t.groupby('user_id')['user_cate_today_sum'].agg('sum').reset_index()
train = pd.merge(train,t,on='user_id',how='left')
# 用户当天点击商家数目
t = train[['user_id','shop_id']]
t.drop_duplicates(inplace=True)
t['user_today_num']=1
t = t.groupby('user_id')['user_today_num'].agg('sum').reset_index()
train = pd.merge(train,t,on='user_id',how='left')
# 用户当天点击不同商品数目
t = train[['user_id','item_id']]
t.drop_duplicates(inplace=True)
t['user_today_item_num']=1
t = t.groupby('user_id')['user_today_item_num'].agg('sum').reset_index()
train = pd.merge(train,t,on='user_id',how='left')
# 用户当天点击品牌数目
t = train[['user_id','item_brand_id']]
t = t.drop_duplicates()
t['user_brand_today_sum'] = 1
t = t.groupby('user_id')['user_brand_today_sum'].agg('sum').reset_index()
train = pd.merge(train,t,on='user_id',how='left')
# 用户上午点击次数
t = train[['user_id','is_am']]
t = t.groupby('user_id').agg('sum').reset_index()
t = t.rename(columns={'is_am':'user_am_click'})
train = pd.merge(train,t,on='user_id',how='left')
# 用户下午点击次数
train['user_pm_click'] = train['user_today_click']-train['user_am_click']
# 用户下午点击次数占总的百分比
train['user_pm_rate'] = train['user_pm_click']/train['user_today_click']
train[['user_today_click','user_today_num','user_today_item_num','user_am_click','user_pm_click',
       'user_pm_rate','user_cate_today_sum','user_brand_today_sum','user_id']].to_csv('data/other_user_feature2.csv',index=None)

# 商品呃外特征
# 该商品当天被点击次数
t = train[['item_id']]
t['item_today_click'] = 1
t = t.groupby('item_id').agg('sum').reset_index()
train = pd.merge(train,t,on='item_id',how='left')
# 该商品当天被多少不同的用户点击
t = train[['item_id','user_id']]
t = t.drop_duplicates()
t['item_diffuser_click'] = 1
t = t.groupby(['item_id'])['item_diffuser_click'].agg('sum').reset_index()
train = pd.merge(train,t,on='item_id',how='left')
# # 该商品的点击率
# train['item_today_rate'] = train['item_today_click']/train.shape[0]
train[['item_today_click','item_diffuser_click','item_id']].to_csv('data/other_item_feature2.csv',index=None)
# ,'item_today_rate'
# ,'item_diffuser_click'
# 用户商品交互二外特征
# 用户当天点击不同商品数量
t = train[['user_id','item_id']]
t['user_today_click_diff'] = 1
t = t.groupby(['user_id','item_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_id'],how='left')
# 用户在多少不同商家点击该商品
t = train[['user_id','item_id','shop_id']]
t = t.drop_duplicates()
t['user_item_ns'] = 1
t = t.groupby(['user_id','item_id'])['user_item_ns'].agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_id'],how='left')
# 用户点击该商品数量占用户当天总点击的比值
t = train[['user_id','item_id']]
t['user_item_today'] = 1
t = t.groupby(['user_id','item_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_id'],how='left')
train['user_item_rate'] = train['user_item_today']/train['user_today_click']
# 用户点击该商品占该商品当天点击的比值
train['item_user_rate'] = train['user_item_today']/train['item_today_click']
train[['user_today_click_diff','user_item_today','user_item_rate','user_item_ns','item_user_rate','item_id','user_id']].to_csv('data/other_user_item_feature2.csv',index=None)

# 商家呃外特征
# 商家当天被点击次数
t  = train[['shop_id']]
t['shop_today_beclick'] = 1
t = t.groupby('shop_id').agg('sum').reset_index()
train = pd.merge(train,t,on='shop_id',how='left')
# 商家当天被多少不同的用户点击
t  = train[['shop_id','user_id']]
t = t.drop_duplicates()
t['shop_diffuser_click'] = 1
t = t.groupby(['shop_id'])['shop_diffuser_click'].agg('sum').reset_index()
train = pd.merge(train,t,on='shop_id',how='left')
train[['shop_today_beclick','shop_diffuser_click','shop_id']].to_csv('data/other_shop_feature2.csv',index=None)
# 'shop_diffuser_click',
# 用户商家交互呃外特征
# 用户在该商家点击的商品数目
t = train[['user_id','shop_id','item_id']]
t = t.drop_duplicates()
t['user_shop_itnum'] = 1
t = t.groupby(['user_id','shop_id'])['user_shop_itnum'].agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','shop_id'],how='left')

# 用户点击该商家次数占用户当天点击比值
t = train[['user_id','shop_id']]
t['user_shop_today'] = 1
t = t.groupby(['user_id','shop_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','shop_id'],how='left')
train['user_shop_rate'] = train['user_shop_today']/train['user_today_click']
# 用户点击该商家次数占商家当天点击比值
train['shop_user_rate'] = train['user_shop_today']/train['shop_today_beclick']
train[['user_shop_rate','user_shop_today','shop_user_rate','user_shop_itnum','user_id','shop_id']].to_csv('data/other_user_shop_feature2.csv',index=None)

# 商品店铺的交互呃外特征
t = train[['shop_id','item_id']]
# 该商品在该店铺被点击的次数
t['item_shop_click'] = 1
t = t.groupby(['shop_id','item_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['shop_id','item_id'],how='left')
# 该商品在该店铺被点击的次数占该店铺点击比
train['item_shop_rate'] = train['item_shop_click']/train['shop_today_beclick']
# 该商品在该店铺被点击的次数占该商品点击比
train['shop_item_rate'] = train['item_shop_click']/train['item_today_click']
train[['item_shop_click','item_shop_rate','shop_item_rate','item_id','shop_id']].to_csv('data/other_item_shop_feature2.csv',index=None)

# 品牌呃外特征
# 该品牌当天被点击次数
t = train[['item_brand_id']]
t['brand_today_click'] =1
t = t.groupby('item_brand_id').agg('sum').reset_index()
train = pd.merge(train,t,on='item_brand_id',how='left')
train[['item_brand_id','brand_today_click']].to_csv('data/other_brand_feature2.csv',index=None)

# 用户品牌交互特征
t = train[['item_brand_id','user_id']]
# 该用户当天点击该品牌次数
t['user_brand_today_click'] = 1
t = t.groupby(['user_id','item_brand_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_brand_id'],how='left')
# 该用户点击该品牌次数占用户当天点击比
train['user_brand_today_rate'] = train['user_brand_today_click']/train['user_today_click']
# 该用户点击该品牌占该品牌当天点击比
train['brand_user_today_rate'] = train['user_brand_today_click']/train['brand_today_click']
train[['item_brand_id','user_id','user_brand_today_click','user_brand_today_rate','brand_user_today_rate']].to_csv('data/other_user_brand_feature2.csv',index=None)
#
# 商户品牌交互特征
# t = train[['shop_id','item_brand_id']]
# # 该商户的该品牌当天被点击次数
# t['shop_brand_today_click'] = 1
# t = t.groupby(['shop_id','item_brand_id']).agg('sum').reset_index()
# train = pd.merge(train,t,on=['shop_id','item_brand_id'],how='left')
# # 该商户该品牌被点击次数占该商户被点击次数比
# train['shop_brand_today_rate'] = train['shop_brand_today_click']/train['shop_today_beclick']
# # 该商户该品牌被点击次数占该品牌被点击次数比
# train['brand_shop_today_rate'] = train['shop_brand_today_click']/train['brand_today_click']
# train[['item_brand_id','shop_id','shop_brand_today_click','shop_brand_today_rate','brand_shop_today_rate']].to_csv('data/other_shop_brand_feature2.csv',index=None)

# 用户在该店铺点击该商品次数
t = train[['user_id','shop_id','item_id']]
t['u_s_i_tclick'] = 1
t = t.groupby(['user_id','shop_id','item_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','shop_id','item_id'],how='left')
# 用户在该店铺点击该商品次数占用户点击该商品比
train['u_s_i_tclick_rate'] = train['u_s_i_tclick']/train['user_item_today']
# 用户在该店铺点击该商品次数占商户被点击该商品比
train['u_i_s_tclick_rate'] = train['u_s_i_tclick']/train['item_shop_click']
train[['u_s_i_tclick','u_s_i_tclick_rate','user_id','u_i_s_tclick_rate','shop_id','item_id']].to_csv('data/other_user_shop_item_feature2.csv',index=None)

# 用户商品时间特征提取
t = train[['user_id','item_id','hour']]
h = train[['user_id','item_id','hour']]
h = h.sort_values('hour')
h['hour'] = h['hour'].astype(str)
h = h.groupby(['user_id','item_id'])['hour'].agg(lambda x:':'.join(x)).reset_index()
h = h.rename(columns={'hour':'time'})
t = pd.merge(t,h,on=['user_id','item_id'],how='left')
t['hour'] = t['hour'].astype(str)
t['time_list'] = t['time']+'-'+t['hour']
t['before'] = t['time_list'].apply(get_bnum)
t['after'] = t['time_list'].apply(get_anum)
t['now'] = t['time_list'].apply(get_nnum)
t['b/a'] = t['before']/t['after']
t['hour'] = t['hour'].astype(int)
t = t.drop_duplicates(subset=['user_id','item_id','hour'])
train = pd.merge(train,t,on=['user_id','item_id','hour'],how='left')
train[['before','after','now','user_id','b/a','item_id','hour']].to_csv('data/other_user_item_hour_feature2.csv',index=None)

# 用户当天点击该价格等级的数目
t = train[['user_id','item_price_level']]
t['user_price_ctoday'] = 1
t = t.groupby(['user_id','item_price_level']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_price_level'],how='left')
# 用户点击该价格等级商品占用户点击比
train['user_price_tcrate'] = train['user_price_ctoday']/train['user_today_click']
# # 用户点击改价格等级商品占该价格等级点击比
# t = train[['item_price_level']]
# t['price_num'] = 1
# t = t.groupby('item_price_level').agg('sum').reset_index()
# train = pd.merge(train,t,on='item_price_level',how='left')
# train['price_user_tcrate'] = train['user_price_ctoday']/train['price_num']

train[['user_price_ctoday','user_price_tcrate','user_id','item_price_level']].to_csv('data/other_user_price_feature2.csv',index=None)
# ,'price_user_tcrate'
# 用户当天点击该收藏等级的数目
# t = train[['user_id','item_collected_level']]
# t['user_collected_ctoday'] = 1
# t = t.groupby(['user_id','item_collected_level']).agg('sum').reset_index()
# train = pd.merge(train,t,on=['user_id','item_collected_level'],how='left')
# # 用户点击该价格等级商品占用户点击比
# train['user_collected_tcrate'] = train['user_collected_ctoday']/train['user_today_click']
# # 用户点击该价格等级商品占价格等级点击比
# t = train[['item_collected_level']]
# t['collected_num'] = 1
# t = t.groupby('item_collected_level').agg('sum').reset_index()
# train = pd.merge(train,t,on='item_collected_level',how='left')
# train['collected_user_tcrate'] = train['user_collected_ctoday']/train['collected_num']
#
# train[['user_collected_ctoday','user_collected_tcrate','collected_user_tcrate','user_id','item_collected_level']].to_csv('data/other_user_collected_feature2.csv',index=None)

# # 用户当天点击该sale等级的数目
# t = train[['user_id','item_sales_level']]
# t['user_sales_ctoday'] = 1
# t = t.groupby(['user_id','item_sales_level']).agg('sum').reset_index()
# train = pd.merge(train,t,on=['user_id','item_sales_level'],how='left')
# # 用户点击该价格等级商品占用户点击比
# train['user_sales_tcrate'] = train['user_sales_ctoday']/train['user_today_click']
# # 用户点击该价格等级商品占价格等级点击比
# t = train[['item_sales_level']]
# t['sales_num'] = 1
# t = t.groupby('item_sales_level').agg('sum').reset_index()
# train = pd.merge(train,t,on='item_sales_level',how='left')
# train['sales_user_tcrate'] = train['user_sales_ctoday']/train['sales_num']
#
# train[['user_sales_ctoday','user_sales_tcrate','user_id','sales_user_tcrate','item_sales_level']].to_csv('data/other_user_sales_feature2.csv',index=None)

# 用户时间特征提取
t = train[['user_id','hour']]
h = train[['user_id','hour']]
h = h.sort_values('hour')
h['hour'] = h['hour'].astype(str)
h = h.groupby(['user_id'])['hour'].agg(lambda x:':'.join(x)).reset_index()
h = h.rename(columns={'hour':'time'})
t = pd.merge(t,h,on=['user_id'],how='left')
t['hour'] = t['hour'].astype(str)
t['time_list'] = t['time']+'-'+t['hour']
t['ubefore'] = t['time_list'].apply(get_bnum)
t['uafter'] = t['time_list'].apply(get_anum)
t['unow'] = t['time_list'].apply(get_nnum)
t['hour'] = t['hour'].astype(int)
t['ub/ua'] = t['ubefore']/t['uafter']
t = t.drop_duplicates(subset=['user_id','hour'])
train = pd.merge(train,t,on=['user_id','hour'],how='left')
train['unow_rate'] = train['unow']/train['user_today_click']
train[['ubefore','uafter','unow','unow_rate','ub/ua','user_id','hour']].to_csv('data/other_user_hour_feature2.csv',index=None)

# 商户时间提取
t = train[['shop_id','hour']]
# 该商户在该时段被点击次数
t['shop_hour_tclick'] = 1
t = t.groupby(['shop_id','hour']).agg('sum').reset_index()
train = pd.merge(train,t,on=['shop_id','hour'],how='left')
# 该商户在该时段被点击率
train['shop_hour_trate'] = train['shop_hour_tclick']/train['shop_today_beclick']
train[['shop_hour_tclick','shop_hour_trate','shop_id','hour']].to_csv('data/other_shop_hour_feature2.csv',index=None)
# 用户类目提取
t = train[['user_id','cate2']]
# 用户点击该类目次数
t['user_cate_tclick'] = 1
t = t.groupby(['user_id','cate2']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','cate2'],how='left')
# 用户点击该类目占用户当天点击次数比
train['user_cate_trate'] = train['user_cate_tclick']/train['user_today_click']
# # 用户点击该类目占该类目当天点击次数比
# t = train[['cate2']]
# t['cate2_today_click'] = 1
# t = t.groupby('cate2').agg('sum').reset_index()
# train = pd.merge(train,t,on='cate2',how='left')
# train['cate_user_trade'] = train['user_cate_tclick']/train['cate2_today_click']
train[['user_cate_tclick','user_cate_trate','user_id','cate2']].to_csv('data/other_user_cate_feature2.csv',index=None)

# # 用户在一个小时点击次数
# t1 =train[['user_id','hour']]
# t1['hourclick'] = 1
# t1 = t1.groupby(['user_id','hour'])['hourclick'].agg('sum').reset_index()
# train = pd.merge(train,t1,on=['user_id','hour'],how='left')
# train[['user_id','hour','hourclick']].to_csv('data/other_user_hourclick_feature2.csv',index=None)
# 该年龄等级当天点击时间与平均的距离的大小
t = pd.read_csv('data/age_time_feature2.csv')
t1 = train[['user_age_level','hour']]
t = t.drop_duplicates(subset=['user_age_level'])
t1 = t1.drop_duplicates(subset=['user_age_level','hour'])
t1 = pd.merge(t1,t,on=['user_age_level'],how='left')
t1['distance'] = abs(t1['hour']-t1['mean_time'])
train = pd.merge(train,t1,on=['user_age_level','hour'],how='left')
train[['distance','mean_time','user_age_level','hour']].to_csv('data/other_age_time_feature2.csv',index=None)





# 训练预测集3上提取
train = pd.read_csv('data/train3_p.csv')
# 用户呃外特征
# 用户当天点击次数
t = train[['user_id']]
t['user_today_click'] = 1
t = t.groupby('user_id').agg('sum').reset_index()
train = pd.merge(train,t,on='user_id',how='left')
# 用户当天点击种类数目
t = train[['user_id','cate2']]
t = t.drop_duplicates()
t['user_cate_today_sum'] = 1
t = t.groupby('user_id')['user_cate_today_sum'].agg('sum').reset_index()
train = pd.merge(train,t,on='user_id',how='left')
# 用户当天点击商家数目
t = train[['user_id','shop_id']]
t.drop_duplicates(inplace=True)
t['user_today_num']=1
t = t.groupby('user_id')['user_today_num'].agg('sum').reset_index()
train = pd.merge(train,t,on='user_id',how='left')
# 用户当天点击不同商品数目
t = train[['user_id','item_id']]
t.drop_duplicates(inplace=True)
t['user_today_item_num']=1
t = t.groupby('user_id')['user_today_item_num'].agg('sum').reset_index()
train = pd.merge(train,t,on='user_id',how='left')
# 用户当天点击品牌数目
t = train[['user_id','item_brand_id']]
t = t.drop_duplicates()
t['user_brand_today_sum'] = 1
t = t.groupby('user_id')['user_brand_today_sum'].agg('sum').reset_index()
train = pd.merge(train,t,on='user_id',how='left')
# 用户上午点击次数
t = train[['user_id','is_am']]
t = t.groupby('user_id').agg('sum').reset_index()
t = t.rename(columns={'is_am':'user_am_click'})
train = pd.merge(train,t,on='user_id',how='left')
# 用户下午点击次数
train['user_pm_click'] = train['user_today_click']-train['user_am_click']
# 用户下午点击次数占总的百分比
train['user_pm_rate'] = train['user_pm_click']/train['user_today_click']
train[['user_today_click','user_today_num','user_today_item_num','user_am_click','user_pm_click',
       'user_pm_rate','user_cate_today_sum','user_brand_today_sum','user_id']].to_csv('data/other_user_feature3.csv',index=None)

# 商品呃外特征
# 该商品当天被点击次数
t = train[['item_id']]
t['item_today_click'] = 1
t = t.groupby('item_id').agg('sum').reset_index()
train = pd.merge(train,t,on='item_id',how='left')
# 该商品当天被多少不同的用户点击
t = train[['item_id','user_id']]
t = t.drop_duplicates()
t['item_diffuser_click'] = 1
t = t.groupby(['item_id'])['item_diffuser_click'].agg('sum').reset_index()
train = pd.merge(train,t,on='item_id',how='left')
# # 该商品的点击率
# train['item_today_rate'] = train['item_today_click']/train.shape[0]
train[['item_today_click','item_diffuser_click','item_id']].to_csv('data/other_item_feature3.csv',index=None)
# ,'item_today_rate'
# 用户商品交互二外特征
# 用户当天点击不同商品数量
t = train[['user_id','item_id']]
t['user_today_click_diff'] = 1
t = t.groupby(['user_id','item_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_id'],how='left')
# 用户在多少不同商家点击该商品
t = train[['user_id','item_id','shop_id']]
t = t.drop_duplicates()
t['user_item_ns'] = 1
t = t.groupby(['user_id','item_id'])['user_item_ns'].agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_id'],how='left')
# 用户点击该商品数量占用户当天总点击的比值
t = train[['user_id','item_id']]
t['user_item_today'] = 1
t = t.groupby(['user_id','item_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_id'],how='left')
train['user_item_rate'] = train['user_item_today']/train['user_today_click']
# 用户点击该商品占该商品当天点击的比值
train['item_user_rate'] = train['user_item_today']/train['item_today_click']
train[['user_today_click_diff','user_item_today','user_item_rate','user_item_ns','item_user_rate','item_id','user_id']].to_csv('data/other_user_item_feature3.csv',index=None)

# 商家呃外特征
# 商家当天被点击次数
t  = train[['shop_id']]
t['shop_today_beclick'] = 1
t = t.groupby('shop_id').agg('sum').reset_index()
train = pd.merge(train,t,on='shop_id',how='left')
# 商家当天被多少不同的用户点击
t  = train[['shop_id','user_id']]
t = t.drop_duplicates()
t['shop_diffuser_click'] = 1
t = t.groupby(['shop_id'])['shop_diffuser_click'].agg('sum').reset_index()
train = pd.merge(train,t,on='shop_id',how='left')

train[['shop_today_beclick','shop_diffuser_click','shop_id']].to_csv('data/other_shop_feature3.csv',index=None)

# 用户商家交互呃外特征
# 用户在该商家点击的商品数目
t = train[['user_id','shop_id','item_id']]
t = t.drop_duplicates()
t['user_shop_itnum'] = 1
t = t.groupby(['user_id','shop_id'])['user_shop_itnum'].agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','shop_id'],how='left')

# 用户点击该商家次数占用户当天点击比值
t = train[['user_id','shop_id']]
t['user_shop_today'] = 1
t = t.groupby(['user_id','shop_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','shop_id'],how='left')
train['user_shop_rate'] = train['user_shop_today']/train['user_today_click']
# 用户点击该商家次数占商家当天点击比值
train['shop_user_rate'] = train['user_shop_today']/train['shop_today_beclick']
train[['user_shop_rate','user_shop_today','shop_user_rate','user_shop_itnum','user_id','shop_id']].to_csv('data/other_user_shop_feature3.csv',index=None)

# 商品店铺的交互呃外特征
t = train[['shop_id','item_id']]
# 该商品在该店铺被点击的次数
t['item_shop_click'] = 1
t = t.groupby(['shop_id','item_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['shop_id','item_id'],how='left')
# 该商品在该店铺被点击的次数占该店铺点击比
train['item_shop_rate'] = train['item_shop_click']/train['shop_today_beclick']
# 该商品在该店铺被点击的次数占该商品点击比
train['shop_item_rate'] = train['item_shop_click']/train['item_today_click']
train[['item_shop_click','item_shop_rate','shop_item_rate','item_id','shop_id']].to_csv('data/other_item_shop_feature3.csv',index=None)

# 品牌呃外特征
# 该品牌当天被点击次数
t = train[['item_brand_id']]
t['brand_today_click'] =1
t = t.groupby('item_brand_id').agg('sum').reset_index()
train = pd.merge(train,t,on='item_brand_id',how='left')
train[['item_brand_id','brand_today_click']].to_csv('data/other_brand_feature3.csv',index=None)

# 用户品牌交互特征
t = train[['item_brand_id','user_id']]
# 该用户当天点击该品牌次数
t['user_brand_today_click'] = 1
t = t.groupby(['user_id','item_brand_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_brand_id'],how='left')
# 该用户点击该品牌次数占用户当天点击比
train['user_brand_today_rate'] = train['user_brand_today_click']/train['user_today_click']
# 该用户点击该品牌占该品牌当天点击比
train['brand_user_today_rate'] = train['user_brand_today_click']/train['brand_today_click']
train[['item_brand_id','user_id','user_brand_today_click','user_brand_today_rate','brand_user_today_rate']].to_csv('data/other_user_brand_feature3.csv',index=None)
#
# 商户品牌交互特征
# t = train[['shop_id','item_brand_id']]
# # 该商户的该品牌当天被点击次数
# t['shop_brand_today_click'] = 1
# t = t.groupby(['shop_id','item_brand_id']).agg('sum').reset_index()
# train = pd.merge(train,t,on=['shop_id','item_brand_id'],how='left')
# # 该商户该品牌被点击次数占该商户被点击次数比
# train['shop_brand_today_rate'] = train['shop_brand_today_click']/train['shop_today_beclick']
# # 该商户该品牌被点击次数占该品牌被点击次数比
# train['brand_shop_today_rate'] = train['shop_brand_today_click']/train['brand_today_click']
# train[['item_brand_id','shop_id','shop_brand_today_click','shop_brand_today_rate','brand_shop_today_rate']].to_csv('data/other_shop_brand_feature3.csv',index=None)

# 用户在该店铺点击该商品次数
t = train[['user_id','shop_id','item_id']]
t['u_s_i_tclick'] = 1
t = t.groupby(['user_id','shop_id','item_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','shop_id','item_id'],how='left')
# 用户在该店铺点击该商品次数占用户点击该商品比
train['u_s_i_tclick_rate'] = train['u_s_i_tclick']/train['user_item_today']
# 用户在该店铺点击该商品次数占商户被点击该商品比
train['u_i_s_tclick_rate'] = train['u_s_i_tclick']/train['item_shop_click']
train[['u_s_i_tclick','u_s_i_tclick_rate','user_id','u_i_s_tclick_rate','shop_id','item_id']].to_csv('data/other_user_shop_item_feature3.csv',index=None)

# 用户商品时间特征提取
t = train[['user_id','item_id','hour']]
h = train[['user_id','item_id','hour']]
h = h.sort_values('hour')
h['hour'] = h['hour'].astype(str)
h = h.groupby(['user_id','item_id'])['hour'].agg(lambda x:':'.join(x)).reset_index()
h = h.rename(columns={'hour':'time'})
t = pd.merge(t,h,on=['user_id','item_id'],how='left')
t['hour'] = t['hour'].astype(str)
t['time_list'] = t['time']+'-'+t['hour']
t['before'] = t['time_list'].apply(get_bnum)
t['after'] = t['time_list'].apply(get_anum)
t['now'] = t['time_list'].apply(get_nnum)
t['b/a'] = t['before']/t['after']
t['hour'] = t['hour'].astype(int)
t = t.drop_duplicates(subset=['user_id','item_id','hour'])
train = pd.merge(train,t,on=['user_id','item_id','hour'],how='left')
train[['before','after','now','user_id','b/a','item_id','hour']].to_csv('data/other_user_item_hour_feature3.csv',index=None)

# 用户当天点击该价格等级的数目
t = train[['user_id','item_price_level']]
t['user_price_ctoday'] = 1
t = t.groupby(['user_id','item_price_level']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_price_level'],how='left')
# 用户点击该价格等级商品占用户点击比
train['user_price_tcrate'] = train['user_price_ctoday']/train['user_today_click']
# # 用户点击改价格等级商品占该价格等级点击比
# t = train[['item_price_level']]
# t['price_num'] = 1
# t = t.groupby('item_price_level').agg('sum').reset_index()
# train = pd.merge(train,t,on='item_price_level',how='left')
# train['price_user_tcrate'] = train['user_price_ctoday']/train['price_num']

train[['user_price_ctoday','user_price_tcrate','user_id','item_price_level']].to_csv('data/other_user_price_feature3.csv',index=None)
# ,'price_user_tcrate'

# 用户当天点击该收藏等级的数目
t = train[['user_id','item_collected_level']]
t['user_collected_ctoday'] = 1
t = t.groupby(['user_id','item_collected_level']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_collected_level'],how='left')
# 用户点击该价格等级商品占用户点击比
train['user_collected_tcrate'] = train['user_collected_ctoday']/train['user_today_click']
# 用户点击该价格等级商品占价格等级点击比
t = train[['item_collected_level']]
t['collected_num'] = 1
t = t.groupby('item_collected_level').agg('sum').reset_index()
train = pd.merge(train,t,on='item_collected_level',how='left')
train['collected_user_tcrate'] = train['user_collected_ctoday']/train['collected_num']

train[['user_collected_ctoday','user_collected_tcrate','collected_user_tcrate','user_id','item_collected_level']].to_csv('data/other_user_collected_feature3.csv',index=None)

# 用户当天点击该sale等级的数目
# t = train[['user_id','item_sales_level']]
# t['user_sales_ctoday'] = 1
# t = t.groupby(['user_id','item_sales_level']).agg('sum').reset_index()
# train = pd.merge(train,t,on=['user_id','item_sales_level'],how='left')
# # 用户点击该价格等级商品占用户点击比
# train['user_sales_tcrate'] = train['user_sales_ctoday']/train['user_today_click']
# # 用户点击该价格等级商品占价格等级点击比
# t = train[['item_sales_level']]
# t['sales_num'] = 1
# t = t.groupby('item_sales_level').agg('sum').reset_index()
# train = pd.merge(train,t,on='item_sales_level',how='left')
# train['sales_user_tcrate'] = train['user_sales_ctoday']/train['sales_num']
#
# train[['user_sales_ctoday','user_sales_tcrate','user_id','sales_user_tcrate','item_sales_level']].to_csv('data/other_user_sales_feature3.csv',index=None)

# 用户时间特征提取
t = train[['user_id','hour']]
h = train[['user_id','hour']]
h = h.sort_values('hour')
h['hour'] = h['hour'].astype(str)
h = h.groupby(['user_id'])['hour'].agg(lambda x:':'.join(x)).reset_index()
h = h.rename(columns={'hour':'time'})
t = pd.merge(t,h,on=['user_id'],how='left')
t['hour'] = t['hour'].astype(str)
t['time_list'] = t['time']+'-'+t['hour']
t['ubefore'] = t['time_list'].apply(get_bnum)
t['uafter'] = t['time_list'].apply(get_anum)
t['unow'] = t['time_list'].apply(get_nnum)
t['ub/ua'] = t['ubefore']/t['uafter']
t['hour'] = t['hour'].astype(int)
t = t.drop_duplicates(subset=['user_id','hour'])
train = pd.merge(train,t,on=['user_id','hour'],how='left')
train['unow_rate'] = train['unow']/train['user_today_click']
train[['ubefore','uafter','unow','unow_rate','ub/ua','user_id','hour']].to_csv('data/other_user_hour_feature3.csv',index=None)
# 商户时间提取
t = train[['shop_id','hour']]
# 该商户在该时段被点击次数
t['shop_hour_tclick'] = 1
t = t.groupby(['shop_id','hour']).agg('sum').reset_index()
train = pd.merge(train,t,on=['shop_id','hour'],how='left')
# 该商户在该时段被点击率
train['shop_hour_trate'] = train['shop_hour_tclick']/train['shop_today_beclick']
train[['shop_hour_tclick','shop_hour_trate','shop_id','hour']].to_csv('data/other_shop_hour_feature3.csv',index=None)

# 用户类目提取
t = train[['user_id','cate2']]
# 用户点击该类目次数
t['user_cate_tclick'] = 1
t = t.groupby(['user_id','cate2']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','cate2'],how='left')
# 用户点击该类目占用户当天点击次数比
train['user_cate_trate'] = train['user_cate_tclick']/train['user_today_click']
# # 用户点击该类目占该类目当天点击次数比
# t = train[['cate2']]
# t['cate2_today_click'] = 1
# t = t.groupby('cate2').agg('sum').reset_index()
# train = pd.merge(train,t,on='cate2',how='left')
# train['cate_user_trade'] = train['user_cate_tclick']/train['cate2_today_click']
train[['user_cate_tclick','user_cate_trate','user_id','cate2']].to_csv('data/other_user_cate_feature3.csv',index=None)

# # 用户在一个小时点击次数
# t1 =train[['user_id','hour']]
# t1['hourclick'] = 1
# t1 = t1.groupby(['user_id','hour'])['hourclick'].agg('sum').reset_index()
# train = pd.merge(train,t1,on=['user_id','hour'],how='left')
# train[['user_id','hour','hourclick']].to_csv('data/other_user_hourclick_feature3.csv',index=None)

# 该年龄等级当天点击时间与平均的距离的大小
t = pd.read_csv('data/age_time_feature3.csv')
t1 = train[['user_age_level','hour']]
t = t.drop_duplicates(subset=['user_age_level'])
t1 = t1.drop_duplicates(subset=['user_age_level','hour'])
t1 = pd.merge(t1,t,on=['user_age_level'],how='left')
t1['distance'] = abs(t1['hour']-t1['mean_time'])
train = pd.merge(train,t1,on=['user_age_level','hour'],how='left')
train[['distance','mean_time','user_age_level','hour']].to_csv('data/other_age_time_feature3.csv',index=None)





# 训练预测集3上提取
train = pd.read_csv('data/train4_p.csv')
# 用户呃外特征
# 用户当天点击次数
t = train[['user_id']]
t['user_today_click'] = 1
t = t.groupby('user_id').agg('sum').reset_index()
train = pd.merge(train,t,on='user_id',how='left')
# 用户当天点击种类数目
t = train[['user_id','cate2']]
t = t.drop_duplicates()
t['user_cate_today_sum'] = 1
t = t.groupby('user_id')['user_cate_today_sum'].agg('sum').reset_index()
train = pd.merge(train,t,on='user_id',how='left')
# 用户当天点击商家数目
t = train[['user_id','shop_id']]
t.drop_duplicates(inplace=True)
t['user_today_num']=1
t = t.groupby('user_id')['user_today_num'].agg('sum').reset_index()
train = pd.merge(train,t,on='user_id',how='left')
# 用户当天点击不同商品数目
t = train[['user_id','item_id']]
t.drop_duplicates(inplace=True)
t['user_today_item_num']=1
t = t.groupby('user_id')['user_today_item_num'].agg('sum').reset_index()
train = pd.merge(train,t,on='user_id',how='left')
# 用户当天点击品牌数目
t = train[['user_id','item_brand_id']]
t = t.drop_duplicates()
t['user_brand_today_sum'] = 1
t = t.groupby('user_id')['user_brand_today_sum'].agg('sum').reset_index()
train = pd.merge(train,t,on='user_id',how='left')
# 用户上午点击次数
t = train[['user_id','is_am']]
t = t.groupby('user_id').agg('sum').reset_index()
t = t.rename(columns={'is_am':'user_am_click'})
train = pd.merge(train,t,on='user_id',how='left')
# 用户下午点击次数
train['user_pm_click'] = train['user_today_click']-train['user_am_click']
# 用户下午点击次数占总的百分比
train['user_pm_rate'] = train['user_pm_click']/train['user_today_click']
train[['user_today_click','user_today_num','user_today_item_num','user_am_click','user_pm_click',
       'user_pm_rate','user_cate_today_sum','user_brand_today_sum','user_id']].to_csv('data/other_user_feature4.csv',index=None)

# 商品呃外特征
# 该商品当天被点击次数
t = train[['item_id']]
t['item_today_click'] = 1
t = t.groupby('item_id').agg('sum').reset_index()
train = pd.merge(train,t,on='item_id',how='left')
# 该商品当天被多少不同的用户点击
t = train[['item_id','user_id']]
t = t.drop_duplicates()
t['item_diffuser_click'] = 1
t = t.groupby(['item_id'])['item_diffuser_click'].agg('sum').reset_index()
train = pd.merge(train,t,on='item_id',how='left')
# # 该商品的点击率
# train['item_today_rate'] = train['item_today_click']/train.shape[0]
train[['item_today_click','item_diffuser_click','item_id']].to_csv('data/other_item_feature4.csv',index=None)
# ,'item_today_rate'
# 用户商品交互二外特征
# 用户当天点击不同商品数量
t = train[['user_id','item_id']]
t['user_today_click_diff'] = 1
t = t.groupby(['user_id','item_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_id'],how='left')
# 用户在多少不同商家点击该商品
t = train[['user_id','item_id','shop_id']]
t = t.drop_duplicates()
t['user_item_ns'] = 1
t = t.groupby(['user_id','item_id'])['user_item_ns'].agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_id'],how='left')
# 用户点击该商品数量占用户当天总点击的比值
t = train[['user_id','item_id']]
t['user_item_today'] = 1
t = t.groupby(['user_id','item_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_id'],how='left')
train['user_item_rate'] = train['user_item_today']/train['user_today_click']
# 用户点击该商品占该商品当天点击的比值
train['item_user_rate'] = train['user_item_today']/train['item_today_click']
train[['user_today_click_diff','user_item_today','user_item_rate','user_item_ns','item_user_rate','item_id','user_id']].to_csv('data/other_user_item_feature4.csv',index=None)

# 商家呃外特征
# 商家当天被点击次数
t  = train[['shop_id']]
t['shop_today_beclick'] = 1
t = t.groupby('shop_id').agg('sum').reset_index()
train = pd.merge(train,t,on='shop_id',how='left')
# 商家当天被多少不同的用户点击
t  = train[['shop_id','user_id']]
t = t.drop_duplicates()
t['shop_diffuser_click'] = 1
t = t.groupby(['shop_id'])['shop_diffuser_click'].agg('sum').reset_index()
train = pd.merge(train,t,on='shop_id',how='left')

train[['shop_today_beclick','shop_diffuser_click','shop_id']].to_csv('data/other_shop_feature4.csv',index=None)

# 用户商家交互呃外特征
# 用户在该商家点击的商品数目
t = train[['user_id','shop_id','item_id']]
t = t.drop_duplicates()
t['user_shop_itnum'] = 1
t = t.groupby(['user_id','shop_id'])['user_shop_itnum'].agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','shop_id'],how='left')

# 用户点击该商家次数占用户当天点击比值
t = train[['user_id','shop_id']]
t['user_shop_today'] = 1
t = t.groupby(['user_id','shop_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','shop_id'],how='left')
train['user_shop_rate'] = train['user_shop_today']/train['user_today_click']
# 用户点击该商家次数占商家当天点击比值
train['shop_user_rate'] = train['user_shop_today']/train['shop_today_beclick']
train[['user_shop_rate','user_shop_today','shop_user_rate','user_shop_itnum','user_id','shop_id']].to_csv('data/other_user_shop_feature4.csv',index=None)

# 商品店铺的交互呃外特征
t = train[['shop_id','item_id']]
# 该商品在该店铺被点击的次数
t['item_shop_click'] = 1
t = t.groupby(['shop_id','item_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['shop_id','item_id'],how='left')
# 该商品在该店铺被点击的次数占该店铺点击比
train['item_shop_rate'] = train['item_shop_click']/train['shop_today_beclick']
# 该商品在该店铺被点击的次数占该商品点击比
train['shop_item_rate'] = train['item_shop_click']/train['item_today_click']
train[['item_shop_click','item_shop_rate','shop_item_rate','item_id','shop_id']].to_csv('data/other_item_shop_feature4.csv',index=None)

# 品牌呃外特征
# 该品牌当天被点击次数
t = train[['item_brand_id']]
t['brand_today_click'] =1
t = t.groupby('item_brand_id').agg('sum').reset_index()
train = pd.merge(train,t,on='item_brand_id',how='left')
train[['item_brand_id','brand_today_click']].to_csv('data/other_brand_feature4.csv',index=None)

# 用户品牌交互特征
t = train[['item_brand_id','user_id']]
# 该用户当天点击该品牌次数
t['user_brand_today_click'] = 1
t = t.groupby(['user_id','item_brand_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_brand_id'],how='left')
# 该用户点击该品牌次数占用户当天点击比
train['user_brand_today_rate'] = train['user_brand_today_click']/train['user_today_click']
# 该用户点击该品牌占该品牌当天点击比
train['brand_user_today_rate'] = train['user_brand_today_click']/train['brand_today_click']
train[['item_brand_id','user_id','user_brand_today_click','user_brand_today_rate','brand_user_today_rate']].to_csv('data/other_user_brand_feature4.csv',index=None)
#
# 商户品牌交互特征
# t = train[['shop_id','item_brand_id']]
# # 该商户的该品牌当天被点击次数
# t['shop_brand_today_click'] = 1
# t = t.groupby(['shop_id','item_brand_id']).agg('sum').reset_index()
# train = pd.merge(train,t,on=['shop_id','item_brand_id'],how='left')
# # 该商户该品牌被点击次数占该商户被点击次数比
# train['shop_brand_today_rate'] = train['shop_brand_today_click']/train['shop_today_beclick']
# # 该商户该品牌被点击次数占该品牌被点击次数比
# train['brand_shop_today_rate'] = train['shop_brand_today_click']/train['brand_today_click']
# train[['item_brand_id','shop_id','shop_brand_today_click','shop_brand_today_rate','brand_shop_today_rate']].to_csv('data/other_shop_brand_feature3.csv',index=None)

# 用户在该店铺点击该商品次数
t = train[['user_id','shop_id','item_id']]
t['u_s_i_tclick'] = 1
t = t.groupby(['user_id','shop_id','item_id']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','shop_id','item_id'],how='left')
# 用户在该店铺点击该商品次数占用户点击该商品比
train['u_s_i_tclick_rate'] = train['u_s_i_tclick']/train['user_item_today']
# 用户在该店铺点击该商品次数占商户被点击该商品比
train['u_i_s_tclick_rate'] = train['u_s_i_tclick']/train['item_shop_click']
train[['u_s_i_tclick','u_s_i_tclick_rate','user_id','u_i_s_tclick_rate','shop_id','item_id']].to_csv('data/other_user_shop_item_feature4.csv',index=None)

# 用户商品时间特征提取
t = train[['user_id','item_id','hour']]
h = train[['user_id','item_id','hour']]
h = h.sort_values('hour')
h['hour'] = h['hour'].astype(str)
h = h.groupby(['user_id','item_id'])['hour'].agg(lambda x:':'.join(x)).reset_index()
h = h.rename(columns={'hour':'time'})
t = pd.merge(t,h,on=['user_id','item_id'],how='left')
t['hour'] = t['hour'].astype(str)
t['time_list'] = t['time']+'-'+t['hour']
t['before'] = t['time_list'].apply(get_bnum)
t['after'] = t['time_list'].apply(get_anum)
t['now'] = t['time_list'].apply(get_nnum)
t['b/a'] = t['before']/t['after']
t['hour'] = t['hour'].astype(int)
t = t.drop_duplicates(subset=['user_id','item_id','hour'])
train = pd.merge(train,t,on=['user_id','item_id','hour'],how='left')
train[['before','after','now','user_id','b/a','item_id','hour']].to_csv('data/other_user_item_hour_feature4.csv',index=None)

# 用户当天点击该价格等级的数目
t = train[['user_id','item_price_level']]
t['user_price_ctoday'] = 1
t = t.groupby(['user_id','item_price_level']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_price_level'],how='left')
# 用户点击该价格等级商品占用户点击比
train['user_price_tcrate'] = train['user_price_ctoday']/train['user_today_click']
# # 用户点击改价格等级商品占该价格等级点击比
# t = train[['item_price_level']]
# t['price_num'] = 1
# t = t.groupby('item_price_level').agg('sum').reset_index()
# train = pd.merge(train,t,on='item_price_level',how='left')
# train['price_user_tcrate'] = train['user_price_ctoday']/train['price_num']

train[['user_price_ctoday','user_price_tcrate','user_id','item_price_level']].to_csv('data/other_user_price_feature4.csv',index=None)
# ,'price_user_tcrate'

# 用户当天点击该收藏等级的数目
t = train[['user_id','item_collected_level']]
t['user_collected_ctoday'] = 1
t = t.groupby(['user_id','item_collected_level']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','item_collected_level'],how='left')
# 用户点击该价格等级商品占用户点击比
train['user_collected_tcrate'] = train['user_collected_ctoday']/train['user_today_click']
# 用户点击该价格等级商品占价格等级点击比
t = train[['item_collected_level']]
t['collected_num'] = 1
t = t.groupby('item_collected_level').agg('sum').reset_index()
train = pd.merge(train,t,on='item_collected_level',how='left')
train['collected_user_tcrate'] = train['user_collected_ctoday']/train['collected_num']

train[['user_collected_ctoday','user_collected_tcrate','collected_user_tcrate','user_id','item_collected_level']].to_csv('data/other_user_collected_feature4.csv',index=None)

# 用户当天点击该sale等级的数目
# t = train[['user_id','item_sales_level']]
# t['user_sales_ctoday'] = 1
# t = t.groupby(['user_id','item_sales_level']).agg('sum').reset_index()
# train = pd.merge(train,t,on=['user_id','item_sales_level'],how='left')
# # 用户点击该价格等级商品占用户点击比
# train['user_sales_tcrate'] = train['user_sales_ctoday']/train['user_today_click']
# # 用户点击该价格等级商品占价格等级点击比
# t = train[['item_sales_level']]
# t['sales_num'] = 1
# t = t.groupby('item_sales_level').agg('sum').reset_index()
# train = pd.merge(train,t,on='item_sales_level',how='left')
# train['sales_user_tcrate'] = train['user_sales_ctoday']/train['sales_num']
#
# train[['user_sales_ctoday','user_sales_tcrate','user_id','sales_user_tcrate','item_sales_level']].to_csv('data/other_user_sales_feature3.csv',index=None)

# 用户时间特征提取
t = train[['user_id','hour']]
h = train[['user_id','hour']]
h = h.sort_values('hour')
h['hour'] = h['hour'].astype(str)
h = h.groupby(['user_id'])['hour'].agg(lambda x:':'.join(x)).reset_index()
h = h.rename(columns={'hour':'time'})
t = pd.merge(t,h,on=['user_id'],how='left')
t['hour'] = t['hour'].astype(str)
t['time_list'] = t['time']+'-'+t['hour']
t['ubefore'] = t['time_list'].apply(get_bnum)
t['uafter'] = t['time_list'].apply(get_anum)
t['unow'] = t['time_list'].apply(get_nnum)
t['ub/ua'] = t['ubefore']/t['uafter']
t['hour'] = t['hour'].astype(int)
t = t.drop_duplicates(subset=['user_id','hour'])
train = pd.merge(train,t,on=['user_id','hour'],how='left')
train['unow_rate'] = train['unow']/train['user_today_click']
train[['ubefore','uafter','unow','unow_rate','ub/ua','user_id','hour']].to_csv('data/other_user_hour_feature4.csv',index=None)
# 商户时间提取
t = train[['shop_id','hour']]
# 该商户在该时段被点击次数
t['shop_hour_tclick'] = 1
t = t.groupby(['shop_id','hour']).agg('sum').reset_index()
train = pd.merge(train,t,on=['shop_id','hour'],how='left')
# 该商户在该时段被点击率
train['shop_hour_trate'] = train['shop_hour_tclick']/train['shop_today_beclick']
train[['shop_hour_tclick','shop_hour_trate','shop_id','hour']].to_csv('data/other_shop_hour_feature4.csv',index=None)

# 用户类目提取
t = train[['user_id','cate2']]
# 用户点击该类目次数
t['user_cate_tclick'] = 1
t = t.groupby(['user_id','cate2']).agg('sum').reset_index()
train = pd.merge(train,t,on=['user_id','cate2'],how='left')
# 用户点击该类目占用户当天点击次数比
train['user_cate_trate'] = train['user_cate_tclick']/train['user_today_click']
# # 用户点击该类目占该类目当天点击次数比
# t = train[['cate2']]
# t['cate2_today_click'] = 1
# t = t.groupby('cate2').agg('sum').reset_index()
# train = pd.merge(train,t,on='cate2',how='left')
# train['cate_user_trade'] = train['user_cate_tclick']/train['cate2_today_click']
train[['user_cate_tclick','user_cate_trate','user_id','cate2']].to_csv('data/other_user_cate_feature4.csv',index=None)

# # 用户在一个小时点击次数
# t1 =train[['user_id','hour']]
# t1['hourclick'] = 1
# t1 = t1.groupby(['user_id','hour'])['hourclick'].agg('sum').reset_index()
# train = pd.merge(train,t1,on=['user_id','hour'],how='left')
# train[['user_id','hour','hourclick']].to_csv('data/other_user_hourclick_feature3.csv',index=None)

# 该年龄等级当天点击时间与平均的距离的大小
t = pd.read_csv('data/age_time_feature4.csv')
t1 = train[['user_age_level','hour']]
t = t.drop_duplicates(subset=['user_age_level'])
t1 = t1.drop_duplicates(subset=['user_age_level','hour'])
t1 = pd.merge(t1,t,on=['user_age_level'],how='left')
t1['distance'] = abs(t1['hour']-t1['mean_time'])
train = pd.merge(train,t1,on=['user_age_level','hour'],how='left')
train[['distance','mean_time','user_age_level','hour']].to_csv('data/other_age_time_feature4.csv',index=None)





# # 训练预测集3上提取
# train = pd.read_csv('data/train5_p.csv')
# # 用户呃外特征
# # 用户当天点击次数
# t = train[['user_id']]
# t['user_today_click'] = 1
# t = t.groupby('user_id').agg('sum').reset_index()
# train = pd.merge(train,t,on='user_id',how='left')
# # 用户当天点击种类数目
# t = train[['user_id','cate2']]
# t = t.drop_duplicates()
# t['user_cate_today_sum'] = 1
# t = t.groupby('user_id')['user_cate_today_sum'].agg('sum').reset_index()
# train = pd.merge(train,t,on='user_id',how='left')
# # 用户当天点击商家数目
# t = train[['user_id','shop_id']]
# t.drop_duplicates(inplace=True)
# t['user_today_num']=1
# t = t.groupby('user_id')['user_today_num'].agg('sum').reset_index()
# train = pd.merge(train,t,on='user_id',how='left')
# # 用户当天点击不同商品数目
# t = train[['user_id','item_id']]
# t.drop_duplicates(inplace=True)
# t['user_today_item_num']=1
# t = t.groupby('user_id')['user_today_item_num'].agg('sum').reset_index()
# train = pd.merge(train,t,on='user_id',how='left')
# # 用户当天点击品牌数目
# t = train[['user_id','item_brand_id']]
# t = t.drop_duplicates()
# t['user_brand_today_sum'] = 1
# t = t.groupby('user_id')['user_brand_today_sum'].agg('sum').reset_index()
# train = pd.merge(train,t,on='user_id',how='left')
# # 用户上午点击次数
# t = train[['user_id','is_am']]
# t = t.groupby('user_id').agg('sum').reset_index()
# t = t.rename(columns={'is_am':'user_am_click'})
# train = pd.merge(train,t,on='user_id',how='left')
# # 用户下午点击次数
# train['user_pm_click'] = train['user_today_click']-train['user_am_click']
# # 用户下午点击次数占总的百分比
# train['user_pm_rate'] = train['user_pm_click']/train['user_today_click']
# train[['user_today_click','user_today_num','user_today_item_num','user_am_click','user_pm_click',
#        'user_pm_rate','user_cate_today_sum','user_brand_today_sum','user_id']].to_csv('data/other_user_feature5.csv',index=None)
#
# # 商品呃外特征
# # 该商品当天被点击次数
# t = train[['item_id']]
# t['item_today_click'] = 1
# t = t.groupby('item_id').agg('sum').reset_index()
# train = pd.merge(train,t,on='item_id',how='left')
# # 该商品当天被多少不同的用户点击
# t = train[['item_id','user_id']]
# t = t.drop_duplicates()
# t['item_diffuser_click'] = 1
# t = t.groupby(['item_id'])['item_diffuser_click'].agg('sum').reset_index()
# train = pd.merge(train,t,on='item_id',how='left')
# # # 该商品的点击率
# # train['item_today_rate'] = train['item_today_click']/train.shape[0]
# train[['item_today_click','item_diffuser_click','item_id']].to_csv('data/other_item_feature5.csv',index=None)
# # ,'item_today_rate'
# # 用户商品交互二外特征
# # 用户当天点击不同商品数量
# t = train[['user_id','item_id']]
# t['user_today_click_diff'] = 1
# t = t.groupby(['user_id','item_id']).agg('sum').reset_index()
# train = pd.merge(train,t,on=['user_id','item_id'],how='left')
# # 用户在多少不同商家点击该商品
# t = train[['user_id','item_id','shop_id']]
# t = t.drop_duplicates()
# t['user_item_ns'] = 1
# t = t.groupby(['user_id','item_id'])['user_item_ns'].agg('sum').reset_index()
# train = pd.merge(train,t,on=['user_id','item_id'],how='left')
# # 用户点击该商品数量占用户当天总点击的比值
# t = train[['user_id','item_id']]
# t['user_item_today'] = 1
# t = t.groupby(['user_id','item_id']).agg('sum').reset_index()
# train = pd.merge(train,t,on=['user_id','item_id'],how='left')
# train['user_item_rate'] = train['user_item_today']/train['user_today_click']
# # 用户点击该商品占该商品当天点击的比值
# train['item_user_rate'] = train['user_item_today']/train['item_today_click']
# train[['user_today_click_diff','user_item_today','user_item_rate','user_item_ns','item_user_rate','item_id','user_id']].to_csv('data/other_user_item_feature5.csv',index=None)
#
# # 商家呃外特征
# # 商家当天被点击次数
# t  = train[['shop_id']]
# t['shop_today_beclick'] = 1
# t = t.groupby('shop_id').agg('sum').reset_index()
# train = pd.merge(train,t,on='shop_id',how='left')
# # 商家当天被多少不同的用户点击
# t  = train[['shop_id','user_id']]
# t = t.drop_duplicates()
# t['shop_diffuser_click'] = 1
# t = t.groupby(['shop_id'])['shop_diffuser_click'].agg('sum').reset_index()
# train = pd.merge(train,t,on='shop_id',how='left')
#
# train[['shop_today_beclick','shop_diffuser_click','shop_id']].to_csv('data/other_shop_feature5.csv',index=None)
#
# # 用户商家交互呃外特征
# # 用户在该商家点击的商品数目
# t = train[['user_id','shop_id','item_id']]
# t = t.drop_duplicates()
# t['user_shop_itnum'] = 1
# t = t.groupby(['user_id','shop_id'])['user_shop_itnum'].agg('sum').reset_index()
# train = pd.merge(train,t,on=['user_id','shop_id'],how='left')
#
# # 用户点击该商家次数占用户当天点击比值
# t = train[['user_id','shop_id']]
# t['user_shop_today'] = 1
# t = t.groupby(['user_id','shop_id']).agg('sum').reset_index()
# train = pd.merge(train,t,on=['user_id','shop_id'],how='left')
# train['user_shop_rate'] = train['user_shop_today']/train['user_today_click']
# # 用户点击该商家次数占商家当天点击比值
# train['shop_user_rate'] = train['user_shop_today']/train['shop_today_beclick']
# train[['user_shop_rate','user_shop_today','shop_user_rate','user_shop_itnum','user_id','shop_id']].to_csv('data/other_user_shop_feature5.csv',index=None)
#
# # 商品店铺的交互呃外特征
# t = train[['shop_id','item_id']]
# # 该商品在该店铺被点击的次数
# t['item_shop_click'] = 1
# t = t.groupby(['shop_id','item_id']).agg('sum').reset_index()
# train = pd.merge(train,t,on=['shop_id','item_id'],how='left')
# # 该商品在该店铺被点击的次数占该店铺点击比
# train['item_shop_rate'] = train['item_shop_click']/train['shop_today_beclick']
# # 该商品在该店铺被点击的次数占该商品点击比
# train['shop_item_rate'] = train['item_shop_click']/train['item_today_click']
# train[['item_shop_click','item_shop_rate','shop_item_rate','item_id','shop_id']].to_csv('data/other_item_shop_feature5.csv',index=None)
#
# # 品牌呃外特征
# # 该品牌当天被点击次数
# t = train[['item_brand_id']]
# t['brand_today_click'] =1
# t = t.groupby('item_brand_id').agg('sum').reset_index()
# train = pd.merge(train,t,on='item_brand_id',how='left')
# train[['item_brand_id','brand_today_click']].to_csv('data/other_brand_feature5.csv',index=None)
#
# # 用户品牌交互特征
# t = train[['item_brand_id','user_id']]
# # 该用户当天点击该品牌次数
# t['user_brand_today_click'] = 1
# t = t.groupby(['user_id','item_brand_id']).agg('sum').reset_index()
# train = pd.merge(train,t,on=['user_id','item_brand_id'],how='left')
# # 该用户点击该品牌次数占用户当天点击比
# train['user_brand_today_rate'] = train['user_brand_today_click']/train['user_today_click']
# # 该用户点击该品牌占该品牌当天点击比
# train['brand_user_today_rate'] = train['user_brand_today_click']/train['brand_today_click']
# train[['item_brand_id','user_id','user_brand_today_click','user_brand_today_rate','brand_user_today_rate']].to_csv('data/other_user_brand_feature5.csv',index=None)
# #
# # 商户品牌交互特征
# # t = train[['shop_id','item_brand_id']]
# # # 该商户的该品牌当天被点击次数
# # t['shop_brand_today_click'] = 1
# # t = t.groupby(['shop_id','item_brand_id']).agg('sum').reset_index()
# # train = pd.merge(train,t,on=['shop_id','item_brand_id'],how='left')
# # # 该商户该品牌被点击次数占该商户被点击次数比
# # train['shop_brand_today_rate'] = train['shop_brand_today_click']/train['shop_today_beclick']
# # # 该商户该品牌被点击次数占该品牌被点击次数比
# # train['brand_shop_today_rate'] = train['shop_brand_today_click']/train['brand_today_click']
# # train[['item_brand_id','shop_id','shop_brand_today_click','shop_brand_today_rate','brand_shop_today_rate']].to_csv('data/other_shop_brand_feature3.csv',index=None)
#
# # 用户在该店铺点击该商品次数
# t = train[['user_id','shop_id','item_id']]
# t['u_s_i_tclick'] = 1
# t = t.groupby(['user_id','shop_id','item_id']).agg('sum').reset_index()
# train = pd.merge(train,t,on=['user_id','shop_id','item_id'],how='left')
# # 用户在该店铺点击该商品次数占用户点击该商品比
# train['u_s_i_tclick_rate'] = train['u_s_i_tclick']/train['user_item_today']
# # 用户在该店铺点击该商品次数占商户被点击该商品比
# train['u_i_s_tclick_rate'] = train['u_s_i_tclick']/train['item_shop_click']
# train[['u_s_i_tclick','u_s_i_tclick_rate','user_id','u_i_s_tclick_rate','shop_id','item_id']].to_csv('data/other_user_shop_item_feature5.csv',index=None)
#
# # 用户商品时间特征提取
# t = train[['user_id','item_id','hour']]
# h = train[['user_id','item_id','hour']]
# h = h.sort_values('hour')
# h['hour'] = h['hour'].astype(str)
# h = h.groupby(['user_id','item_id'])['hour'].agg(lambda x:':'.join(x)).reset_index()
# h = h.rename(columns={'hour':'time'})
# t = pd.merge(t,h,on=['user_id','item_id'],how='left')
# t['hour'] = t['hour'].astype(str)
# t['time_list'] = t['time']+'-'+t['hour']
# t['before'] = t['time_list'].apply(get_bnum)
# t['after'] = t['time_list'].apply(get_anum)
# t['now'] = t['time_list'].apply(get_nnum)
# t['b/a'] = t['before']/t['after']
# t['hour'] = t['hour'].astype(int)
# t = t.drop_duplicates(subset=['user_id','item_id','hour'])
# train = pd.merge(train,t,on=['user_id','item_id','hour'],how='left')
# train[['before','after','now','user_id','b/a','item_id','hour']].to_csv('data/other_user_item_hour_feature5.csv',index=None)
#
# # 用户当天点击该价格等级的数目
# t = train[['user_id','item_price_level']]
# t['user_price_ctoday'] = 1
# t = t.groupby(['user_id','item_price_level']).agg('sum').reset_index()
# train = pd.merge(train,t,on=['user_id','item_price_level'],how='left')
# # 用户点击该价格等级商品占用户点击比
# train['user_price_tcrate'] = train['user_price_ctoday']/train['user_today_click']
# # # 用户点击改价格等级商品占该价格等级点击比
# # t = train[['item_price_level']]
# # t['price_num'] = 1
# # t = t.groupby('item_price_level').agg('sum').reset_index()
# # train = pd.merge(train,t,on='item_price_level',how='left')
# # train['price_user_tcrate'] = train['user_price_ctoday']/train['price_num']
#
# train[['user_price_ctoday','user_price_tcrate','user_id','item_price_level']].to_csv('data/other_user_price_feature5.csv',index=None)
# # ,'price_user_tcrate'
#
# # 用户当天点击该收藏等级的数目
# t = train[['user_id','item_collected_level']]
# t['user_collected_ctoday'] = 1
# t = t.groupby(['user_id','item_collected_level']).agg('sum').reset_index()
# train = pd.merge(train,t,on=['user_id','item_collected_level'],how='left')
# # 用户点击该价格等级商品占用户点击比
# train['user_collected_tcrate'] = train['user_collected_ctoday']/train['user_today_click']
# # 用户点击该价格等级商品占价格等级点击比
# t = train[['item_collected_level']]
# t['collected_num'] = 1
# t = t.groupby('item_collected_level').agg('sum').reset_index()
# train = pd.merge(train,t,on='item_collected_level',how='left')
# train['collected_user_tcrate'] = train['user_collected_ctoday']/train['collected_num']
#
# train[['user_collected_ctoday','user_collected_tcrate','collected_user_tcrate','user_id','item_collected_level']].to_csv('data/other_user_collected_feature5.csv',index=None)
#
# # 用户当天点击该sale等级的数目
# # t = train[['user_id','item_sales_level']]
# # t['user_sales_ctoday'] = 1
# # t = t.groupby(['user_id','item_sales_level']).agg('sum').reset_index()
# # train = pd.merge(train,t,on=['user_id','item_sales_level'],how='left')
# # # 用户点击该价格等级商品占用户点击比
# # train['user_sales_tcrate'] = train['user_sales_ctoday']/train['user_today_click']
# # # 用户点击该价格等级商品占价格等级点击比
# # t = train[['item_sales_level']]
# # t['sales_num'] = 1
# # t = t.groupby('item_sales_level').agg('sum').reset_index()
# # train = pd.merge(train,t,on='item_sales_level',how='left')
# # train['sales_user_tcrate'] = train['user_sales_ctoday']/train['sales_num']
# #
# # train[['user_sales_ctoday','user_sales_tcrate','user_id','sales_user_tcrate','item_sales_level']].to_csv('data/other_user_sales_feature3.csv',index=None)
#
# # 用户时间特征提取
# t = train[['user_id','hour']]
# h = train[['user_id','hour']]
# h = h.sort_values('hour')
# h['hour'] = h['hour'].astype(str)
# h = h.groupby(['user_id'])['hour'].agg(lambda x:':'.join(x)).reset_index()
# h = h.rename(columns={'hour':'time'})
# t = pd.merge(t,h,on=['user_id'],how='left')
# t['hour'] = t['hour'].astype(str)
# t['time_list'] = t['time']+'-'+t['hour']
# t['ubefore'] = t['time_list'].apply(get_bnum)
# t['uafter'] = t['time_list'].apply(get_anum)
# t['unow'] = t['time_list'].apply(get_nnum)
# t['ub/ua'] = t['ubefore']/t['uafter']
# t['hour'] = t['hour'].astype(int)
# t = t.drop_duplicates(subset=['user_id','hour'])
# train = pd.merge(train,t,on=['user_id','hour'],how='left')
# train['unow_rate'] = train['unow']/train['user_today_click']
# train[['ubefore','uafter','unow','unow_rate','ub/ua','user_id','hour']].to_csv('data/other_user_hour_feature5.csv',index=None)
# # 商户时间提取
# t = train[['shop_id','hour']]
# # 该商户在该时段被点击次数
# t['shop_hour_tclick'] = 1
# t = t.groupby(['shop_id','hour']).agg('sum').reset_index()
# train = pd.merge(train,t,on=['shop_id','hour'],how='left')
# # 该商户在该时段被点击率
# train['shop_hour_trate'] = train['shop_hour_tclick']/train['shop_today_beclick']
# train[['shop_hour_tclick','shop_hour_trate','shop_id','hour']].to_csv('data/other_shop_hour_feature5.csv',index=None)
#
# # 用户类目提取
# t = train[['user_id','cate2']]
# # 用户点击该类目次数
# t['user_cate_tclick'] = 1
# t = t.groupby(['user_id','cate2']).agg('sum').reset_index()
# train = pd.merge(train,t,on=['user_id','cate2'],how='left')
# # 用户点击该类目占用户当天点击次数比
# train['user_cate_trate'] = train['user_cate_tclick']/train['user_today_click']
# # # 用户点击该类目占该类目当天点击次数比
# # t = train[['cate2']]
# # t['cate2_today_click'] = 1
# # t = t.groupby('cate2').agg('sum').reset_index()
# # train = pd.merge(train,t,on='cate2',how='left')
# # train['cate_user_trade'] = train['user_cate_tclick']/train['cate2_today_click']
# train[['user_cate_tclick','user_cate_trate','user_id','cate2']].to_csv('data/other_user_cate_feature5.csv',index=None)
#
# # # 用户在一个小时点击次数
# # t1 =train[['user_id','hour']]
# # t1['hourclick'] = 1
# # t1 = t1.groupby(['user_id','hour'])['hourclick'].agg('sum').reset_index()
# # train = pd.merge(train,t1,on=['user_id','hour'],how='left')
# # train[['user_id','hour','hourclick']].to_csv('data/other_user_hourclick_feature3.csv',index=None)
#
# # 该年龄等级当天点击时间与平均的距离的大小
# t = pd.read_csv('data/age_time_feature5.csv')
# t1 = train[['user_age_level','hour']]
# t = t.drop_duplicates(subset=['user_age_level'])
# t1 = t1.drop_duplicates(subset=['user_age_level','hour'])
# t1 = pd.merge(t1,t,on=['user_age_level'],how='left')
# t1['distance'] = abs(t1['hour']-t1['mean_time'])
# train = pd.merge(train,t1,on=['user_age_level','hour'],how='left')
# train[['distance','mean_time','user_age_level','hour']].to_csv('data/other_age_time_feature5.csv',index=None)