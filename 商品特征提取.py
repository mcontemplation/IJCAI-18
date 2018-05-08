import pandas as pd
import xgboost as xgb
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import time
from datetime import date
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 提取训练集一广告商品相关特征
train1_f = pd.read_csv('data/train1_f.csv')
tra = train1_f
# 该展示商品被点击次数
t = tra[['item_id']]
t['item_click']=1
t = t.groupby('item_id').agg('sum').reset_index()
tra = pd.merge(tra,t,on='item_id',how='left')
# 该展示商品被购买次数
t = tra[['item_id','is_trade']]
t = t.groupby('item_id').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'item_buy'})
tra = pd.merge(tra,t,on='item_id',how='left')
# 该展示商品被购买率
tra['item_buy_rate'] = tra['item_buy']/tra['item_click']
# 该展示商品在上午购买的次数
t = tra[['item_id','is_am','is_trade']]
t = t[t.is_trade==1][['item_id','is_am']]
t = t.groupby('item_id').agg('sum').reset_index()
t.rename(columns={'is_am':'sale_on_am'},inplace=True)
tra = pd.merge(tra,t,on='item_id',how='left')
# 该展示商品再下午购买的次数
tra['sale_on_pm'] = tra['item_buy']-tra['sale_on_am']
# 该商品是否属于高销量类别
t = tra[['cate2','is_trade']]
t = tra.groupby('cate2').agg('sum').reset_index()
t['is_high_sale'] = t['is_trade'].apply(lambda x:1 if x>=1000 else 0)
tra = pd.merge(tra,t[['cate2','is_high_sale']],on='cate2',how='left')
#商品的属性列表处理
t = tra[['item_id','item_property_list','is_trade']]
#将属性列表从字符串转变为list
def getPropertySet(s):
     PropertySet=set(s.split(';'))
     return PropertySet
t['item_property_list']=t.item_property_list.apply(getPropertySet)
#获取受欢迎商品的属性
t1=t[t.is_trade==1]
Popular=set()
for i in t1['item_property_list']:
     Popular=set(list(Popular)+list(i))
#取出每种商品
t2=t[['item_id','item_property_list']]
t2.drop_duplicates(['item_id'],inplace=True)
#记录每个商品的属性总数
def setSum_propertys(s):
      return len(s)
t2['Sum_Propertys']=t2.item_property_list.apply(setSum_propertys)
#记录每个商品的受欢迎属性个数
t2['Sum_Popular_Propertys']=0
# t['Sum_Popular_Propertys']=len([i for i in t.item_property_list if i in list(Popular)])
def getSum_Popular_Propertys(s):
     return len(Popular.intersection(s))
t2['Sum_Popular_Propertys']=t2.item_property_list.apply(getSum_Popular_Propertys)
#记录每个商品受欢迎属性占总属性个数百分比
t2['Popular_rate']=t2['Sum_Popular_Propertys']/t2['Sum_Propertys']
#合并
tra=pd.merge(tra,t2[['item_id','Sum_Propertys','Sum_Popular_Propertys','Popular_rate']],on='item_id',how='left')


# 该类目点击次数
t = tra[['cate2']]
t['cate2_click']=1
t = t.groupby('cate2').agg('sum').reset_index()
tra = pd.merge(tra,t,on='cate2',how='left')
# 该类目平均年龄
t = tra[tra.is_trade==1][['cate2','user_age_level']]
t = t.groupby(['cate2']).agg('mean').reset_index()
t = t.rename(columns={'user_age_level':'cmean_age'})
tra = pd.merge(tra,t,on='cate2',how='left')
# 该类目购买次数
t = tra[['cate2','is_trade']]
t = t.groupby('cate2').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'cate2_buy'})
tra = pd.merge(tra,t,on='cate2',how='left')
# 该类目购买率
tra['cate2_buy_rate'] = tra['cate2_buy']/tra['cate2_click']

tra[['cate2_buy_rate','cate2_click','cate2_buy','cmean_age','cate2']].to_csv('data/cate_feature1.csv',index=None)
# 该价格等级点击次数
t = tra[['item_price_level']]
t['price_click']=1
t = t.groupby('item_price_level').agg('sum').reset_index()
tra = pd.merge(tra,t,on='item_price_level',how='left')
# 改价格等级购买次数
t = tra[['item_price_level','is_trade']]
t = t.groupby('item_price_level').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'price_buy'})
tra = pd.merge(tra,t,on='item_price_level',how='left')
# 改价格等级购买率
tra['price_buy_rate'] = tra['price_buy']/tra['price_click']
tra[['price_buy_rate','price_click','price_buy','item_price_level']].to_csv('data/price_feature1.csv',index=None)

# 该销售等级点击数
t = tra[['item_sales_level']]
t['sales_click']=1
t = t.groupby('item_sales_level').agg('sum').reset_index()
tra = pd.merge(tra,t,on='item_sales_level',how='left')
# 该销售等级购买数
t = tra[['item_sales_level','is_trade']]
t = t.groupby('item_sales_level').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'sales_buy'})
tra = pd.merge(tra,t,on='item_sales_level',how='left')
# 该销售等级购买率
tra['sales_buy_rate'] = tra['sales_buy']/tra['sales_click']
tra[['sales_buy_rate','sales_click','sales_buy','item_sales_level']].to_csv('data/sales_feature1.csv',index=None)

# 该收藏等级点击数
t = tra[['item_collected_level']]
t['collected_click']=1
t = t.groupby('item_collected_level').agg('sum').reset_index()
tra = pd.merge(tra,t,on='item_collected_level',how='left')
# 该销售等级购买数
t = tra[['item_collected_level','is_trade']]
t = t.groupby('item_collected_level').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'collected_buy'})
tra = pd.merge(tra,t,on='item_collected_level',how='left')
# 该销售等级购买率
tra['collected_buy_rate'] = tra['collected_buy']/tra['collected_click']
tra[['collected_buy_rate','collected_click','collected_buy','item_collected_level']].to_csv('data/collected_feature1.csv',index=None)

# 该品牌点击次数
t = tra[['item_brand_id']]
t['brand_click']=1
t = t.groupby('item_brand_id').agg('sum').reset_index()
tra = pd.merge(tra,t,on='item_brand_id',how='left')
# 该品牌被购买次数
t = tra[['item_brand_id','is_trade']]
t = t.groupby('item_brand_id').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'brand_buy'})
tra = pd.merge(tra,t,on='item_brand_id',how='left')
# 该品牌购买率
tra['brand_buy_rate'] = tra['brand_buy']/tra['brand_click']
tra[['brand_buy_rate','brand_click','brand_buy','item_brand_id']].to_csv('data/brand_feature1.csv',index=None)

# 该类目在该价格点击次数
t = tra[['item_price_level','cate2']]
t['cate_price_click'] = 1
t = t.groupby(['item_price_level','cate2']).agg('sum').reset_index()
tra = pd.merge(tra,t,on=['item_price_level','cate2'],how='left')
#
t = tra[['item_price_level','cate2','is_trade']]
t = t.groupby(['item_price_level','cate2']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'cate_price_buy'})
tra = pd.merge(tra,t,on=['item_price_level','cate2'],how='left')
#
tra['cate_price_rate'] = tra['cate_price_buy']/tra['cate_price_click']
tra[['cate_price_click','cate_price_buy','cate_price_rate','cate2','item_price_level']].to_csv('data/cate_price_feature1.csv',index=None)

# # 该商品在改价格等级
# t = tra[['item_price_level','item_id']]
# t['item_price_click'] = 1
# t = t.groupby(['item_price_level','item_id']).agg('sum').reset_index()
# tra = pd.merge(tra,t,on=['item_price_level','item_id'],how='left')
# #
# t = tra[['item_price_level','item_id','is_trade']]
# t = t.groupby(['item_price_level','item_id']).agg('sum').reset_index()
# t = t.rename(columns={'is_trade':'item_price_buy'})
# tra = pd.merge(tra,t,on=['item_price_level','item_id'],how='left')
# #
# tra['item_price_rate'] = tra['item_price_buy']/tra['item_price_click']
# tra[['item_price_click','item_price_buy','item_price_rate','item_price_level','item_id']].to_csv('data/item_price_feature1.csv',index=None)

# 该商品占改价格等级购买比
tra['item_price_rate'] = tra['item_buy']/tra['price_buy']
# 类目
tra['item_cate_rate'] = tra['item_buy']/tra['cate2_buy']

tra[['item_id','sale_on_pm','item_click','item_buy','item_buy_rate','sale_on_am','is_high_sale','Sum_Propertys','Sum_Popular_Propertys',
     'Popular_rate','item_price_rate','item_cate_rate',
     ]].to_csv('data/item_feature1.csv',index=None)






# 提取训练集一广告商品相关特征
train1_f = pd.read_csv('data/train2_f.csv')
tra = train1_f
# 该展示商品被点击次数
t = tra[['item_id']]
t['item_click']=1
t = t.groupby('item_id').agg('sum').reset_index()
tra = pd.merge(tra,t,on='item_id',how='left')
# 该展示商品被购买次数
t = tra[['item_id','is_trade']]
t = t.groupby('item_id').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'item_buy'})
tra = pd.merge(tra,t,on='item_id',how='left')
# 该展示商品被购买率
tra['item_buy_rate'] = tra['item_buy']/tra['item_click']
# 该展示商品在上午购买的次数
t = tra[['item_id','is_am','is_trade']]
t = t[t.is_trade==1][['item_id','is_am']]
t = t.groupby('item_id').agg('sum').reset_index()
t.rename(columns={'is_am':'sale_on_am'},inplace=True)
tra = pd.merge(tra,t,on='item_id',how='left')
# 该展示商品再下午购买的次数
tra['sale_on_pm'] = tra['item_buy']-tra['sale_on_am']
# 该商品是否属于高销量类别
t = tra[['cate2','is_trade']]
t = tra.groupby('cate2').agg('sum').reset_index()
t['is_high_sale'] = t['is_trade'].apply(lambda x:1 if x>=1000 else 0)
tra = pd.merge(tra,t[['cate2','is_high_sale']],on='cate2',how='left')
#商品的属性列表处理
t = tra[['item_id','item_property_list','is_trade']]
#将属性列表从字符串转变为list
def getPropertySet(s):
     PropertySet=set(s.split(';'))
     return PropertySet
t['item_property_list']=t.item_property_list.apply(getPropertySet)
#获取受欢迎商品的属性
t1=t[t.is_trade==1]
Popular=set()
for i in t1['item_property_list']:
     Popular=set(list(Popular)+list(i))
#取出每种商品
t2=t[['item_id','item_property_list']]
t2.drop_duplicates(['item_id'],inplace=True)
#记录每个商品的属性总数
def setSum_propertys(s):
      return len(s)
t2['Sum_Propertys']=t2.item_property_list.apply(setSum_propertys)
#记录每个商品的受欢迎属性个数
t2['Sum_Popular_Propertys']=0
# t['Sum_Popular_Propertys']=len([i for i in t.item_property_list if i in list(Popular)])
def getSum_Popular_Propertys(s):
     return len(Popular.intersection(s))
t2['Sum_Popular_Propertys']=t2.item_property_list.apply(getSum_Popular_Propertys)
#记录每个商品受欢迎属性占总属性个数百分比
t2['Popular_rate']=t2['Sum_Popular_Propertys']/t2['Sum_Propertys']
#合并
tra=pd.merge(tra,t2[['item_id','Sum_Propertys','Sum_Popular_Propertys','Popular_rate']],on='item_id',how='left')



# 该类目点击次数
t = tra[['cate2']]
t['cate2_click']=1
t = t.groupby('cate2').agg('sum').reset_index()
tra = pd.merge(tra,t,on='cate2',how='left')
# 该类目购买次数
t = tra[['cate2','is_trade']]
t = t.groupby('cate2').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'cate2_buy'})
tra = pd.merge(tra,t,on='cate2',how='left')
# 该类目购买率
tra['cate2_buy_rate'] = tra['cate2_buy']/tra['cate2_click']
# 该类目平均年龄
t = tra[tra.is_trade==1][['cate2','user_age_level']]
t = t.groupby(['cate2']).agg('mean').reset_index()
t = t.rename(columns={'user_age_level':'cmean_age'})
tra = pd.merge(tra,t,on='cate2',how='left')

tra[['cate2_buy_rate','cate2_click','cate2_buy','cmean_age','cate2']].to_csv('data/cate_feature2.csv',index=None)

# 该价格等级点击次数
t = tra[['item_price_level']]
t['price_click']=1
t = t.groupby('item_price_level').agg('sum').reset_index()
tra = pd.merge(tra,t,on='item_price_level',how='left')
# 改价格等级购买次数
t = tra[['item_price_level','is_trade']]
t = t.groupby('item_price_level').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'price_buy'})
tra = pd.merge(tra,t,on='item_price_level',how='left')
# 改价格等级购买率
tra['price_buy_rate'] = tra['price_buy']/tra['price_click']
tra[['price_buy_rate','price_click','price_buy','item_price_level']].to_csv('data/price_feature2.csv',index=None)

# 该销售等级点击数
t = tra[['item_sales_level']]
t['sales_click']=1
t = t.groupby('item_sales_level').agg('sum').reset_index()
tra = pd.merge(tra,t,on='item_sales_level',how='left')
# 该销售等级购买数
t = tra[['item_sales_level','is_trade']]
t = t.groupby('item_sales_level').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'sales_buy'})
tra = pd.merge(tra,t,on='item_sales_level',how='left')
# 该销售等级购买率
tra['sales_buy_rate'] = tra['sales_buy']/tra['sales_click']
tra[['sales_buy_rate','sales_click','sales_buy','item_sales_level']].to_csv('data/sales_feature2.csv',index=None)

# 该收藏等级点击数
t = tra[['item_collected_level']]
t['collected_click']=1
t = t.groupby('item_collected_level').agg('sum').reset_index()
tra = pd.merge(tra,t,on='item_collected_level',how='left')
# 该销售等级购买数
t = tra[['item_collected_level','is_trade']]
t = t.groupby('item_collected_level').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'collected_buy'})
tra = pd.merge(tra,t,on='item_collected_level',how='left')
# 该销售等级购买率
tra['collected_buy_rate'] = tra['collected_buy']/tra['collected_click']
tra[['collected_buy_rate','collected_click','collected_buy','item_collected_level']].to_csv('data/collected_feature2.csv',index=None)

# 该品牌点击次数
t = tra[['item_brand_id']]
t['brand_click']=1
t = t.groupby('item_brand_id').agg('sum').reset_index()
tra = pd.merge(tra,t,on='item_brand_id',how='left')
# 该品牌被购买次数
t = tra[['item_brand_id','is_trade']]
t = t.groupby('item_brand_id').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'brand_buy'})
tra = pd.merge(tra,t,on='item_brand_id',how='left')
# 该品牌购买率
tra['brand_buy_rate'] = tra['brand_buy']/tra['brand_click']
tra[['brand_buy_rate','brand_click','brand_buy','item_brand_id']].to_csv('data/brand_feature2.csv',index=None)

# 该类目在该价格点击次数
t = tra[['item_price_level','cate2']]
t['cate_price_click'] = 1
t = t.groupby(['item_price_level','cate2']).agg('sum').reset_index()
tra = pd.merge(tra,t,on=['item_price_level','cate2'],how='left')
#
t = tra[['item_price_level','cate2','is_trade']]
t = t.groupby(['item_price_level','cate2']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'cate_price_buy'})
tra = pd.merge(tra,t,on=['item_price_level','cate2'],how='left')
#
tra['cate_price_rate'] = tra['cate_price_buy']/tra['cate_price_click']
tra[['cate_price_click','cate_price_buy','cate_price_rate','cate2','item_price_level']].to_csv('data/cate_price_feature2.csv',index=None)

# # 该商品在改价格等级
# t = tra[['item_price_level','item_id']]
# t['item_price_click'] = 1
# t = t.groupby(['item_price_level','item_id']).agg('sum').reset_index()
# tra = pd.merge(tra,t,on=['item_price_level','item_id'],how='left')
# #
# t = tra[['item_price_level','item_id','is_trade']]
# t = t.groupby(['item_price_level','item_id']).agg('sum').reset_index()
# t = t.rename(columns={'is_trade':'item_price_buy'})
# tra = pd.merge(tra,t,on=['item_price_level','item_id'],how='left')
# #
# tra['item_price_rate'] = tra['item_price_buy']/tra['item_price_click']
# tra[['item_price_click','item_price_buy','item_price_rate','item_price_level','item_id']].to_csv('data/item_price_feature2.csv',index=None)

# 该商品占改价格等级购买比
tra['item_price_rate'] = tra['item_buy']/tra['price_buy']
# 类目
tra['item_cate_rate'] = tra['item_buy']/tra['cate2_buy']

tra[['item_id','sale_on_pm','item_click','item_buy','item_buy_rate','sale_on_am','is_high_sale','Sum_Propertys','Sum_Popular_Propertys',
     'Popular_rate','item_price_rate','item_cate_rate',
     ]].to_csv('data/item_feature2.csv',index=None)


# 提取训练集一广告商品相关特征
train1_f = pd.read_csv('data/train3_f.csv')
tra = train1_f
# 该展示商品被点击次数
t = tra[['item_id']]
t['item_click']=1
t = t.groupby('item_id').agg('sum').reset_index()
tra = pd.merge(tra,t,on='item_id',how='left')
# 该展示商品被购买次数
t = tra[['item_id','is_trade']]
t = t.groupby('item_id').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'item_buy'})
tra = pd.merge(tra,t,on='item_id',how='left')
# 该展示商品被购买率
tra['item_buy_rate'] = tra['item_buy']/tra['item_click']
# 该展示商品在上午购买的次数
t = tra[['item_id','is_am','is_trade']]
t = t[t.is_trade==1][['item_id','is_am']]
t = t.groupby('item_id').agg('sum').reset_index()
t.rename(columns={'is_am':'sale_on_am'},inplace=True)
tra = pd.merge(tra,t,on='item_id',how='left')
# 该展示商品再下午购买的次数
tra['sale_on_pm'] = tra['item_buy']-tra['sale_on_am']
# 该商品是否属于高销量类别
t = tra[['cate2','is_trade']]
t = tra.groupby('cate2').agg('sum').reset_index()
t['is_high_sale'] = t['is_trade'].apply(lambda x:1 if x>=1000 else 0)
tra = pd.merge(tra,t[['cate2','is_high_sale']],on='cate2',how='left')
#商品的属性列表处理
t = tra[['item_id','item_property_list','is_trade']]
#将属性列表从字符串转变为list
def getPropertySet(s):
     PropertySet=set(s.split(';'))
     return PropertySet
t['item_property_list']=t.item_property_list.apply(getPropertySet)
#获取受欢迎商品的属性
t1=t[t.is_trade==1]
Popular=set()
for i in t1['item_property_list']:
     Popular=set(list(Popular)+list(i))
#取出每种商品
t2=t[['item_id','item_property_list']]
t2.drop_duplicates(['item_id'],inplace=True)
#记录每个商品的属性总数
def setSum_propertys(s):
      return len(s)
t2['Sum_Propertys']=t2.item_property_list.apply(setSum_propertys)
#记录每个商品的受欢迎属性个数
t2['Sum_Popular_Propertys']=0
# t['Sum_Popular_Propertys']=len([i for i in t.item_property_list if i in list(Popular)])
def getSum_Popular_Propertys(s):
     return len(Popular.intersection(s))
t2['Sum_Popular_Propertys']=t2.item_property_list.apply(getSum_Popular_Propertys)
#记录每个商品受欢迎属性占总属性个数百分比
t2['Popular_rate']=t2['Sum_Popular_Propertys']/t2['Sum_Propertys']
#合并
tra=pd.merge(tra,t2[['item_id','Sum_Propertys','Sum_Popular_Propertys','Popular_rate']],on='item_id',how='left')

tra[['item_id','sale_on_pm','item_click','item_buy','item_buy_rate','sale_on_am','is_high_sale','Sum_Propertys','Sum_Popular_Propertys',
     'Popular_rate'
     ]].to_csv('data/item_feature3.csv',index=None)

# 该类目点击次数
t = tra[['cate2']]
t['cate2_click']=1
t = t.groupby('cate2').agg('sum').reset_index()
tra = pd.merge(tra,t,on='cate2',how='left')
# 该类目购买次数
t = tra[['cate2','is_trade']]
t = t.groupby('cate2').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'cate2_buy'})
tra = pd.merge(tra,t,on='cate2',how='left')
# 该类目购买率
tra['cate2_buy_rate'] = tra['cate2_buy']/tra['cate2_click']
# 该类目平均年龄
t = tra[tra.is_trade==1][['cate2','user_age_level']]
t = t.groupby(['cate2']).agg('mean').reset_index()
t = t.rename(columns={'user_age_level':'cmean_age'})
tra = pd.merge(tra,t,on='cate2',how='left')

tra[['cate2_buy_rate','cate2_click','cate2_buy','cmean_age','cate2']].to_csv('data/cate_feature3.csv',index=None)

# 该价格等级点击次数
t = tra[['item_price_level']]
t['price_click']=1
t = t.groupby('item_price_level').agg('sum').reset_index()
tra = pd.merge(tra,t,on='item_price_level',how='left')
# 改价格等级购买次数
t = tra[['item_price_level','is_trade']]
t = t.groupby('item_price_level').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'price_buy'})
tra = pd.merge(tra,t,on='item_price_level',how='left')
# 改价格等级购买率
tra['price_buy_rate'] = tra['price_buy']/tra['price_click']
tra[['price_buy_rate','price_click','price_buy','item_price_level']].to_csv('data/price_feature3.csv',index=None)

# 该销售等级点击数
t = tra[['item_sales_level']]
t['sales_click']=1
t = t.groupby('item_sales_level').agg('sum').reset_index()
tra = pd.merge(tra,t,on='item_sales_level',how='left')
# 该销售等级购买数
t = tra[['item_sales_level','is_trade']]
t = t.groupby('item_sales_level').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'sales_buy'})
tra = pd.merge(tra,t,on='item_sales_level',how='left')
# 该销售等级购买率
tra['sales_buy_rate'] = tra['sales_buy']/tra['sales_click']
tra[['sales_buy_rate','sales_click','sales_buy','item_sales_level']].to_csv('data/sales_feature3.csv',index=None)
# 该收藏等级点击数
t = tra[['item_collected_level']]
t['collected_click']=1
t = t.groupby('item_collected_level').agg('sum').reset_index()
tra = pd.merge(tra,t,on='item_collected_level',how='left')
# 该销售等级购买数
t = tra[['item_collected_level','is_trade']]
t = t.groupby('item_collected_level').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'collected_buy'})
tra = pd.merge(tra,t,on='item_collected_level',how='left')
# 该销售等级购买率
tra['collected_buy_rate'] = tra['collected_buy']/tra['collected_click']
tra[['collected_buy_rate','collected_click','collected_buy','item_collected_level']].to_csv('data/collected_feature3.csv',index=None)

# 该品牌点击次数
t = tra[['item_brand_id']]
t['brand_click']=1
t = t.groupby('item_brand_id').agg('sum').reset_index()
tra = pd.merge(tra,t,on='item_brand_id',how='left')
# 该品牌被购买次数
t = tra[['item_brand_id','is_trade']]
t = t.groupby('item_brand_id').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'brand_buy'})
tra = pd.merge(tra,t,on='item_brand_id',how='left')
# 该品牌购买率
tra['brand_buy_rate'] = tra['brand_buy']/tra['brand_click']
tra[['brand_buy_rate','brand_click','brand_buy','item_brand_id']].to_csv('data/brand_feature3.csv',index=None)


# 该类目在该价格点击次数
t = tra[['item_price_level','cate2']]
t['cate_price_click'] = 1
t = t.groupby(['item_price_level','cate2']).agg('sum').reset_index()
tra = pd.merge(tra,t,on=['item_price_level','cate2'],how='left')
#
t = tra[['item_price_level','cate2','is_trade']]
t = t.groupby(['item_price_level','cate2']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'cate_price_buy'})
tra = pd.merge(tra,t,on=['item_price_level','cate2'],how='left')
#
tra['cate_price_rate'] = tra['cate_price_buy']/tra['cate_price_click']
tra[['cate_price_click','cate_price_buy','cate_price_rate','cate2','item_price_level']].to_csv('data/cate_price_feature3.csv',index=None)

# # 该商品在改价格等级
# t = tra[['item_price_level','item_id']]
# t['item_price_click'] = 1
# t = t.groupby(['item_price_level','item_id']).agg('sum').reset_index()
# tra = pd.merge(tra,t,on=['item_price_level','item_id'],how='left')
# #
# t = tra[['item_price_level','item_id','is_trade']]
# t = t.groupby(['item_price_level','item_id']).agg('sum').reset_index()
# t = t.rename(columns={'is_trade':'item_price_buy'})
# tra = pd.merge(tra,t,on=['item_price_level','item_id'],how='left')
# #
# tra['item_price_rate'] = tra['item_price_buy']/tra['item_price_click']
# tra[['item_price_click','item_price_buy','item_price_rate','item_price_level','item_id']].to_csv('data/item_price_feature3.csv',index=None)
# 该商品占改价格等级购买比
tra['item_price_rate'] = tra['item_buy']/tra['price_buy']
# 类目
tra['item_cate_rate'] = tra['item_buy']/tra['cate2_buy']

tra[['item_id','sale_on_pm','item_click','item_buy','item_buy_rate','sale_on_am','is_high_sale','Sum_Propertys','Sum_Popular_Propertys',
     'Popular_rate','item_price_rate','item_cate_rate',
     ]].to_csv('data/item_feature3.csv',index=None)







# 提取训练集一广告商品相关特征
train1_f = pd.read_csv('data/train4_f.csv')
tra = train1_f
# 该展示商品被点击次数
t = tra[['item_id']]
t['item_click']=1
t = t.groupby('item_id').agg('sum').reset_index()
tra = pd.merge(tra,t,on='item_id',how='left')
# 该展示商品被购买次数
t = tra[['item_id','is_trade']]
t = t.groupby('item_id').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'item_buy'})
tra = pd.merge(tra,t,on='item_id',how='left')
# 该展示商品被购买率
tra['item_buy_rate'] = tra['item_buy']/tra['item_click']
# 该展示商品在上午购买的次数
t = tra[['item_id','is_am','is_trade']]
t = t[t.is_trade==1][['item_id','is_am']]
t = t.groupby('item_id').agg('sum').reset_index()
t.rename(columns={'is_am':'sale_on_am'},inplace=True)
tra = pd.merge(tra,t,on='item_id',how='left')
# 该展示商品再下午购买的次数
tra['sale_on_pm'] = tra['item_buy']-tra['sale_on_am']
# 该商品是否属于高销量类别
t = tra[['cate2','is_trade']]
t = tra.groupby('cate2').agg('sum').reset_index()
t['is_high_sale'] = t['is_trade'].apply(lambda x:1 if x>=1000 else 0)
tra = pd.merge(tra,t[['cate2','is_high_sale']],on='cate2',how='left')
#商品的属性列表处理
t = tra[['item_id','item_property_list','is_trade']]
#将属性列表从字符串转变为list
def getPropertySet(s):
     PropertySet=set(s.split(';'))
     return PropertySet
t['item_property_list']=t.item_property_list.apply(getPropertySet)
#获取受欢迎商品的属性
t1=t[t.is_trade==1]
Popular=set()
for i in t1['item_property_list']:
     Popular=set(list(Popular)+list(i))
#取出每种商品
t2=t[['item_id','item_property_list']]
t2.drop_duplicates(['item_id'],inplace=True)
#记录每个商品的属性总数
def setSum_propertys(s):
      return len(s)
t2['Sum_Propertys']=t2.item_property_list.apply(setSum_propertys)
#记录每个商品的受欢迎属性个数
t2['Sum_Popular_Propertys']=0
# t['Sum_Popular_Propertys']=len([i for i in t.item_property_list if i in list(Popular)])
def getSum_Popular_Propertys(s):
     return len(Popular.intersection(s))
t2['Sum_Popular_Propertys']=t2.item_property_list.apply(getSum_Popular_Propertys)
#记录每个商品受欢迎属性占总属性个数百分比
t2['Popular_rate']=t2['Sum_Popular_Propertys']/t2['Sum_Propertys']
#合并
tra=pd.merge(tra,t2[['item_id','Sum_Propertys','Sum_Popular_Propertys','Popular_rate']],on='item_id',how='left')

tra[['item_id','sale_on_pm','item_click','item_buy','item_buy_rate','sale_on_am','is_high_sale','Sum_Propertys','Sum_Popular_Propertys',
     'Popular_rate'
     ]].to_csv('data/item_feature4.csv',index=None)

# 该类目点击次数
t = tra[['cate2']]
t['cate2_click']=1
t = t.groupby('cate2').agg('sum').reset_index()
tra = pd.merge(tra,t,on='cate2',how='left')
# 该类目购买次数
t = tra[['cate2','is_trade']]
t = t.groupby('cate2').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'cate2_buy'})
tra = pd.merge(tra,t,on='cate2',how='left')
# 该类目购买率
tra['cate2_buy_rate'] = tra['cate2_buy']/tra['cate2_click']
# 该类目平均年龄
t = tra[tra.is_trade==1][['cate2','user_age_level']]
t = t.groupby(['cate2']).agg('mean').reset_index()
t = t.rename(columns={'user_age_level':'cmean_age'})
tra = pd.merge(tra,t,on='cate2',how='left')

tra[['cate2_buy_rate','cate2_click','cate2_buy','cmean_age','cate2']].to_csv('data/cate_feature4.csv',index=None)

# 该价格等级点击次数
t = tra[['item_price_level']]
t['price_click']=1
t = t.groupby('item_price_level').agg('sum').reset_index()
tra = pd.merge(tra,t,on='item_price_level',how='left')
# 改价格等级购买次数
t = tra[['item_price_level','is_trade']]
t = t.groupby('item_price_level').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'price_buy'})
tra = pd.merge(tra,t,on='item_price_level',how='left')
# 改价格等级购买率
tra['price_buy_rate'] = tra['price_buy']/tra['price_click']
tra[['price_buy_rate','price_click','price_buy','item_price_level']].to_csv('data/price_feature4.csv',index=None)

# 该销售等级点击数
t = tra[['item_sales_level']]
t['sales_click']=1
t = t.groupby('item_sales_level').agg('sum').reset_index()
tra = pd.merge(tra,t,on='item_sales_level',how='left')
# 该销售等级购买数
t = tra[['item_sales_level','is_trade']]
t = t.groupby('item_sales_level').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'sales_buy'})
tra = pd.merge(tra,t,on='item_sales_level',how='left')
# 该销售等级购买率
tra['sales_buy_rate'] = tra['sales_buy']/tra['sales_click']
tra[['sales_buy_rate','sales_click','sales_buy','item_sales_level']].to_csv('data/sales_feature4.csv',index=None)
# 该收藏等级点击数
t = tra[['item_collected_level']]
t['collected_click']=1
t = t.groupby('item_collected_level').agg('sum').reset_index()
tra = pd.merge(tra,t,on='item_collected_level',how='left')
# 该销售等级购买数
t = tra[['item_collected_level','is_trade']]
t = t.groupby('item_collected_level').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'collected_buy'})
tra = pd.merge(tra,t,on='item_collected_level',how='left')
# 该销售等级购买率
tra['collected_buy_rate'] = tra['collected_buy']/tra['collected_click']
tra[['collected_buy_rate','collected_click','collected_buy','item_collected_level']].to_csv('data/collected_feature4.csv',index=None)

# 该品牌点击次数
t = tra[['item_brand_id']]
t['brand_click']=1
t = t.groupby('item_brand_id').agg('sum').reset_index()
tra = pd.merge(tra,t,on='item_brand_id',how='left')
# 该品牌被购买次数
t = tra[['item_brand_id','is_trade']]
t = t.groupby('item_brand_id').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'brand_buy'})
tra = pd.merge(tra,t,on='item_brand_id',how='left')
# 该品牌购买率
tra['brand_buy_rate'] = tra['brand_buy']/tra['brand_click']
tra[['brand_buy_rate','brand_click','brand_buy','item_brand_id']].to_csv('data/brand_feature4.csv',index=None)


# 该类目在该价格点击次数
t = tra[['item_price_level','cate2']]
t['cate_price_click'] = 1
t = t.groupby(['item_price_level','cate2']).agg('sum').reset_index()
tra = pd.merge(tra,t,on=['item_price_level','cate2'],how='left')
#
t = tra[['item_price_level','cate2','is_trade']]
t = t.groupby(['item_price_level','cate2']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'cate_price_buy'})
tra = pd.merge(tra,t,on=['item_price_level','cate2'],how='left')
#
tra['cate_price_rate'] = tra['cate_price_buy']/tra['cate_price_click']
tra[['cate_price_click','cate_price_buy','cate_price_rate','cate2','item_price_level']].to_csv('data/cate_price_feature4.csv',index=None)

# # 该商品在改价格等级
# t = tra[['item_price_level','item_id']]
# t['item_price_click'] = 1
# t = t.groupby(['item_price_level','item_id']).agg('sum').reset_index()
# tra = pd.merge(tra,t,on=['item_price_level','item_id'],how='left')
# #
# t = tra[['item_price_level','item_id','is_trade']]
# t = t.groupby(['item_price_level','item_id']).agg('sum').reset_index()
# t = t.rename(columns={'is_trade':'item_price_buy'})
# tra = pd.merge(tra,t,on=['item_price_level','item_id'],how='left')
# #
# tra['item_price_rate'] = tra['item_price_buy']/tra['item_price_click']
# tra[['item_price_click','item_price_buy','item_price_rate','item_price_level','item_id']].to_csv('data/item_price_feature3.csv',index=None)
# 该商品占改价格等级购买比
tra['item_price_rate'] = tra['item_buy']/tra['price_buy']
# 类目
tra['item_cate_rate'] = tra['item_buy']/tra['cate2_buy']

tra[['item_id','sale_on_pm','item_click','item_buy','item_buy_rate','sale_on_am','is_high_sale','Sum_Propertys','Sum_Popular_Propertys',
     'Popular_rate','item_price_rate','item_cate_rate',
     ]].to_csv('data/item_feature4.csv',index=None)





# # 提取训练集一广告商品相关特征
# train1_f = pd.read_csv('data/train5_f.csv')
# tra = train1_f
# # 该展示商品被点击次数
# t = tra[['item_id']]
# t['item_click']=1
# t = t.groupby('item_id').agg('sum').reset_index()
# tra = pd.merge(tra,t,on='item_id',how='left')
# # 该展示商品被购买次数
# t = tra[['item_id','is_trade']]
# t = t.groupby('item_id').agg('sum').reset_index()
# t = t.rename(columns={'is_trade':'item_buy'})
# tra = pd.merge(tra,t,on='item_id',how='left')
# # 该展示商品被购买率
# tra['item_buy_rate'] = tra['item_buy']/tra['item_click']
# # 该展示商品在上午购买的次数
# t = tra[['item_id','is_am','is_trade']]
# t = t[t.is_trade==1][['item_id','is_am']]
# t = t.groupby('item_id').agg('sum').reset_index()
# t.rename(columns={'is_am':'sale_on_am'},inplace=True)
# tra = pd.merge(tra,t,on='item_id',how='left')
# # 该展示商品再下午购买的次数
# tra['sale_on_pm'] = tra['item_buy']-tra['sale_on_am']
# # 该商品是否属于高销量类别
# t = tra[['cate2','is_trade']]
# t = tra.groupby('cate2').agg('sum').reset_index()
# t['is_high_sale'] = t['is_trade'].apply(lambda x:1 if x>=1000 else 0)
# tra = pd.merge(tra,t[['cate2','is_high_sale']],on='cate2',how='left')
# #商品的属性列表处理
# t = tra[['item_id','item_property_list','is_trade']]
# #将属性列表从字符串转变为list
# def getPropertySet(s):
#      PropertySet=set(s.split(';'))
#      return PropertySet
# t['item_property_list']=t.item_property_list.apply(getPropertySet)
# #获取受欢迎商品的属性
# t1=t[t.is_trade==1]
# Popular=set()
# for i in t1['item_property_list']:
#      Popular=set(list(Popular)+list(i))
# #取出每种商品
# t2=t[['item_id','item_property_list']]
# t2.drop_duplicates(['item_id'],inplace=True)
# #记录每个商品的属性总数
# def setSum_propertys(s):
#       return len(s)
# t2['Sum_Propertys']=t2.item_property_list.apply(setSum_propertys)
# #记录每个商品的受欢迎属性个数
# t2['Sum_Popular_Propertys']=0
# # t['Sum_Popular_Propertys']=len([i for i in t.item_property_list if i in list(Popular)])
# def getSum_Popular_Propertys(s):
#      return len(Popular.intersection(s))
# t2['Sum_Popular_Propertys']=t2.item_property_list.apply(getSum_Popular_Propertys)
# #记录每个商品受欢迎属性占总属性个数百分比
# t2['Popular_rate']=t2['Sum_Popular_Propertys']/t2['Sum_Propertys']
# #合并
# tra=pd.merge(tra,t2[['item_id','Sum_Propertys','Sum_Popular_Propertys','Popular_rate']],on='item_id',how='left')
#
# tra[['item_id','sale_on_pm','item_click','item_buy','item_buy_rate','sale_on_am','is_high_sale','Sum_Propertys','Sum_Popular_Propertys',
#      'Popular_rate'
#      ]].to_csv('data/item_feature5.csv',index=None)
#
# # 该类目点击次数
# t = tra[['cate2']]
# t['cate2_click']=1
# t = t.groupby('cate2').agg('sum').reset_index()
# tra = pd.merge(tra,t,on='cate2',how='left')
# # 该类目购买次数
# t = tra[['cate2','is_trade']]
# t = t.groupby('cate2').agg('sum').reset_index()
# t = t.rename(columns={'is_trade':'cate2_buy'})
# tra = pd.merge(tra,t,on='cate2',how='left')
# # 该类目购买率
# tra['cate2_buy_rate'] = tra['cate2_buy']/tra['cate2_click']
# # 该类目平均年龄
# t = tra[tra.is_trade==1][['cate2','user_age_level']]
# t = t.groupby(['cate2']).agg('mean').reset_index()
# t = t.rename(columns={'user_age_level':'cmean_age'})
# tra = pd.merge(tra,t,on='cate2',how='left')
#
# tra[['cate2_buy_rate','cate2_click','cate2_buy','cmean_age','cate2']].to_csv('data/cate_feature5.csv',index=None)
#
# # 该价格等级点击次数
# t = tra[['item_price_level']]
# t['price_click']=1
# t = t.groupby('item_price_level').agg('sum').reset_index()
# tra = pd.merge(tra,t,on='item_price_level',how='left')
# # 改价格等级购买次数
# t = tra[['item_price_level','is_trade']]
# t = t.groupby('item_price_level').agg('sum').reset_index()
# t = t.rename(columns={'is_trade':'price_buy'})
# tra = pd.merge(tra,t,on='item_price_level',how='left')
# # 改价格等级购买率
# tra['price_buy_rate'] = tra['price_buy']/tra['price_click']
# tra[['price_buy_rate','price_click','price_buy','item_price_level']].to_csv('data/price_feature5.csv',index=None)
#
# # 该销售等级点击数
# t = tra[['item_sales_level']]
# t['sales_click']=1
# t = t.groupby('item_sales_level').agg('sum').reset_index()
# tra = pd.merge(tra,t,on='item_sales_level',how='left')
# # 该销售等级购买数
# t = tra[['item_sales_level','is_trade']]
# t = t.groupby('item_sales_level').agg('sum').reset_index()
# t = t.rename(columns={'is_trade':'sales_buy'})
# tra = pd.merge(tra,t,on='item_sales_level',how='left')
# # 该销售等级购买率
# tra['sales_buy_rate'] = tra['sales_buy']/tra['sales_click']
# tra[['sales_buy_rate','sales_click','sales_buy','item_sales_level']].to_csv('data/sales_feature5.csv',index=None)
# # 该收藏等级点击数
# t = tra[['item_collected_level']]
# t['collected_click']=1
# t = t.groupby('item_collected_level').agg('sum').reset_index()
# tra = pd.merge(tra,t,on='item_collected_level',how='left')
# # 该销售等级购买数
# t = tra[['item_collected_level','is_trade']]
# t = t.groupby('item_collected_level').agg('sum').reset_index()
# t = t.rename(columns={'is_trade':'collected_buy'})
# tra = pd.merge(tra,t,on='item_collected_level',how='left')
# # 该销售等级购买率
# tra['collected_buy_rate'] = tra['collected_buy']/tra['collected_click']
# tra[['collected_buy_rate','collected_click','collected_buy','item_collected_level']].to_csv('data/collected_feature5.csv',index=None)
#
# # 该品牌点击次数
# t = tra[['item_brand_id']]
# t['brand_click']=1
# t = t.groupby('item_brand_id').agg('sum').reset_index()
# tra = pd.merge(tra,t,on='item_brand_id',how='left')
# # 该品牌被购买次数
# t = tra[['item_brand_id','is_trade']]
# t = t.groupby('item_brand_id').agg('sum').reset_index()
# t = t.rename(columns={'is_trade':'brand_buy'})
# tra = pd.merge(tra,t,on='item_brand_id',how='left')
# # 该品牌购买率
# tra['brand_buy_rate'] = tra['brand_buy']/tra['brand_click']
# tra[['brand_buy_rate','brand_click','brand_buy','item_brand_id']].to_csv('data/brand_feature5.csv',index=None)
#
#
# # 该类目在该价格点击次数
# t = tra[['item_price_level','cate2']]
# t['cate_price_click'] = 1
# t = t.groupby(['item_price_level','cate2']).agg('sum').reset_index()
# tra = pd.merge(tra,t,on=['item_price_level','cate2'],how='left')
# #
# t = tra[['item_price_level','cate2','is_trade']]
# t = t.groupby(['item_price_level','cate2']).agg('sum').reset_index()
# t = t.rename(columns={'is_trade':'cate_price_buy'})
# tra = pd.merge(tra,t,on=['item_price_level','cate2'],how='left')
# #
# tra['cate_price_rate'] = tra['cate_price_buy']/tra['cate_price_click']
# tra[['cate_price_click','cate_price_buy','cate_price_rate','cate2','item_price_level']].to_csv('data/cate_price_feature5.csv',index=None)
#
# # # 该商品在改价格等级
# # t = tra[['item_price_level','item_id']]
# # t['item_price_click'] = 1
# # t = t.groupby(['item_price_level','item_id']).agg('sum').reset_index()
# # tra = pd.merge(tra,t,on=['item_price_level','item_id'],how='left')
# # #
# # t = tra[['item_price_level','item_id','is_trade']]
# # t = t.groupby(['item_price_level','item_id']).agg('sum').reset_index()
# # t = t.rename(columns={'is_trade':'item_price_buy'})
# # tra = pd.merge(tra,t,on=['item_price_level','item_id'],how='left')
# # #
# # tra['item_price_rate'] = tra['item_price_buy']/tra['item_price_click']
# # tra[['item_price_click','item_price_buy','item_price_rate','item_price_level','item_id']].to_csv('data/item_price_feature3.csv',index=None)
# # 该商品占改价格等级购买比
# tra['item_price_rate'] = tra['item_buy']/tra['price_buy']
# # 类目
# tra['item_cate_rate'] = tra['item_buy']/tra['cate2_buy']
#
# tra[['item_id','sale_on_pm','item_click','item_buy','item_buy_rate','sale_on_am','is_high_sale','Sum_Propertys','Sum_Popular_Propertys',
#      'Popular_rate','item_price_rate','item_cate_rate',
#      ]].to_csv('data/item_feature5.csv',index=None)