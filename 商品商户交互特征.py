import pandas as pd
import xgboost as xgb
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import time
from datetime import date
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 从训练集一中获取商品店铺相关的特征
train1_f = pd.read_csv('data/train1_f.csv')
item = pd.read_csv('data/item_feature1.csv')
shop = pd.read_csv('data/shop_feature1.csv')
item = item.drop_duplicates()
shop = shop.drop_duplicates()
train1_f = pd.merge(train1_f,item[['item_buy','item_id']],on='item_id',how='left')
train1_f = pd.merge(train1_f,shop[['shop_click_buy_total','shop_id']],on='shop_id',how='left')
commodity_shop = train1_f
# 该商品在该店铺点击次数
t = commodity_shop[['item_id','shop_id']]
t['item_shop_click_total'] = 1
t = t.groupby(['item_id','shop_id']).agg(sum).reset_index()
commodity_shop = pd.merge(commodity_shop,t,on=['item_id','shop_id'],how='left')
# 该商品在该店铺购买次数
t = commodity_shop[['item_id','shop_id','is_trade']]
t = t.groupby(['item_id','shop_id']).agg(sum).reset_index()
t.rename(columns={'is_trade':'item_shop_click_buy_total'},inplace=True)
commodity_shop = pd.merge(commodity_shop,t,on=['item_id','shop_id'],how='left')
# 该商品在该店铺购买率
commodity_shop['item_shop_click_buy_rate'] = commodity_shop['item_shop_click_buy_total']/commodity_shop['item_shop_click_total']
# 该商品在该店铺售卖数量/该商品总售卖量
commodity_shop['item_shop_sale_rate'] = commodity_shop['item_shop_click_buy_total']/commodity_shop['item_buy']
# 该商品在该店铺售卖数量/该商户总售卖量
commodity_shop['shop_item_sale_rate'] = commodity_shop['item_shop_click_buy_total']/commodity_shop['shop_click_buy_total']

commodity_shop[['item_shop_click_buy_rate','item_shop_click_buy_total','item_shop_click_total','item_shop_sale_rate','shop_item_sale_rate','shop_id','item_id']].to_csv('data/shop_item_feature1.csv',index=None)



# 从训练集二中获取商品店铺相关的特征
train1_f = pd.read_csv('data/train2_f.csv')
item = pd.read_csv('data/item_feature2.csv')
shop = pd.read_csv('data/shop_feature2.csv')
item = item.drop_duplicates()
shop = shop.drop_duplicates()
train1_f = pd.merge(train1_f,item[['item_buy','item_id']],on='item_id',how='left')
train1_f = pd.merge(train1_f,shop[['shop_click_buy_total','shop_id']],on='shop_id',how='left')
commodity_shop = train1_f
# 该商品在该店铺点击次数
t = commodity_shop[['item_id','shop_id']]
t['item_shop_click_total'] = 1
t = t.groupby(['item_id','shop_id']).agg(sum).reset_index()
commodity_shop = pd.merge(commodity_shop,t,on=['item_id','shop_id'],how='left')
# 该商品在该店铺购买次数
t = commodity_shop[['item_id','shop_id','is_trade']]
t = t.groupby(['item_id','shop_id']).agg(sum).reset_index()
t.rename(columns={'is_trade':'item_shop_click_buy_total'},inplace=True)
commodity_shop = pd.merge(commodity_shop,t,on=['item_id','shop_id'],how='left')
# 该商品在该店铺购买率
commodity_shop['item_shop_click_buy_rate'] = commodity_shop['item_shop_click_buy_total']/commodity_shop['item_shop_click_total']
# 该商品在该店铺售卖数量/该商品总售卖量
commodity_shop['item_shop_sale_rate'] = commodity_shop['item_shop_click_buy_total']/commodity_shop['item_buy']
# 该商品在该店铺售卖数量/该商户总售卖量
commodity_shop['shop_item_sale_rate'] = commodity_shop['item_shop_click_buy_total']/commodity_shop['shop_click_buy_total']

commodity_shop[['item_shop_click_buy_rate','item_shop_click_buy_total','item_shop_click_total','item_shop_sale_rate','shop_item_sale_rate','shop_id','item_id']].to_csv('data/shop_item_feature2.csv',index=None)




# 从训练集三中获取商品店铺相关的特征
train1_f = pd.read_csv('data/train3_f.csv')
item = pd.read_csv('data/item_feature3.csv')
shop = pd.read_csv('data/shop_feature3.csv')
item = item.drop_duplicates()
shop = shop.drop_duplicates()
train1_f = pd.merge(train1_f,item[['item_buy','item_id']],on='item_id',how='left')
train1_f = pd.merge(train1_f,shop[['shop_click_buy_total','shop_id']],on='shop_id',how='left')
commodity_shop = train1_f
# 该商品在该店铺点击次数
t = commodity_shop[['item_id','shop_id']]
t['item_shop_click_total'] = 1
t = t.groupby(['item_id','shop_id']).agg(sum).reset_index()
commodity_shop = pd.merge(commodity_shop,t,on=['item_id','shop_id'],how='left')
# 该商品在该店铺购买次数
t = commodity_shop[['item_id','shop_id','is_trade']]
t = t.groupby(['item_id','shop_id']).agg(sum).reset_index()
t.rename(columns={'is_trade':'item_shop_click_buy_total'},inplace=True)
commodity_shop = pd.merge(commodity_shop,t,on=['item_id','shop_id'],how='left')
# 该商品在该店铺购买率
commodity_shop['item_shop_click_buy_rate'] = commodity_shop['item_shop_click_buy_total']/commodity_shop['item_shop_click_total']
# 该商品在该店铺售卖数量/该商品总售卖量
commodity_shop['item_shop_sale_rate'] = commodity_shop['item_shop_click_buy_total']/commodity_shop['item_buy']
# 该商品在该店铺售卖数量/该商户总售卖量
commodity_shop['shop_item_sale_rate'] = commodity_shop['item_shop_click_buy_total']/commodity_shop['shop_click_buy_total']

commodity_shop[['item_shop_click_buy_rate','item_shop_click_buy_total','item_shop_click_total','item_shop_sale_rate','shop_item_sale_rate','shop_id','item_id']].to_csv('data/shop_item_feature3.csv',index=None)




# 从训练集4中获取商品店铺相关的特征
train1_f = pd.read_csv('data/train4_f.csv')
item = pd.read_csv('data/item_feature4.csv')
shop = pd.read_csv('data/shop_feature4.csv')
item = item.drop_duplicates()
shop = shop.drop_duplicates()
train1_f = pd.merge(train1_f,item[['item_buy','item_id']],on='item_id',how='left')
train1_f = pd.merge(train1_f,shop[['shop_click_buy_total','shop_id']],on='shop_id',how='left')
commodity_shop = train1_f
# 该商品在该店铺点击次数
t = commodity_shop[['item_id','shop_id']]
t['item_shop_click_total'] = 1
t = t.groupby(['item_id','shop_id']).agg(sum).reset_index()
commodity_shop = pd.merge(commodity_shop,t,on=['item_id','shop_id'],how='left')
# 该商品在该店铺购买次数
t = commodity_shop[['item_id','shop_id','is_trade']]
t = t.groupby(['item_id','shop_id']).agg(sum).reset_index()
t.rename(columns={'is_trade':'item_shop_click_buy_total'},inplace=True)
commodity_shop = pd.merge(commodity_shop,t,on=['item_id','shop_id'],how='left')
# 该商品在该店铺购买率
commodity_shop['item_shop_click_buy_rate'] = commodity_shop['item_shop_click_buy_total']/commodity_shop['item_shop_click_total']
# 该商品在该店铺售卖数量/该商品总售卖量
commodity_shop['item_shop_sale_rate'] = commodity_shop['item_shop_click_buy_total']/commodity_shop['item_buy']
# 该商品在该店铺售卖数量/该商户总售卖量
commodity_shop['shop_item_sale_rate'] = commodity_shop['item_shop_click_buy_total']/commodity_shop['shop_click_buy_total']

commodity_shop[['item_shop_click_buy_rate','item_shop_click_buy_total','item_shop_click_total','item_shop_sale_rate','shop_item_sale_rate','shop_id','item_id']].to_csv('data/shop_item_feature4.csv',index=None)





# # 从训练集5中获取商品店铺相关的特征
# train1_f = pd.read_csv('data/train5_f.csv')
# item = pd.read_csv('data/item_feature5.csv')
# shop = pd.read_csv('data/shop_feature5.csv')
# item = item.drop_duplicates()
# shop = shop.drop_duplicates()
# train1_f = pd.merge(train1_f,item[['item_buy','item_id']],on='item_id',how='left')
# train1_f = pd.merge(train1_f,shop[['shop_click_buy_total','shop_id']],on='shop_id',how='left')
# commodity_shop = train1_f
# # 该商品在该店铺点击次数
# t = commodity_shop[['item_id','shop_id']]
# t['item_shop_click_total'] = 1
# t = t.groupby(['item_id','shop_id']).agg(sum).reset_index()
# commodity_shop = pd.merge(commodity_shop,t,on=['item_id','shop_id'],how='left')
# # 该商品在该店铺购买次数
# t = commodity_shop[['item_id','shop_id','is_trade']]
# t = t.groupby(['item_id','shop_id']).agg(sum).reset_index()
# t.rename(columns={'is_trade':'item_shop_click_buy_total'},inplace=True)
# commodity_shop = pd.merge(commodity_shop,t,on=['item_id','shop_id'],how='left')
# # 该商品在该店铺购买率
# commodity_shop['item_shop_click_buy_rate'] = commodity_shop['item_shop_click_buy_total']/commodity_shop['item_shop_click_total']
# # 该商品在该店铺售卖数量/该商品总售卖量
# commodity_shop['item_shop_sale_rate'] = commodity_shop['item_shop_click_buy_total']/commodity_shop['item_buy']
# # 该商品在该店铺售卖数量/该商户总售卖量
# commodity_shop['shop_item_sale_rate'] = commodity_shop['item_shop_click_buy_total']/commodity_shop['shop_click_buy_total']
#
# commodity_shop[['item_shop_click_buy_rate','item_shop_click_buy_total','item_shop_click_total','item_shop_sale_rate','shop_item_sale_rate','shop_id','item_id']].to_csv('data/shop_item_feature5.csv',index=None)