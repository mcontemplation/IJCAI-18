import pandas as pd
import xgboost as xgb
from sklearn.ensemble import  GradientBoostingClassifier,GradientBoostingRegressor
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import time
from datetime import date
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

train1_p = pd.read_csv('data/train1_p.csv')

cate = pd.get_dummies(train1_p['cate2'])
train1_p = pd.concat([train1_p,cate],axis=1)

user_occupation_id = pd.get_dummies(train1_p['user_occupation_id'])
train1_p = pd.concat([train1_p,user_occupation_id],axis=1)

train1_p['user_gender_id'] = train1_p['user_gender_id'].apply(lambda x:0 if x==2 else x)
user_gender_id = pd.get_dummies(train1_p['user_gender_id'])
train1_p = pd.concat([train1_p,user_gender_id],axis=1)

user_star_level = pd.get_dummies(train1_p['user_star_level'])
train1_p = pd.concat([train1_p,user_star_level],axis=1)


timeduan = pd.get_dummies(train1_p['timeduan'])
train1_p = pd.concat([train1_p,timeduan],axis=1)



user_feature1 = pd.read_csv('data/user_feature1.csv')
user_feature1.drop_duplicates(inplace=True)
shop_feature1 = pd.read_csv('data/shop_feature1.csv')
shop_feature1.drop_duplicates(inplace=True)
item_feature1 = pd.read_csv('data/item_feature1.csv')
item_feature1.drop_duplicates(subset=['item_id'],inplace=True)#
context_feature1 = pd.read_csv('data/context_feature1.csv')
context_feature1.drop_duplicates(inplace=True)
user_shop_feature1 = pd.read_csv('data/user_shop_feature1.csv')
user_shop_feature1.drop_duplicates(inplace=True)
user_item_feature1 = pd.read_csv('data/user_item_feature1.csv')
user_item_feature1.drop_duplicates(subset=['user_id','item_id'],inplace=True)#
user_brand_feature1 = pd.read_csv('data/user_brand_feature1.csv')
user_brand_feature1.drop_duplicates(subset=['user_id','item_brand_id'],inplace=True)
user_context_feature1 = pd.read_csv('data/user_context_feature1.csv')
user_context_feature1.drop_duplicates(subset=['user_id','context_page_id'],inplace=True)
user_price_feature1 = pd.read_csv('data/user_price_feature1.csv')
user_price_feature1.drop_duplicates(subset=['user_id','item_price_level'],inplace=True)
user_collected_feature1 = pd.read_csv('data/user_collected_feature1.csv')
user_collected_feature1.drop_duplicates(subset=['user_id','item_collected_level'],inplace=True)
user_cate_feature1 = pd.read_csv('data/user_cate_feature1.csv')
user_cate_feature1.drop_duplicates(subset=['user_id','cate2'],inplace=True)
shop_item_feature1 = pd.read_csv('data/shop_item_feature1.csv')
shop_item_feature1.drop_duplicates(inplace=True)
user_shop_item_feature1 = pd.read_csv('data/user_shop_item_feature1.csv')
user_shop_item_feature1.drop_duplicates(inplace=True)
age_star_item_feature1 = pd.read_csv('data/age_star_item_feature1.csv')
age_star_item_feature1.drop_duplicates(subset=['user_age_level','user_star_level','item_id'],inplace=True)
star_cate_feature1 = pd.read_csv('data/star_cate_feature1.csv')
star_cate_feature1.drop_duplicates(subset=['user_star_level','cate2'],inplace=True)
star_price_feature1 = pd.read_csv('data/star_price_feature1.csv')
star_price_feature1.drop_duplicates(subset=['user_star_level','item_price_level'],inplace=True)
star_sale_feature1 = pd.read_csv('data/star_sale_feature1.csv')
star_sale_feature1.drop_duplicates(subset=['user_star_level','item_sales_level'],inplace=True)
cate_feature1 = pd.read_csv('data/cate_feature1.csv')
cate_feature1.drop_duplicates(subset=['cate2'],inplace=True)
price_feature1 = pd.read_csv('data/price_feature1.csv')
price_feature1.drop_duplicates(subset=['item_price_level'],inplace=True)
sales_feature1 = pd.read_csv('data/sales_feature1.csv')
sales_feature1.drop_duplicates(subset=['item_sales_level'],inplace=True)
collected_feature1 = pd.read_csv('data/collected_feature1.csv')
collected_feature1.drop_duplicates(subset=['item_collected_level'],inplace=True)
user_time_feature1 = pd.read_csv('data/user_time_feature1.csv')
user_time_feature1.drop_duplicates(subset=['user_id','timeduan'],inplace=True)
cate_price_feature1 = pd.read_csv('data/cate_price_feature1.csv')
cate_price_feature1.drop_duplicates(subset=['cate2','item_price_level'],inplace=True)
# item_price_feature1 = pd.read_csv('data/item_price_feature1.csv')
# item_price_feature1.drop_duplicates(subset=['item_id','item_price_level'],inplace=True)


other_user_feature1 = pd.read_csv('data/other_user_feature1.csv')
other_user_feature1.drop_duplicates(inplace=True)
other_item_feature1 = pd.read_csv('data/other_item_feature1.csv')
other_item_feature1.drop_duplicates(inplace=True)
other_shop_feature1 = pd.read_csv('data/other_shop_feature1.csv')
other_shop_feature1.drop_duplicates(inplace=True)
other_user_item_feature1 = pd.read_csv('data/other_user_item_feature1.csv')
other_user_item_feature1.drop_duplicates(inplace=True)
other_user_shop_feature1 = pd.read_csv('data/other_user_shop_feature1.csv')
other_user_shop_feature1.drop_duplicates(inplace=True)
other_user_hour_feature1 = pd.read_csv('data/other_user_hour_feature1.csv')
other_user_hour_feature1.drop_duplicates(inplace=True)
other_user_cate_feature1 = pd.read_csv('data/other_user_cate_feature1.csv')
other_user_cate_feature1.drop_duplicates(inplace=True)
other_user_brand_feature1 = pd.read_csv('data/other_user_brand_feature1.csv')
other_user_brand_feature1.drop_duplicates(inplace=True)
other_user_shop_item_feature1 = pd.read_csv('data/other_user_shop_item_feature1.csv')
other_user_shop_item_feature1.drop_duplicates(inplace=True)
other_user_item_hour_feature1 = pd.read_csv('data/other_user_item_hour_feature1.csv')
other_user_item_hour_feature1.drop_duplicates(inplace=True)
other_age_time_feature1 = pd.read_csv('data/other_age_time_feature1.csv')
other_age_time_feature1.drop_duplicates(subset=['user_age_level','hour'],inplace=True)



print(train1_p.shape)
train1_p = pd.merge(train1_p,user_feature1,on='user_id',how='left')
train1_p = pd.merge(train1_p,shop_feature1,on='shop_id',how='left')
train1_p = pd.merge(train1_p,item_feature1,on='item_id',how='left')
train1_p = pd.merge(train1_p,context_feature1,on='context_page_id',how='left')
train1_p = pd.merge(train1_p,user_shop_feature1,on=['user_id','shop_id'],how='left')
train1_p = pd.merge(train1_p,user_item_feature1,on=['user_id','item_id'],how='left')
train1_p = pd.merge(train1_p,user_brand_feature1,on=['user_id','item_brand_id'],how='left')
train1_p = pd.merge(train1_p,user_price_feature1,on=['user_id','item_price_level'],how='left')
train1_p = pd.merge(train1_p,user_collected_feature1,on=['user_id','item_collected_level'],how='left')
train1_p = pd.merge(train1_p,user_context_feature1,on=['user_id','context_page_id'],how='left')
train1_p = pd.merge(train1_p,user_cate_feature1,on=['user_id','cate2'],how='left')
train1_p = pd.merge(train1_p,shop_item_feature1,on=['shop_id','item_id'],how='left')
train1_p = pd.merge(train1_p,user_shop_item_feature1,on=['user_id','shop_id','item_id'],how='left')
train1_p = pd.merge(train1_p,other_user_feature1,on=['user_id'],how='left')
train1_p = pd.merge(train1_p,other_item_feature1,on=['item_id'],how='left')
train1_p = pd.merge(train1_p,other_shop_feature1,on=['shop_id'],how='left')
train1_p = pd.merge(train1_p,other_user_item_feature1,on=['user_id','item_id'],how='left')
train1_p = pd.merge(train1_p,other_user_shop_feature1,on=['user_id','shop_id'],how='left')
train1_p = pd.merge(train1_p,other_user_hour_feature1,on=['user_id','hour'],how='left')
train1_p = pd.merge(train1_p,other_user_brand_feature1,on=['user_id','item_brand_id'],how='left')
train1_p = pd.merge(train1_p,other_user_shop_item_feature1,on=['user_id','shop_id','item_id'],how='left')
train1_p = pd.merge(train1_p,other_user_item_hour_feature1,on=['user_id','hour','item_id'],how='left')
train1_p = pd.merge(train1_p,other_user_cate_feature1,on=['user_id','cate2'],how='left')
train1_p = pd.merge(train1_p,other_age_time_feature1,on=['user_age_level','hour'],how='left')
# train1_p = pd.merge(train1_p,other_user_hourclick_feature1,on=['user_id','hour'],how='left')
train1_p = pd.merge(train1_p,age_star_item_feature1,on=['user_age_level','user_star_level','item_id'],how='left')
train1_p = pd.merge(train1_p,star_cate_feature1,on=['user_star_level','cate2'],how='left')
train1_p = pd.merge(train1_p,star_price_feature1,on=['user_star_level','item_price_level'],how='left')
train1_p = pd.merge(train1_p,star_sale_feature1,on=['user_star_level','item_sales_level'],how='left')
# 商品本身
train1_p = pd.merge(train1_p,cate_feature1,on=['cate2'],how='left')
train1_p = pd.merge(train1_p,price_feature1,on=['item_price_level'],how='left')
train1_p = pd.merge(train1_p,sales_feature1,on=['item_sales_level'],how='left')
train1_p = pd.merge(train1_p,collected_feature1,on=['item_collected_level'],how='left')

train1_p = pd.merge(train1_p,user_time_feature1,on=['user_id','timeduan'],how='left')
train1_p = pd.merge(train1_p,cate_price_feature1,on=['cate2','item_price_level'],how='left')
# train1_p = pd.merge(train1_p,item_price_feature1,on=['item_id','item_price_level'],how='left')

print(train1_p.shape)

train2_p = pd.read_csv('data/train2_p.csv')
# train2_ceshi = pd.read_csv('data/train2_p.csv')

cate = pd.get_dummies(train2_p['cate2'])
train2_p = pd.concat([train2_p,cate],axis=1)

user_occupation_id = pd.get_dummies(train2_p['user_occupation_id'])
train2_p = pd.concat([train2_p,user_occupation_id],axis=1)

train2_p['user_gender_id'] = train2_p['user_gender_id'].apply(lambda x:0 if x==2 else x)
user_gender_id = pd.get_dummies(train2_p['user_gender_id'])
train2_p = pd.concat([train2_p,user_gender_id],axis=1)

user_star_level = pd.get_dummies(train2_p['user_star_level'])
train2_p = pd.concat([train2_p,user_star_level],axis=1)


timeduan = pd.get_dummies(train2_p['timeduan'])
train2_p = pd.concat([train2_p,timeduan],axis=1)

user_feature2 = pd.read_csv('data/user_feature2.csv')
user_feature2.drop_duplicates(inplace=True)
shop_feature2 = pd.read_csv('data/shop_feature2.csv')
shop_feature2.drop_duplicates(inplace=True)
item_feature2 = pd.read_csv('data/item_feature2.csv')
item_feature2.drop_duplicates(subset=['item_id'],inplace=True)#
context_feature2 = pd.read_csv('data/context_feature2.csv')
context_feature2.drop_duplicates(inplace=True)
user_shop_feature2 = pd.read_csv('data/user_shop_feature2.csv')
user_shop_feature2.drop_duplicates(inplace=True)
user_item_feature2 = pd.read_csv('data/user_item_feature2.csv')
user_item_feature2.drop_duplicates(subset=['user_id','item_id'],inplace=True)#
user_brand_feature2 = pd.read_csv('data/user_brand_feature2.csv')
user_brand_feature2.drop_duplicates(subset=['user_id','item_brand_id'],inplace=True)
user_context_feature2 = pd.read_csv('data/user_context_feature2.csv')
user_context_feature2.drop_duplicates(subset=['user_id','context_page_id'],inplace=True)
user_price_feature2 = pd.read_csv('data/user_price_feature2.csv')
user_price_feature2.drop_duplicates(subset=['user_id','item_price_level'],inplace=True)
user_collected_feature2 = pd.read_csv('data/user_collected_feature2.csv')
user_collected_feature2.drop_duplicates(subset=['user_id','item_collected_level'],inplace=True)
user_cate_feature2 = pd.read_csv('data/user_cate_feature2.csv')
user_cate_feature2.drop_duplicates(subset=['user_id','cate2'],inplace=True)
shop_item_feature2 = pd.read_csv('data/shop_item_feature2.csv')
shop_item_feature2.drop_duplicates(inplace=True)
user_shop_item_feature2 = pd.read_csv('data/user_shop_item_feature2.csv')
user_shop_item_feature2.drop_duplicates(inplace=True)
age_star_item_feature2 = pd.read_csv('data/age_star_item_feature2.csv')
age_star_item_feature2.drop_duplicates(subset=['user_age_level','user_star_level','item_id'],inplace=True)
star_cate_feature2 = pd.read_csv('data/star_cate_feature2.csv')
star_cate_feature2.drop_duplicates(subset=['user_star_level','cate2'],inplace=True)
star_price_feature2 = pd.read_csv('data/star_price_feature2.csv')
star_price_feature2.drop_duplicates(subset=['user_star_level','item_price_level'],inplace=True)
star_sale_feature2 = pd.read_csv('data/star_sale_feature2.csv')
star_sale_feature2.drop_duplicates(subset=['user_star_level','item_sales_level'],inplace=True)
cate_feature2 = pd.read_csv('data/cate_feature2.csv')
cate_feature2.drop_duplicates(subset=['cate2'],inplace=True)
price_feature2 = pd.read_csv('data/price_feature2.csv')
price_feature2.drop_duplicates(subset=['item_price_level'],inplace=True)
sales_feature2 = pd.read_csv('data/sales_feature2.csv')
sales_feature2.drop_duplicates(subset=['item_sales_level'],inplace=True)
collected_feature2 = pd.read_csv('data/collected_feature2.csv')
collected_feature2.drop_duplicates(subset=['item_collected_level'],inplace=True)
user_time_feature2 = pd.read_csv('data/user_time_feature2.csv')
user_time_feature2.drop_duplicates(subset=['user_id','timeduan'],inplace=True)
cate_price_feature2 = pd.read_csv('data/cate_price_feature2.csv')
cate_price_feature2.drop_duplicates(subset=['cate2','item_price_level'],inplace=True)


other_user_feature2 = pd.read_csv('data/other_user_feature2.csv')
other_user_feature2.drop_duplicates(inplace=True)
other_item_feature2 = pd.read_csv('data/other_item_feature2.csv')
other_item_feature2.drop_duplicates(inplace=True)
other_shop_feature2 = pd.read_csv('data/other_shop_feature2.csv')
other_shop_feature2.drop_duplicates(inplace=True)
other_user_item_feature2 = pd.read_csv('data/other_user_item_feature2.csv')
other_user_item_feature2.drop_duplicates(inplace=True)
other_user_shop_feature2 = pd.read_csv('data/other_user_shop_feature2.csv')
other_user_shop_feature2.drop_duplicates(inplace=True)
other_user_hour_feature2 = pd.read_csv('data/other_user_hour_feature2.csv')
other_user_hour_feature2.drop_duplicates(inplace=True)
other_user_cate_feature2 = pd.read_csv('data/other_user_cate_feature2.csv')
other_user_cate_feature2.drop_duplicates(inplace=True)
other_user_brand_feature2 = pd.read_csv('data/other_user_brand_feature2.csv')
other_user_brand_feature2.drop_duplicates(inplace=True)
# other_shop_brand_feature2 = pd.read_csv('data/other_shop_brand_feature2.csv')
# other_shop_brand_feature2.drop_duplicates(inplace=True)
other_user_shop_item_feature2 = pd.read_csv('data/other_user_shop_item_feature2.csv')
other_user_shop_item_feature2.drop_duplicates(inplace=True)
other_user_item_hour_feature2 = pd.read_csv('data/other_user_item_hour_feature2.csv')
other_user_item_hour_feature2.drop_duplicates(inplace=True)
other_age_time_feature2 = pd.read_csv('data/other_age_time_feature2.csv')
other_age_time_feature2.drop_duplicates(subset=['user_age_level','hour'],inplace=True)


print(train2_p.shape)
train2_p = pd.merge(train2_p,user_feature2,on='user_id',how='left')
train2_p = pd.merge(train2_p,shop_feature2,on='shop_id',how='left')
train2_p = pd.merge(train2_p,item_feature2,on='item_id',how='left')
train2_p = pd.merge(train2_p,context_feature2,on='context_page_id',how='left')
train2_p = pd.merge(train2_p,user_shop_feature2,on=['user_id','shop_id'],how='left')
train2_p = pd.merge(train2_p,user_item_feature2,on=['user_id','item_id'],how='left')
train2_p = pd.merge(train2_p,user_brand_feature2,on=['user_id','item_brand_id'],how='left')
train2_p = pd.merge(train2_p,user_price_feature2,on=['user_id','item_price_level'],how='left')
train2_p = pd.merge(train2_p,user_collected_feature2,on=['user_id','item_collected_level'],how='left')
train2_p = pd.merge(train2_p,user_context_feature2,on=['user_id','context_page_id'],how='left')
train2_p = pd.merge(train2_p,user_cate_feature2,on=['user_id','cate2'],how='left')
train2_p = pd.merge(train2_p,shop_item_feature2,on=['shop_id','item_id'],how='left')
train2_p = pd.merge(train2_p,user_shop_item_feature2,on=['user_id','shop_id','item_id'],how='left')
train2_p = pd.merge(train2_p,other_user_feature2,on=['user_id'],how='left')
train2_p = pd.merge(train2_p,other_item_feature2,on=['item_id'],how='left')
train2_p = pd.merge(train2_p,other_shop_feature2,on=['shop_id'],how='left')
train2_p = pd.merge(train2_p,other_user_item_feature2,on=['user_id','item_id'],how='left')
train2_p = pd.merge(train2_p,other_user_shop_feature2,on=['user_id','shop_id'],how='left')
train2_p = pd.merge(train2_p,other_user_hour_feature2,on=['user_id','hour'],how='left')
train2_p = pd.merge(train2_p,other_user_brand_feature2,on=['user_id','item_brand_id'],how='left')
train2_p = pd.merge(train2_p,other_user_shop_item_feature2,on=['user_id','shop_id','item_id'],how='left')
train2_p = pd.merge(train2_p,other_user_item_hour_feature2,on=['user_id','hour','item_id'],how='left')
train2_p = pd.merge(train2_p,other_user_cate_feature2,on=['user_id','cate2'],how='left')
train2_p = pd.merge(train2_p,other_age_time_feature2,on=['user_age_level','hour'],how='left')


train2_p = pd.merge(train2_p,age_star_item_feature2,on=['user_age_level','user_star_level','item_id'],how='left')
train2_p = pd.merge(train2_p,star_cate_feature2,on=['user_star_level','cate2'],how='left')
train2_p = pd.merge(train2_p,star_price_feature2,on=['user_star_level','item_price_level'],how='left')
train2_p = pd.merge(train2_p,star_sale_feature2,on=['user_star_level','item_sales_level'],how='left')
# 商品本身
train2_p = pd.merge(train2_p,cate_feature2,on=['cate2'],how='left')
train2_p = pd.merge(train2_p,price_feature2,on=['item_price_level'],how='left')
train2_p = pd.merge(train2_p,sales_feature2,on=['item_sales_level'],how='left')
train2_p = pd.merge(train2_p,collected_feature2,on=['item_collected_level'],how='left')

train2_p = pd.merge(train2_p,user_time_feature2,on=['user_id','timeduan'],how='left')
train2_p = pd.merge(train2_p,cate_price_feature2,on=['cate2','item_price_level'],how='left')
# train2_p = pd.merge(train2_p,item_price_feature2,on=['item_id','item_price_level'],how='left')
print(train2_p.shape)


train3_p = pd.read_csv('data/train3_p.csv')

cate = pd.get_dummies(train3_p['cate2'])
train3_p = pd.concat([train3_p,cate],axis=1)


user_occupation_id = pd.get_dummies(train3_p['user_occupation_id'])
train3_p = pd.concat([train3_p,user_occupation_id],axis=1)

train3_p['user_gender_id'] = train3_p['user_gender_id'].apply(lambda x:0 if x==2 else x)
user_gender_id = pd.get_dummies(train3_p['user_gender_id'])
train3_p = pd.concat([train3_p,user_gender_id],axis=1)

user_star_level = pd.get_dummies(train3_p['user_star_level'])
train3_p = pd.concat([train3_p,user_star_level],axis=1)


timeduan = pd.get_dummies(train3_p['timeduan'])
train3_p = pd.concat([train3_p,timeduan],axis=1)


user_feature3 = pd.read_csv('data/user_feature3.csv')
user_feature3.drop_duplicates(inplace=True)
shop_feature3 = pd.read_csv('data/shop_feature3.csv')
shop_feature3.drop_duplicates(inplace=True)
item_feature3 = pd.read_csv('data/item_feature3.csv')
item_feature3.drop_duplicates(subset=['item_id'],inplace=True)#
context_feature3 = pd.read_csv('data/context_feature3.csv')
context_feature3.drop_duplicates(inplace=True)
user_shop_feature3 = pd.read_csv('data/user_shop_feature3.csv')
user_shop_feature3.drop_duplicates(inplace=True)
user_item_feature3 = pd.read_csv('data/user_item_feature3.csv')
user_item_feature3.drop_duplicates(subset=['user_id','item_id'],inplace=True)#
user_brand_feature3 = pd.read_csv('data/user_brand_feature3.csv')
user_brand_feature3.drop_duplicates(subset=['user_id','item_brand_id'],inplace=True)
user_context_feature3 = pd.read_csv('data/user_context_feature3.csv')
user_context_feature3.drop_duplicates(subset=['user_id','context_page_id'],inplace=True)
user_price_feature3 = pd.read_csv('data/user_price_feature3.csv')
user_price_feature3.drop_duplicates(subset=['user_id','item_price_level'],inplace=True)
user_collected_feature3 = pd.read_csv('data/user_collected_feature3.csv')
user_collected_feature3.drop_duplicates(subset=['user_id','item_collected_level'],inplace=True)
user_cate_feature3 = pd.read_csv('data/user_cate_feature3.csv')
user_cate_feature3.drop_duplicates(subset=['user_id','cate2'],inplace=True)
shop_item_feature3 = pd.read_csv('data/shop_item_feature3.csv')
shop_item_feature3.drop_duplicates(inplace=True)
user_shop_item_feature3 = pd.read_csv('data/user_shop_item_feature3.csv')
user_shop_item_feature3.drop_duplicates(inplace=True)
age_star_item_feature3 = pd.read_csv('data/age_star_item_feature3.csv')
age_star_item_feature3.drop_duplicates(subset=['user_age_level','user_star_level','item_id'],inplace=True)
star_cate_feature3 = pd.read_csv('data/star_cate_feature3.csv')
star_cate_feature3.drop_duplicates(subset=['user_star_level','cate2'],inplace=True)
star_price_feature3 = pd.read_csv('data/star_price_feature3.csv')
star_price_feature3.drop_duplicates(subset=['user_star_level','item_price_level'],inplace=True)
star_sale_feature3 = pd.read_csv('data/star_sale_feature3.csv')
star_sale_feature3.drop_duplicates(subset=['user_star_level','item_sales_level'],inplace=True)
cate_feature3 = pd.read_csv('data/cate_feature3.csv')
cate_feature3.drop_duplicates(subset=['cate2'],inplace=True)
price_feature3 = pd.read_csv('data/price_feature3.csv')
price_feature3.drop_duplicates(subset=['item_price_level'],inplace=True)
sales_feature3 = pd.read_csv('data/sales_feature3.csv')
sales_feature3.drop_duplicates(subset=['item_sales_level'],inplace=True)
collected_feature3 = pd.read_csv('data/collected_feature3.csv')
collected_feature3.drop_duplicates(subset=['item_collected_level'],inplace=True)
user_time_feature3 = pd.read_csv('data/user_time_feature3.csv')
user_time_feature3.drop_duplicates(subset=['user_id','timeduan'],inplace=True)
cate_price_feature3 = pd.read_csv('data/cate_price_feature3.csv')
cate_price_feature3.drop_duplicates(subset=['cate2','item_price_level'],inplace=True)
# item_price_feature3 = pd.read_csv('data/item_price_feature3.csv')
# item_price_feature3.drop_duplicates(subset=['item_id','item_price_level'],inplace=True)

other_user_feature3 = pd.read_csv('data/other_user_feature3.csv')
other_user_feature3.drop_duplicates(inplace=True)
other_item_feature3 = pd.read_csv('data/other_item_feature3.csv')
other_item_feature3.drop_duplicates(inplace=True)
other_shop_feature3 = pd.read_csv('data/other_shop_feature3.csv')
other_shop_feature3.drop_duplicates(inplace=True)
other_brand_feature3 = pd.read_csv('data/other_brand_feature3.csv')
other_brand_feature3.drop_duplicates(inplace=True)
other_user_item_feature3 = pd.read_csv('data/other_user_item_feature3.csv')
other_user_item_feature3.drop_duplicates(inplace=True)
other_user_shop_feature3 = pd.read_csv('data/other_user_shop_feature3.csv')
other_user_shop_feature3.drop_duplicates(inplace=True)
other_user_hour_feature3 = pd.read_csv('data/other_user_hour_feature3.csv')
other_user_hour_feature3.drop_duplicates(inplace=True)
other_user_cate_feature3 = pd.read_csv('data/other_user_cate_feature3.csv')
other_user_cate_feature3.drop_duplicates(inplace=True)
other_item_shop_feature3 = pd.read_csv('data/other_item_shop_feature3.csv')
other_item_shop_feature3.drop_duplicates(inplace=True)
other_user_brand_feature3 = pd.read_csv('data/other_user_brand_feature3.csv')
other_user_brand_feature3.drop_duplicates(inplace=True)
# other_shop_brand_feature3 = pd.read_csv('data/other_shop_brand_feature3.csv')
# other_shop_brand_feature3.drop_duplicates(inplace=True)
other_user_shop_item_feature3 = pd.read_csv('data/other_user_shop_item_feature3.csv')
other_user_shop_item_feature3.drop_duplicates(inplace=True)
other_user_item_hour_feature3 = pd.read_csv('data/other_user_item_hour_feature3.csv')
other_user_item_hour_feature3.drop_duplicates(inplace=True)
other_age_time_feature3 = pd.read_csv('data/other_age_time_feature3.csv')
other_age_time_feature3.drop_duplicates(subset=['user_age_level','hour'],inplace=True)



print(train3_p.shape)
train3_p = pd.merge(train3_p,user_feature3,on='user_id',how='left')
train3_p = pd.merge(train3_p,shop_feature3,on='shop_id',how='left')
train3_p = pd.merge(train3_p,item_feature3,on='item_id',how='left')
train3_p = pd.merge(train3_p,context_feature3,on='context_page_id',how='left')
train3_p = pd.merge(train3_p,user_shop_feature3,on=['user_id','shop_id'],how='left')
train3_p = pd.merge(train3_p,user_item_feature3,on=['user_id','item_id'],how='left')
train3_p = pd.merge(train3_p,user_brand_feature3,on=['user_id','item_brand_id'],how='left')
train3_p = pd.merge(train3_p,user_price_feature3,on=['user_id','item_price_level'],how='left')
train3_p = pd.merge(train3_p,user_collected_feature3,on=['user_id','item_collected_level'],how='left')
train3_p = pd.merge(train3_p,user_context_feature3,on=['user_id','context_page_id'],how='left')
train3_p = pd.merge(train3_p,user_cate_feature3,on=['user_id','cate2'],how='left')
train3_p = pd.merge(train3_p,shop_item_feature3,on=['shop_id','item_id'],how='left')
train3_p = pd.merge(train3_p,user_shop_item_feature3,on=['user_id','shop_id','item_id'],how='left')

train3_p = pd.merge(train3_p,other_user_feature3,on=['user_id'],how='left')
train3_p = pd.merge(train3_p,other_item_feature3,on=['item_id'],how='left')
train3_p = pd.merge(train3_p,other_shop_feature3,on=['shop_id'],how='left')
train3_p = pd.merge(train3_p,other_user_item_feature3,on=['user_id','item_id'],how='left')
train3_p = pd.merge(train3_p,other_user_shop_feature3,on=['user_id','shop_id'],how='left')
train3_p = pd.merge(train3_p,other_user_hour_feature3,on=['user_id','hour'],how='left')
train3_p = pd.merge(train3_p,other_user_brand_feature3,on=['user_id','item_brand_id'],how='left')
train3_p = pd.merge(train3_p,other_user_shop_item_feature3,on=['user_id','shop_id','item_id'],how='left')
train3_p = pd.merge(train3_p,other_user_item_hour_feature3,on=['user_id','hour','item_id'],how='left')
train3_p = pd.merge(train3_p,other_user_cate_feature3,on=['user_id','cate2'],how='left')
train3_p = pd.merge(train3_p,other_age_time_feature3,on=['user_age_level','hour'],how='left')

train3_p = pd.merge(train3_p,age_star_item_feature3,on=['user_age_level','user_star_level','item_id'],how='left')
train3_p = pd.merge(train3_p,star_cate_feature3,on=['user_star_level','cate2'],how='left')
train3_p = pd.merge(train3_p,star_price_feature3,on=['user_star_level','item_price_level'],how='left')
train3_p = pd.merge(train3_p,star_sale_feature3,on=['user_star_level','item_sales_level'],how='left')
# 商品本身
train3_p = pd.merge(train3_p,cate_feature3,on=['cate2'],how='left')
train3_p = pd.merge(train3_p,price_feature3,on=['item_price_level'],how='left')
train3_p = pd.merge(train3_p,sales_feature3,on=['item_sales_level'],how='left')
train3_p = pd.merge(train3_p,collected_feature3,on=['item_collected_level'],how='left')

train3_p = pd.merge(train3_p,user_time_feature3,on=['user_id','timeduan'],how='left')
train3_p = pd.merge(train3_p,cate_price_feature3,on=['cate2','item_price_level'],how='left')
# train3_p = pd.merge(train3_p,item_price_feature3,on=['item_id','item_price_level'],how='left')
print(train3_p.shape)


train4_p = pd.read_csv('data/train4_p.csv')

cate = pd.get_dummies(train4_p['cate2'])
train4_p = pd.concat([train4_p,cate],axis=1)


user_occupation_id = pd.get_dummies(train4_p['user_occupation_id'])
train4_p = pd.concat([train4_p,user_occupation_id],axis=1)

train4_p['user_gender_id'] = train4_p['user_gender_id'].apply(lambda x:0 if x==2 else x)
user_gender_id = pd.get_dummies(train4_p['user_gender_id'])
train4_p = pd.concat([train4_p,user_gender_id],axis=1)

user_star_level = pd.get_dummies(train4_p['user_star_level'])
train4_p = pd.concat([train4_p,user_star_level],axis=1)


timeduan = pd.get_dummies(train4_p['timeduan'])
train4_p = pd.concat([train4_p,timeduan],axis=1)

user_feature4 = pd.read_csv('data/user_feature4.csv')
user_feature4.drop_duplicates(inplace=True)
shop_feature4 = pd.read_csv('data/shop_feature4.csv')
shop_feature4.drop_duplicates(inplace=True)
item_feature4 = pd.read_csv('data/item_feature4.csv')
item_feature4.drop_duplicates(subset=['item_id'],inplace=True)#
context_feature4 = pd.read_csv('data/context_feature4.csv')
context_feature4.drop_duplicates(inplace=True)
user_shop_feature4 = pd.read_csv('data/user_shop_feature4.csv')
user_shop_feature4.drop_duplicates(inplace=True)
user_item_feature4 = pd.read_csv('data/user_item_feature4.csv')
user_item_feature4.drop_duplicates(subset=['user_id','item_id'],inplace=True)#
user_brand_feature4 = pd.read_csv('data/user_brand_feature4.csv')
user_brand_feature4.drop_duplicates(subset=['user_id','item_brand_id'],inplace=True)
user_context_feature4 = pd.read_csv('data/user_context_feature4.csv')
user_context_feature4.drop_duplicates(subset=['user_id','context_page_id'],inplace=True)
user_price_feature4 = pd.read_csv('data/user_price_feature4.csv')
user_price_feature4.drop_duplicates(subset=['user_id','item_price_level'],inplace=True)
user_collected_feature4 = pd.read_csv('data/user_collected_feature4.csv')
user_collected_feature4.drop_duplicates(subset=['user_id','item_collected_level'],inplace=True)
user_cate_feature4 = pd.read_csv('data/user_cate_feature4.csv')
user_cate_feature4.drop_duplicates(subset=['user_id','cate2'],inplace=True)
shop_item_feature4 = pd.read_csv('data/shop_item_feature4.csv')
shop_item_feature4.drop_duplicates(inplace=True)
user_shop_item_feature4 = pd.read_csv('data/user_shop_item_feature4.csv')
user_shop_item_feature4.drop_duplicates(inplace=True)
age_star_item_feature4 = pd.read_csv('data/age_star_item_feature4.csv')
age_star_item_feature4.drop_duplicates(subset=['user_age_level','user_star_level','item_id'],inplace=True)
star_cate_feature4 = pd.read_csv('data/star_cate_feature4.csv')
star_cate_feature4.drop_duplicates(subset=['user_star_level','cate2'],inplace=True)
star_price_feature4 = pd.read_csv('data/star_price_feature4.csv')
star_price_feature4.drop_duplicates(subset=['user_star_level','item_price_level'],inplace=True)
star_sale_feature4 = pd.read_csv('data/star_sale_feature4.csv')
star_sale_feature4.drop_duplicates(subset=['user_star_level','item_sales_level'],inplace=True)
cate_feature4 = pd.read_csv('data/cate_feature4.csv')
cate_feature4.drop_duplicates(subset=['cate2'],inplace=True)
price_feature4 = pd.read_csv('data/price_feature4.csv')
price_feature4.drop_duplicates(subset=['item_price_level'],inplace=True)
sales_feature4 = pd.read_csv('data/sales_feature4.csv')
sales_feature4.drop_duplicates(subset=['item_sales_level'],inplace=True)
collected_feature4 = pd.read_csv('data/collected_feature4.csv')
collected_feature4.drop_duplicates(subset=['item_collected_level'],inplace=True)
user_time_feature4 = pd.read_csv('data/user_time_feature4.csv')
user_time_feature4.drop_duplicates(subset=['user_id','timeduan'],inplace=True)
cate_price_feature4 = pd.read_csv('data/cate_price_feature4.csv')
cate_price_feature4.drop_duplicates(subset=['cate2','item_price_level'],inplace=True)
# item_price_feature3 = pd.read_csv('data/item_price_feature3.csv')
# item_price_feature3.drop_duplicates(subset=['item_id','item_price_level'],inplace=True)

other_user_feature4 = pd.read_csv('data/other_user_feature4.csv')
other_user_feature4.drop_duplicates(inplace=True)
other_item_feature4 = pd.read_csv('data/other_item_feature4.csv')
other_item_feature4.drop_duplicates(inplace=True)
other_shop_feature4 = pd.read_csv('data/other_shop_feature4.csv')
other_shop_feature4.drop_duplicates(inplace=True)
other_brand_feature4 = pd.read_csv('data/other_brand_feature4.csv')
other_brand_feature4.drop_duplicates(inplace=True)
other_user_item_feature4 = pd.read_csv('data/other_user_item_feature4.csv')
other_user_item_feature4.drop_duplicates(inplace=True)
other_user_shop_feature4 = pd.read_csv('data/other_user_shop_feature4.csv')
other_user_shop_feature4.drop_duplicates(inplace=True)
other_user_hour_feature4 = pd.read_csv('data/other_user_hour_feature4.csv')
other_user_hour_feature4.drop_duplicates(inplace=True)
other_user_cate_feature4 = pd.read_csv('data/other_user_cate_feature4.csv')
other_user_cate_feature4.drop_duplicates(inplace=True)
other_item_shop_feature4 = pd.read_csv('data/other_item_shop_feature4.csv')
other_item_shop_feature4.drop_duplicates(inplace=True)
other_user_brand_feature4 = pd.read_csv('data/other_user_brand_feature4.csv')
other_user_brand_feature4.drop_duplicates(inplace=True)
# other_shop_brand_feature3 = pd.read_csv('data/other_shop_brand_feature3.csv')
# other_shop_brand_feature3.drop_duplicates(inplace=True)
other_user_shop_item_feature4 = pd.read_csv('data/other_user_shop_item_feature4.csv')
other_user_shop_item_feature4.drop_duplicates(inplace=True)
other_user_item_hour_feature4 = pd.read_csv('data/other_user_item_hour_feature4.csv')
other_user_item_hour_feature4.drop_duplicates(inplace=True)
other_age_time_feature4 = pd.read_csv('data/other_age_time_feature4.csv')
other_age_time_feature4.drop_duplicates(subset=['user_age_level','hour'],inplace=True)



print(train4_p.shape)
train4_p = pd.merge(train4_p,user_feature4,on='user_id',how='left')
train4_p = pd.merge(train4_p,shop_feature4,on='shop_id',how='left')
train4_p = pd.merge(train4_p,item_feature4,on='item_id',how='left')
train4_p = pd.merge(train4_p,context_feature4,on='context_page_id',how='left')
train4_p = pd.merge(train4_p,user_shop_feature4,on=['user_id','shop_id'],how='left')
train4_p = pd.merge(train4_p,user_item_feature4,on=['user_id','item_id'],how='left')
train4_p = pd.merge(train4_p,user_brand_feature4,on=['user_id','item_brand_id'],how='left')
train4_p = pd.merge(train4_p,user_price_feature4,on=['user_id','item_price_level'],how='left')
train4_p = pd.merge(train4_p,user_collected_feature4,on=['user_id','item_collected_level'],how='left')
train4_p = pd.merge(train4_p,user_context_feature4,on=['user_id','context_page_id'],how='left')
train4_p = pd.merge(train4_p,user_cate_feature4,on=['user_id','cate2'],how='left')
train4_p = pd.merge(train4_p,shop_item_feature4,on=['shop_id','item_id'],how='left')
train4_p = pd.merge(train4_p,user_shop_item_feature4,on=['user_id','shop_id','item_id'],how='left')

train4_p = pd.merge(train4_p,other_user_feature4,on=['user_id'],how='left')
train4_p = pd.merge(train4_p,other_item_feature4,on=['item_id'],how='left')
train4_p = pd.merge(train4_p,other_shop_feature4,on=['shop_id'],how='left')
train4_p = pd.merge(train4_p,other_user_item_feature4,on=['user_id','item_id'],how='left')
train4_p = pd.merge(train4_p,other_user_shop_feature4,on=['user_id','shop_id'],how='left')
train4_p = pd.merge(train4_p,other_user_hour_feature4,on=['user_id','hour'],how='left')
train4_p = pd.merge(train4_p,other_user_brand_feature4,on=['user_id','item_brand_id'],how='left')
train4_p = pd.merge(train4_p,other_user_shop_item_feature4,on=['user_id','shop_id','item_id'],how='left')
train4_p = pd.merge(train4_p,other_user_item_hour_feature4,on=['user_id','hour','item_id'],how='left')
train4_p = pd.merge(train4_p,other_user_cate_feature4,on=['user_id','cate2'],how='left')
train4_p = pd.merge(train4_p,other_age_time_feature4,on=['user_age_level','hour'],how='left')

train4_p = pd.merge(train4_p,age_star_item_feature4,on=['user_age_level','user_star_level','item_id'],how='left')
train4_p = pd.merge(train4_p,star_cate_feature4,on=['user_star_level','cate2'],how='left')
train4_p = pd.merge(train4_p,star_price_feature4,on=['user_star_level','item_price_level'],how='left')
train4_p = pd.merge(train4_p,star_sale_feature4,on=['user_star_level','item_sales_level'],how='left')
# 商品本身
train4_p = pd.merge(train4_p,cate_feature4,on=['cate2'],how='left')
train4_p = pd.merge(train4_p,price_feature4,on=['item_price_level'],how='left')
train4_p = pd.merge(train4_p,sales_feature4,on=['item_sales_level'],how='left')
train4_p = pd.merge(train4_p,collected_feature4,on=['item_collected_level'],how='left')

train4_p = pd.merge(train4_p,user_time_feature4,on=['user_id','timeduan'],how='left')
train4_p = pd.merge(train4_p,cate_price_feature4,on=['cate2','item_price_level'],how='left')
# train3_p = pd.merge(train3_p,item_price_feature3,on=['item_id','item_price_level'],how='left')
print(train4_p.shape)


# train5_p = pd.read_csv('data/train5_p.csv')
#
# cate = pd.get_dummies(train5_p['cate2'])
# train5_p = pd.concat([train5_p,cate],axis=1)
#
#
# user_occupation_id = pd.get_dummies(train5_p['user_occupation_id'])
# train5_p = pd.concat([train5_p,user_occupation_id],axis=1)
#
# train5_p['user_gender_id'] = train5_p['user_gender_id'].apply(lambda x:0 if x==2 else x)
# user_gender_id = pd.get_dummies(train5_p['user_gender_id'])
# train5_p = pd.concat([train5_p,user_gender_id],axis=1)
#
# user_star_level = pd.get_dummies(train5_p['user_star_level'])
# train5_p = pd.concat([train5_p,user_star_level],axis=1)
#
#
# timeduan = pd.get_dummies(train5_p['timeduan'])
# train5_p = pd.concat([train5_p,timeduan],axis=1)
#
# user_feature5 = pd.read_csv('data/user_feature5.csv')
# user_feature5.drop_duplicates(inplace=True)
# shop_feature5 = pd.read_csv('data/shop_feature5.csv')
# shop_feature5.drop_duplicates(inplace=True)
# item_feature5 = pd.read_csv('data/item_feature5.csv')
# item_feature5.drop_duplicates(subset=['item_id'],inplace=True)#
# context_feature5 = pd.read_csv('data/context_feature5.csv')
# context_feature5.drop_duplicates(inplace=True)
# user_shop_feature5 = pd.read_csv('data/user_shop_feature5.csv')
# user_shop_feature5.drop_duplicates(inplace=True)
# user_item_feature5 = pd.read_csv('data/user_item_feature5.csv')
# user_item_feature5.drop_duplicates(subset=['user_id','item_id'],inplace=True)#
# user_brand_feature5 = pd.read_csv('data/user_brand_feature5.csv')
# user_brand_feature5.drop_duplicates(subset=['user_id','item_brand_id'],inplace=True)
# user_context_feature5 = pd.read_csv('data/user_context_feature5.csv')
# user_context_feature5.drop_duplicates(subset=['user_id','context_page_id'],inplace=True)
# user_price_feature5 = pd.read_csv('data/user_price_feature5.csv')
# user_price_feature5.drop_duplicates(subset=['user_id','item_price_level'],inplace=True)
# user_collected_feature5 = pd.read_csv('data/user_collected_feature5.csv')
# user_collected_feature5.drop_duplicates(subset=['user_id','item_collected_level'],inplace=True)
# user_cate_feature5 = pd.read_csv('data/user_cate_feature5.csv')
# user_cate_feature5.drop_duplicates(subset=['user_id','cate2'],inplace=True)
# shop_item_feature5 = pd.read_csv('data/shop_item_feature5.csv')
# shop_item_feature5.drop_duplicates(inplace=True)
# user_shop_item_feature5 = pd.read_csv('data/user_shop_item_feature5.csv')
# user_shop_item_feature5.drop_duplicates(inplace=True)
# age_star_item_feature5 = pd.read_csv('data/age_star_item_feature5.csv')
# age_star_item_feature5.drop_duplicates(subset=['user_age_level','user_star_level','item_id'],inplace=True)
# star_cate_feature5 = pd.read_csv('data/star_cate_feature5.csv')
# star_cate_feature5.drop_duplicates(subset=['user_star_level','cate2'],inplace=True)
# star_price_feature5 = pd.read_csv('data/star_price_feature5.csv')
# star_price_feature5.drop_duplicates(subset=['user_star_level','item_price_level'],inplace=True)
# star_sale_feature5 = pd.read_csv('data/star_sale_feature5.csv')
# star_sale_feature5.drop_duplicates(subset=['user_star_level','item_sales_level'],inplace=True)
# cate_feature5 = pd.read_csv('data/cate_feature5.csv')
# cate_feature5.drop_duplicates(subset=['cate2'],inplace=True)
# price_feature5 = pd.read_csv('data/price_feature5.csv')
# price_feature5.drop_duplicates(subset=['item_price_level'],inplace=True)
# sales_feature5 = pd.read_csv('data/sales_feature5.csv')
# sales_feature5.drop_duplicates(subset=['item_sales_level'],inplace=True)
# collected_feature5 = pd.read_csv('data/collected_feature5.csv')
# collected_feature5.drop_duplicates(subset=['item_collected_level'],inplace=True)
# user_time_feature5 = pd.read_csv('data/user_time_feature5.csv')
# user_time_feature5.drop_duplicates(subset=['user_id','timeduan'],inplace=True)
# cate_price_feature5 = pd.read_csv('data/cate_price_feature5.csv')
# cate_price_feature5.drop_duplicates(subset=['cate2','item_price_level'],inplace=True)
# # item_price_feature3 = pd.read_csv('data/item_price_feature3.csv')
# # item_price_feature3.drop_duplicates(subset=['item_id','item_price_level'],inplace=True)
#
# other_user_feature5 = pd.read_csv('data/other_user_feature5.csv')
# other_user_feature5.drop_duplicates(inplace=True)
# other_item_feature5 = pd.read_csv('data/other_item_feature5.csv')
# other_item_feature5.drop_duplicates(inplace=True)
# other_shop_feature5 = pd.read_csv('data/other_shop_feature5.csv')
# other_shop_feature5.drop_duplicates(inplace=True)
# other_brand_feature5 = pd.read_csv('data/other_brand_feature5.csv')
# other_brand_feature5.drop_duplicates(inplace=True)
# other_user_item_feature5 = pd.read_csv('data/other_user_item_feature5.csv')
# other_user_item_feature5.drop_duplicates(inplace=True)
# other_user_shop_feature5 = pd.read_csv('data/other_user_shop_feature5.csv')
# other_user_shop_feature5.drop_duplicates(inplace=True)
# other_user_hour_feature5 = pd.read_csv('data/other_user_hour_feature5.csv')
# other_user_hour_feature5.drop_duplicates(inplace=True)
# other_user_cate_feature5 = pd.read_csv('data/other_user_cate_feature5.csv')
# other_user_cate_feature5.drop_duplicates(inplace=True)
# other_item_shop_feature5 = pd.read_csv('data/other_item_shop_feature5.csv')
# other_item_shop_feature5.drop_duplicates(inplace=True)
# other_user_brand_feature5 = pd.read_csv('data/other_user_brand_feature5.csv')
# other_user_brand_feature5.drop_duplicates(inplace=True)
# # other_shop_brand_feature3 = pd.read_csv('data/other_shop_brand_feature3.csv')
# # other_shop_brand_feature3.drop_duplicates(inplace=True)
# other_user_shop_item_feature5 = pd.read_csv('data/other_user_shop_item_feature5.csv')
# other_user_shop_item_feature5.drop_duplicates(inplace=True)
# other_user_item_hour_feature5 = pd.read_csv('data/other_user_item_hour_feature5.csv')
# other_user_item_hour_feature5.drop_duplicates(inplace=True)
# other_age_time_feature5 = pd.read_csv('data/other_age_time_feature5.csv')
# other_age_time_feature5.drop_duplicates(subset=['user_age_level','hour'],inplace=True)
#
#
#
# print(train5_p.shape)
# train5_p = pd.merge(train5_p,user_feature5,on='user_id',how='left')
# train5_p = pd.merge(train5_p,shop_feature5,on='shop_id',how='left')
# train5_p = pd.merge(train5_p,item_feature5,on='item_id',how='left')
# train5_p = pd.merge(train5_p,context_feature5,on='context_page_id',how='left')
# train5_p = pd.merge(train5_p,user_shop_feature5,on=['user_id','shop_id'],how='left')
# train5_p = pd.merge(train5_p,user_item_feature5,on=['user_id','item_id'],how='left')
# train5_p = pd.merge(train5_p,user_brand_feature5,on=['user_id','item_brand_id'],how='left')
# train5_p = pd.merge(train5_p,user_price_feature5,on=['user_id','item_price_level'],how='left')
# train5_p = pd.merge(train5_p,user_collected_feature5,on=['user_id','item_collected_level'],how='left')
# train5_p = pd.merge(train5_p,user_context_feature5,on=['user_id','context_page_id'],how='left')
# train5_p = pd.merge(train5_p,user_cate_feature5,on=['user_id','cate2'],how='left')
# train5_p = pd.merge(train5_p,shop_item_feature5,on=['shop_id','item_id'],how='left')
# train5_p = pd.merge(train5_p,user_shop_item_feature5,on=['user_id','shop_id','item_id'],how='left')
#
# train5_p = pd.merge(train5_p,other_user_feature5,on=['user_id'],how='left')
# train5_p = pd.merge(train5_p,other_item_feature5,on=['item_id'],how='left')
# train5_p = pd.merge(train5_p,other_shop_feature5,on=['shop_id'],how='left')
# train5_p = pd.merge(train5_p,other_user_item_feature5,on=['user_id','item_id'],how='left')
# train5_p = pd.merge(train5_p,other_user_shop_feature5,on=['user_id','shop_id'],how='left')
# train5_p = pd.merge(train5_p,other_user_hour_feature5,on=['user_id','hour'],how='left')
# train5_p = pd.merge(train5_p,other_user_brand_feature5,on=['user_id','item_brand_id'],how='left')
# train5_p = pd.merge(train5_p,other_user_shop_item_feature5,on=['user_id','shop_id','item_id'],how='left')
# train5_p = pd.merge(train5_p,other_user_item_hour_feature5,on=['user_id','hour','item_id'],how='left')
# train5_p = pd.merge(train5_p,other_user_cate_feature5,on=['user_id','cate2'],how='left')
# train5_p = pd.merge(train5_p,other_age_time_feature5,on=['user_age_level','hour'],how='left')
#
# train5_p = pd.merge(train5_p,age_star_item_feature5,on=['user_age_level','user_star_level','item_id'],how='left')
# train5_p = pd.merge(train5_p,star_cate_feature5,on=['user_star_level','cate2'],how='left')
# train5_p = pd.merge(train5_p,star_price_feature5,on=['user_star_level','item_price_level'],how='left')
# train5_p = pd.merge(train5_p,star_sale_feature5,on=['user_star_level','item_sales_level'],how='left')
# # 商品本身
# train5_p = pd.merge(train5_p,cate_feature5,on=['cate2'],how='left')
# train5_p = pd.merge(train5_p,price_feature5,on=['item_price_level'],how='left')
# train5_p = pd.merge(train5_p,sales_feature5,on=['item_sales_level'],how='left')
# train5_p = pd.merge(train5_p,collected_feature5,on=['item_collected_level'],how='left')
#
# train5_p = pd.merge(train5_p,user_time_feature5,on=['user_id','timeduan'],how='left')
# train5_p = pd.merge(train5_p,cate_price_feature5,on=['cate2','item_price_level'],how='left')
# # train3_p = pd.merge(train3_p,item_price_feature3,on=['item_id','item_price_level'],how='left')
# print(train5_p.shape)


train3_pre = train3_p[['instance_id']]
train2_pre = train2_p[['instance_id']]
def ipri1(x):
    if(x==0)|(x==1)|(x==2)|(x==3):
        return 'ipp1'
    if(x==4)|(x==5):
        return 'ipp2'
    if(x==6)|(x==7)|(x==8):
        return 'ipp3'
    return 'ipp4'
train1_p['price_duan'] = train1_p['item_price_level'].apply(ipri1)
train2_p['price_duan'] = train2_p['item_price_level'].apply(ipri1)
train3_p['price_duan'] = train3_p['item_price_level'].apply(ipri1)
train4_p['price_duan'] = train4_p['item_price_level'].apply(ipri1)
# train5_p['price_duan'] = train5_p['item_price_level'].apply(ipri1)
price_duan = pd.get_dummies(train1_p['price_duan'])
train1_p = pd.concat([train1_p,price_duan],axis=1)
price_duan = pd.get_dummies(train2_p['price_duan'])
train2_p = pd.concat([train2_p,price_duan],axis=1)
price_duan = pd.get_dummies(train3_p['price_duan'])
train3_p = pd.concat([train3_p,price_duan],axis=1)
price_duan = pd.get_dummies(train4_p['price_duan'])
train4_p = pd.concat([train4_p,price_duan],axis=1)
# price_duan = pd.get_dummies(train5_p['price_duan'])
# train5_p = pd.concat([train5_p,price_duan],axis=1)
drop_ele = [
            'instance_id','item_id','item_property_list','item_brand_id','item_city_id','item_pv_level',
            'user_id','user_occupation_id',
           'context_id','context_timestamp','predict_category_property','shop_id','cate2','cate3',
           'user_click_min','user_click_mean' ,'user_am_click',
           'is_high_sale','user_price_crate','user_brand_today_click',
           'max_sale_hour', 'min_age','day',
           'user_brand_rate','user_collected_brate','user_cate_buy',2011981573061447208,
           'ustar','uage','iprice','user_cate_brate','timeduan','us1','us2','us3','user_star_level','ip1','ip2','ip3','price_duan','max_age','min_sale_hour',
           'cate_price_click','cate_price_buy','item_price_rate','item_cate_rate',
           'cate2_click','price_click','page_click','sale_on_am','sale_on_pm','user_pm_click','user_shop_itnum','user_click_max','is_am','collected_click','shop_cate2_btotal',
           'user_click_max','sale_am','sale_pm','shop_user_click_buy_rate','ua1','ua2','ua3','user_gender_id','mean_time','is_weekend',

           #'baitian', 'lingchen', 'yejian','distance', 'age_star_click_buy_rate',
              # 'mean_age',
            # 'ipp1','ipp2','ipp3','ipp4','before','after', 'now','ubefore', 'uafter', 'unow',
            # 'age_star_click_total', 'age_star_click_buy_total','sales_click','user_time_click','user_click_total', 'user_click_buy_total','user_click_difshop_total','user_shop_click_buy_total', 'user_shop_click_total',
            # 'item_collected_level', 'u_s_i_tclick', 'u_s_i_tclick_rate', 'u_i_s_tclick_rate', 'b/a', 'user_cate_tclick', 'user_cate_trate',

           #  'user_age_level', 'user_star_level',
           # 'shop_star_level',
            ]

#  '2.27313E+16', '5.0966E+17', '1.96806E+18', '2.01198E+18', '2.43672E+18','user_occupation_id',
#            '3.20367E+18', '4.87972E+18', '5.75569E+18', '5.79935E+18', '-1', '2004', '2005','ubefore',

# ,'user_cate_buy',  'sale_am',, 'user_shop_click_buy_rate',
#              'user_shop_click_buy_rate.1',\
#             'u_i_s_tclick_rate',
#            'sales_user_tcrate', 'item_user_rate', 'user_shop_today', 'shop_user_rate',
#          , 'user_shop_itnum','user_collected_ctotal','user_price_ctotal',
#
#              'user_cate_click',
train1_p_y = train1_p.is_trade
train1_p_x = train1_p.drop(drop_ele,axis=1)
train1_p_x.corr().to_csv('cor.csv')

train2_p_y = train2_p.is_trade
train2_p_x =train2_p.drop(drop_ele,axis=1)

train4_p_y = train4_p.is_trade
train4_p_x =train4_p.drop(drop_ele,axis=1)

# train5_p_y = train5_p.is_trade
# train5_p_x =train5_p.drop(drop_ele,axis=1)




train12 = pd.concat([train1_p_x,train2_p_x],axis=0)
train12_y = train12.is_trade

# train125 = pd.concat([train5_p_x,train1_p_x,train2_p_x],axis=0)
# train125_y = train125.is_trade

train124 = pd.concat([train1_p_x,train2_p_x,train4_p_x],axis=0)
train124_y = train124.is_trade

train3_p = train3_p.drop(drop_ele,axis=1)


train2_p_x = train2_p_x.drop('is_trade',axis=1)
train1_p_x = train1_p_x.drop('is_trade',axis=1)
train4_p_x = train4_p_x.drop('is_trade',axis=1)
# train5_p_x = train5_p_x.drop('is_trade',axis=1)
train12 = train12.drop('is_trade',axis=1)
# train125 = train125.drop('is_trade',axis=1)
train124 = train124.drop('is_trade',axis=1)
train12_gdbt = train12


# colormap = plt.cm.RdBu
# plt.figure(figsize=(14,12))
# plt.title('Pearson Correlation of Features', y=1.05, size=15)
# data = train12[['shop_hour_tclick','shop_hour_trate','shop_id','hour','is_trade']]
# sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0,square=True,cmap=colormap,linecolor='white',annot=True)
# plt.show()
print(train3_p.columns.tolist())

# train1_p = xgb.DMatrix(train1_p_x, label=train1_p_y)
# train2_p = xgb.DMatrix(train2_p_x, label=train2_p_y)
# train4_p = xgb.DMatrix(train4_p_x, label=train4_p_y)
# # train5_p = xgb.DMatrix(train5_p_x, label=train5_p_y)
# train12 = xgb.DMatrix(train12,label=train12_y)
# train124 = xgb.DMatrix(train124,label=train124_y)
# train3_p = xgb.DMatrix(train3_p)
# params = {'booster': 'gbtree',
#           'objective': 'binary:logistic',
#           'scale_pos_weight': 1,
#           'eval_metric': 'logloss',
#           'gamma': 0.1,
#           'min_child_weight': 1.0,
#           'max_depth': 4,
#           'lambda': 15,
#           'subsample': 0.7,
#           'colsample_bytree': 0.7,
#           # 'colsample_bylevel': 0.7,
#           'eta': 0.009,
#           'tree_method': 'exact',
#           'seed': 0,
#           'nthread': 6
#           }
# watchlist = [(train12,'train'),(train4_p,'test')]
# model = xgb.train(params,train12,num_boost_round=1300,evals=watchlist,early_stopping_rounds=100)
# # #
# # # print(model.get_score(importance_type='gain'))
# # pre1 = model.predict(train4_p)
# # train2_pre = pd.DataFrame(index=None)
# # # # print(log_loss(train2_p_y,pre1))
# # # # # def my(a):
# # # # #     if a<0.5:
# # # # #         if a-0.01>=0:
# # # # #             a=a-0.01
# # # # #         else:
# # # # #             a=0
# # train2_pre['pre'] = pre1
# # # # train2_pre['pre'].apply(my)
# # # # pre1 = train2_pre['pre']
# # # # print(log_loss(train2_p_y,pre1))
# # # # train2_pre['pre'] = pre1
# # train2_pre.to_csv('train2_pre_xgb.csv',index=None)
# watchlist = [(train124,'train')]
# model = xgb.train(params,train124,num_boost_round=1000,evals=watchlist)
# train3_pre['predicted_score'] = model.predict(train3_p)
# train3_pre.to_csv('xgb_adv_pred.csv',sep=' ',index=None)

inf1 = np.isinf(train1_p_x)
inf2 = np.isinf(train2_p_x)
inf4 = np.isinf(train4_p_x)
# inf5 = np.isinf(train5_p_x)
inf12 = np.isinf(train12)
inf124 = np.isinf(train124)
# inf5124 = np.isinf(train5124)
inf3 = np.isinf(train3_p)
train1_p_x[inf1]=0
train2_p_x[inf2]=0
train4_p_x[inf4]=0
# train5_p_x[inf5]=0
train3_p[inf3]=0
train12[inf12]=0
train124[inf124] = 0
# train5124[inf5124]=0
train1_p_x.fillna(0,inplace=True)
train2_p_x.fillna(0,inplace=True)
train4_p_x.fillna(0,inplace=True)
train12.fillna(0,inplace=True)
train124.fillna(0,inplace=True)
train3_p.fillna(0,inplace=True)
clf1 = GradientBoostingClassifier(
loss='deviance',  ##损失函数默认deviance  deviance具有概率输出的分类的偏差
n_estimators=900, ##默认100 回归树个数 弱学习器个数
learning_rate=0.01,  ##默认0.1学习速率/步长0.0-1.0的超参数  每个树学习前一个树的残差的步长
max_depth=4,   ## 默认值为3每个回归树的深度  控制树的大小 也可用叶节点的数量max leaf nodes控制
subsample=0.7,  ##树生成时对样本采样 选择子样本<1.0导致方差的减少和偏差的增加
# min_impurity_decrease=1e-7, ##停止分裂叶子节点的阈值
verbose=2,  ##打印输出 大于1打印每棵树的进度和性能
warm_start=False, ##True在前面基础上增量训练(重设参数减少训练次数) False默认擦除重新训练
random_state=0,  ##随机种子-方便重现
)
# clf1.fit(train12,train12_y)
# pre = clf1.predict_proba(train4_p_x)
# print(log_loss(train4_p_y,pre))
# gbdt = pd.DataFrame(index=None)
# pre = pre[:,1]
# gbdt['pre'] = pre
# gbdt.to_csv('train2_pre_gbdt.csv',index=None)

clf1.fit(train124,train124_y)
pre = clf1.predict_proba(train3_p)
gbdt_adv_pre = pd.DataFrame(index=None)
# gbdt_adv_pre['instance_id'] = train3_pre['instance_id']
gbdt_adv_pre['pre'] = pre[:,1]
gbdt_adv_pre.to_csv('gbdt_adv_pre.csv')

# xgb1 = xgb.XGBClassifier(
#  learning_rate = 0.01,
#  n_estimators= 400,
#  max_depth= 5,
#  min_child_weight= 2,
#  #gamma=1,
#  gamma=0.9,
#  subsample=0.8,
#  colsample_bytree=0.8,
#  objective= 'binary:logistic',
#  nthread= -1,
#  scale_pos_weight=1,
# )
# lr = LogisticRegression()
# xgb = xgb.XGBClassifier(
#           # booster = 'gbtree',
#           objective = 'binary:logistic',
#           scale_pos_weight = 1,
#           gamma = 0.1,
#           min_child_weight = 1.1,
#           max_depth = 5,
#           # lambda = 15,
#           reg_lambda = 15,
#           n_estimators = 800,
#           subsample = 0.8,
#           colsample_bytree = 0.7,
#           colsample_bylevel = 0.7,
#           learning_rate = 0.01,
#           seed = 0,
#           nthread = 4,
#           silent = 0
# )

#
# sclf = StackingClassifier(classifiers=[xgb,clf1],
#                           use_probas=True,
#                           average_probas=False,
#                           verbose = 1,
#                           meta_classifier=xgb1)
# sclf.fit(train1_p_x,train1_p_y)
# pre = sclf.predict_proba(train1_p_x)
# print(log_loss(train1_p_y,pre))
# pre = sclf.predict_proba(train2_p_x)
# print(log_loss(train2_p_y,pre))

# inf1 = np.isinf(train1_p_x)
# inf2 = np.isinf(train2_p_x)
# inf4 = np.isinf(train4_p_x)
# # inf5 = np.isinf(train5_p_x)
# inf12 = np.isinf(train12)
# inf124 = np.isinf(train124)
# # inf5124 = np.isinf(train5124)
# inf3 = np.isinf(train3_p)
#
# train1_p_x[inf1]=0
# train2_p_x[inf2]=0
# train4_p_x[inf4]=0
# # train5_p_x[inf5]=0
# train3_p[inf3]=0
# train12[inf12]=0
# train124[inf124] = 0
# # train5124[inf5124]=0
# train1_p_x.fillna(0,inplace=True)
# train2_p_x.fillna(0,inplace=True)
# train4_p_x.fillna(0,inplace=True)
# # train5_p_x.fillna(0,inplace=True)
# train12.fillna(0,inplace=True)
# train124.fillna(0,inplace=True)
# # train5124.fillna(0,inplace=True)
# train3_p.fillna(0,inplace=True)
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# sc.fit(train2_p_x)
# train1_p_x = sc.transform(train1_p_x)
# train2_p_x = sc.transform(train2_p_x)
# train4_p_x = sc.transform(train4_p_x)
# # train5_p_x = sc.transform(train5_p_x)
# train12 = sc.transform(train12)
# train124 = sc.transform(train124)
# # train5124 = sc.transform(train5124)
# train3_p = sc.transform(train3_p)
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(random_state=0)
# lr.fit(train12,train12_y)
# pre = lr.predict_proba(train4_p_x)
# print(log_loss(train4_p_y,pre))
# lr.fit(train124,train124_y)
# lr_pre = lr.predict_proba(train3_p)
# # #
# pre = pre[:,1]
# lr_pre = lr_pre[:,1]
# lr = pd.DataFrame()
# lr['pre'] = pre
# lr.to_csv('train2_pre_lr.csv',index=None)
# lr_ans = pd.DataFrame()
# lr_ans['pre'] = lr_pre
# lr_ans.to_csv('lr_adv_pred.csv',index=None)

# lr = pd.read_csv('train2_pre_lr.csv')
# xgbd = pd.read_csv('train2_pre_xgb.csv')
# print(log_loss(train2_p_y,xgbd['pre']))
# pre = 0.7*xgbd['pre']+0.3*lr['pre']
# print(log_loss(train2_p_y,pre))

# train1_p_x.fillna(0,inplace=True)
# train2_p_x.fillna(0,inplace=True)
# train12.fillna(0,inplace=True)
# train3_p.fillna(0,inplace=True)
#
# rdt = RandomForestClassifier(max_depth=7,n_estimators=550,random_state=0)
# rdt.fit(train12,train12_y)
# pre = rdt.predict_proba(train3_p)
# pre = pre[:,1]
# c = pd.DataFrame()
# c['pre'] = pre
# c.to_csv('rfc_adv_pred.csv',index=None)






















    # param_test = {
#     'n_estimators': range(2000, 3000, 100),
#     'max_depth': range(4, 7, 1)
# }
# model =clf
#
# from sklearn.model_selection import GridSearchCV
# grid_search = GridSearchCV(estimator = model, param_grid = param_test, scoring='accuracy', cv=5)
# grid_search.fit(train1_p_x, train1_p_y)
# print(grid_search.best_params_, grid_search.best_score_)