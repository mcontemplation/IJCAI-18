import pandas as pd
import xgboost as xgb
from sklearn.ensemble import  GradientBoostingClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import log_loss

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

# gender = pd.get_dummies(train1_p['user_gender_id'])
# train1_p = pd.concat([train1_p,gender],axis=1)


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
user_pv_feature1 = pd.read_csv('data/user_pv_feature1.csv')
user_pv_feature1.drop_duplicates(subset=['user_id','item_pv_level'],inplace=True)
user_hour_feature1 = pd.read_csv('data/user_hour_feature1.csv')
user_hour_feature1.drop_duplicates(subset=['user_id','hour'],inplace=True)
user_cate_feature1 = pd.read_csv('data/user_cate_feature1.csv')
user_cate_feature1.drop_duplicates(subset=['user_id','cate2'],inplace=True)
user_city_feature1 = pd.read_csv('data/user_city_feature1.csv')
user_city_feature1.drop_duplicates(subset=['user_id','item_city_id'],inplace=True)
shop_brand_feature1 = pd.read_csv('data/shop_brand_feature1.csv')
shop_brand_feature1.drop_duplicates(subset=['shop_id','item_brand_id'],inplace=True)
shop_item_feature1 = pd.read_csv('data/shop_item_feature1.csv')
shop_item_feature1.drop_duplicates(inplace=True)
shop_hour_feature1 = pd.read_csv('data/shop_hour_feature1.csv')
shop_hour_feature1.drop_duplicates(inplace=True)
user_shop_item_feature1 = pd.read_csv('data/user_shop_item_feature1.csv')
user_shop_item_feature1.drop_duplicates(subset=['user_id','shop_id','item_id'],inplace=True)
age_item_feature1 = pd.read_csv('data/age_item_feature1.csv')
age_item_feature1.drop_duplicates(subset=['user_age_level','item_id'],inplace=True)
star_item_feature1 = pd.read_csv('data/star_item_feature1.csv')
star_item_feature1.drop_duplicates(subset=['user_star_level','item_id'],inplace=True)
occupation_item_feature1 = pd.read_csv('data/occupation_item_feature1.csv')
occupation_item_feature1.drop_duplicates(subset=['user_occupation_id','item_id'],inplace=True)
age_occupation_item_feature1 = pd.read_csv('data/age_occupation_item_feature1.csv')
age_occupation_item_feature1.drop_duplicates(subset=['user_age_level','user_occupation_id','item_id'],inplace=True)
age_star_item_feature1 = pd.read_csv('data/age_star_item_feature1.csv')
age_star_item_feature1.drop_duplicates(subset=['user_age_level','user_star_level','item_id'],inplace=True)
occupation_star_item_feature1 = pd.read_csv('data/occupation_star_item_feature1.csv')
occupation_star_item_feature1.drop_duplicates(subset=['user_occupation_id','user_star_level','item_id'],inplace=True)
occupation_star_age_item_feature1 = pd.read_csv('data/occupation_star_age_item_feature1.csv')
occupation_star_age_item_feature1.drop_duplicates(subset=['user_occupation_id','user_star_level','user_age_level','item_id'],inplace=True)
age_cate_feature1 = pd.read_csv('data/age_cate_feature1.csv')
age_cate_feature1.drop_duplicates(subset=['user_age_level','cate2'],inplace=True)
star_cate_feature1 = pd.read_csv('data/star_cate_feature1.csv')
star_cate_feature1.drop_duplicates(subset=['user_star_level','cate2'],inplace=True)
occupation_cate_feature1 = pd.read_csv('data/occupation_cate_feature1.csv')
occupation_cate_feature1.drop_duplicates(subset=['user_occupation_id','cate2'],inplace=True)
item_brand_feature1 = pd.read_csv('data/item_brand_feature1.csv')
item_brand_feature1.drop_duplicates(subset=['item_id','item_brand_id'],inplace=True)
occupation_brand_feature1 = pd.read_csv('data/occupation_brand_feature1.csv')
occupation_brand_feature1.drop_duplicates(subset=['user_occupation_id','item_brand_id'],inplace=True)
star_brand_feature1 = pd.read_csv('data/star_brand_feature1.csv')
star_brand_feature1.drop_duplicates(subset=['user_star_level','item_brand_id'],inplace=True)


other_user_feature1 = pd.read_csv('data/other_user_feature1.csv')
other_user_feature1.drop_duplicates(inplace=True)
other_item_feature1 = pd.read_csv('data/other_item_feature1.csv')
other_item_feature1.drop_duplicates(inplace=True)
other_shop_feature1 = pd.read_csv('data/other_shop_feature1.csv')
other_shop_feature1.drop_duplicates(inplace=True)
other_brand_feature1 = pd.read_csv('data/other_brand_feature1.csv')
other_brand_feature1.drop_duplicates(inplace=True)
other_user_item_feature1 = pd.read_csv('data/other_user_item_feature1.csv')
other_user_item_feature1.drop_duplicates(inplace=True)
other_user_shop_feature1 = pd.read_csv('data/other_user_shop_feature1.csv')
other_user_shop_feature1.drop_duplicates(inplace=True)
other_user_hour_feature1 = pd.read_csv('data/other_user_hour_feature1.csv')
other_user_hour_feature1.drop_duplicates(inplace=True)
other_user_cate_feature1 = pd.read_csv('data/other_user_cate_feature1.csv')
other_user_cate_feature1.drop_duplicates(inplace=True)
other_item_shop_feature1 = pd.read_csv('data/other_item_shop_feature1.csv')
other_item_shop_feature1.drop_duplicates(inplace=True)
other_user_brand_feature1 = pd.read_csv('data/other_user_brand_feature1.csv')
other_user_brand_feature1.drop_duplicates(inplace=True)
# other_shop_brand_feature1 = pd.read_csv('data/other_shop_brand_feature1.csv')
# other_shop_brand_feature1.drop_duplicates(inplace=True)
other_user_shop_item_feature1 = pd.read_csv('data/other_user_shop_item_feature1.csv')
other_user_shop_item_feature1.drop_duplicates(inplace=True)
other_user_item_hour_feature1 = pd.read_csv('data/other_user_item_hour_feature1.csv')
other_user_item_hour_feature1.drop_duplicates(inplace=True)
other_user_price_feature1 = pd.read_csv('data/other_user_price_feature1.csv')
other_user_price_feature1.drop_duplicates(inplace=True)
other_user_collected_feature1 = pd.read_csv('data/other_user_collected_feature1.csv')
other_user_collected_feature1.drop_duplicates(inplace=True)
other_user_sales_feature1 = pd.read_csv('data/other_user_sales_feature1.csv')
other_user_sales_feature1.drop_duplicates(inplace=True)
other_shop_hour_feature1 = pd.read_csv('data/other_shop_hour_feature1.csv')
other_shop_hour_feature1.drop_duplicates(inplace=True)
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
# train1_p = pd.merge(train1_p,user_pv_feature1,on=['user_id','item_pv_level'],how='left')
train1_p = pd.merge(train1_p,user_context_feature1,on=['user_id','context_page_id'],how='left')
# train1_p = pd.merge(train1_p,user_hour_feature1,on=['user_id','hour'],how='left')
train1_p = pd.merge(train1_p,user_cate_feature1,on=['user_id','cate2'],how='left')
train1_p = pd.merge(train1_p,shop_item_feature1,on=['shop_id','item_id'],how='left')
# train1_p = pd.merge(train1_p,shop_hour_feature1,on=['shop_id','hour'],how='left')
train1_p = pd.merge(train1_p,user_shop_item_feature1,on=['user_id','shop_id','item_id'],how='left')
train1_p = pd.merge(train1_p,other_user_feature1,on=['user_id'],how='left')
train1_p = pd.merge(train1_p,other_item_feature1,on=['item_id'],how='left')
train1_p = pd.merge(train1_p,other_shop_feature1,on=['shop_id'],how='left')
train1_p = pd.merge(train1_p,other_brand_feature1,on=['item_brand_id'],how='left')
train1_p = pd.merge(train1_p,other_user_item_feature1,on=['user_id','item_id'],how='left')
train1_p = pd.merge(train1_p,other_user_shop_feature1,on=['user_id','shop_id'],how='left')
train1_p = pd.merge(train1_p,other_user_hour_feature1,on=['user_id','hour'],how='left')
train1_p = pd.merge(train1_p,other_item_shop_feature1,on=['shop_id','item_id'],how='left')
train1_p = pd.merge(train1_p,other_user_brand_feature1,on=['user_id','item_brand_id'],how='left')
# train1_p = pd.merge(train1_p,other_shop_brand_feature1,on=['shop_id','item_brand_id'],how='left')
train1_p = pd.merge(train1_p,other_user_shop_item_feature1,on=['user_id','shop_id','item_id'],how='left')
train1_p = pd.merge(train1_p,other_user_item_hour_feature1,on=['user_id','hour','item_id'],how='left')
train1_p = pd.merge(train1_p,other_user_price_feature1,on=['user_id','item_price_level'],how='left')
train1_p = pd.merge(train1_p,other_user_collected_feature1,on=['user_id','item_collected_level'],how='left')
# train1_p = pd.merge(train1_p,other_user_sales_feature1,on=['user_id','item_sales_level'],how='left')
train1_p = pd.merge(train1_p,other_user_cate_feature1,on=['user_id','cate2'],how='left')
# train1_p = pd.merge(train1_p,age_item_feature1,on=['user_age_level','item_id'],how='left')
train1_p = pd.merge(train1_p,star_item_feature1,on=['user_star_level','item_id'],how='left')
train1_p = pd.merge(train1_p,occupation_item_feature1,on=['user_occupation_id','item_id'],how='left')
# train1_p = pd.merge(train1_p,age_star_item_feature1,on=['user_age_level','user_star_level','item_id'],how='left')
# train1_p = pd.merge(train1_p,age_occupation_item_feature1,on=['user_age_level','user_occupation_id','item_id'],how='left')
train1_p = pd.merge(train1_p,occupation_star_item_feature1,on=['user_occupation_id','user_star_level','item_id'],how='left')
# train1_p = pd.merge(train1_p,occupation_star_age_item_feature1,on=['user_occupation_id','user_star_level','user_age_level','item_id'],how='left')
# train1_p = pd.merge(train1_p,age_cate_feature1,on=['user_age_level','cate2'],how='left')
train1_p = pd.merge(train1_p,star_cate_feature1,on=['user_star_level','cate2'],how='left')
train1_p = pd.merge(train1_p,occupation_cate_feature1,on=['user_occupation_id','cate2'],how='left')
train1_p = pd.merge(train1_p,item_brand_feature1,on=['item_id','item_brand_id'],how='left')
# train1_p = pd.merge(train1_p,occupation_brand_feature1,on=['user_occupation_id','item_brand_id'],how='left')
# train1_p = pd.merge(train1_p,star_brand_feature1,on=['item_brand_id','user_star_level'],how='left')
print(train1_p.shape)

train2_p = pd.read_csv('data/train2_p.csv')
# train2_ceshi = pd.read_csv('data/train2_p.csv')

cate = pd.get_dummies(train2_p['cate2'])
train2_p = pd.concat([train2_p,cate],axis=1)

user_occupation_id = pd.get_dummies(train2_p['user_occupation_id'])
train2_p = pd.concat([train2_p,user_occupation_id],axis=1)

# gender = pd.get_dummies(train2_p['user_gender_id'])
# train2_p = pd.concat([train2_p,gender],axis=1)

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
user_pv_feature2 = pd.read_csv('data/user_pv_feature2.csv')
user_pv_feature2.drop_duplicates(subset=['user_id','item_pv_level'],inplace=True)
user_hour_feature2 = pd.read_csv('data/user_hour_feature2.csv')
user_hour_feature2.drop_duplicates(subset=['user_id','hour'],inplace=True)
user_cate_feature2 = pd.read_csv('data/user_cate_feature2.csv')
user_cate_feature2.drop_duplicates(subset=['user_id','cate2'],inplace=True)
user_city_feature2 = pd.read_csv('data/user_city_feature2.csv')
user_city_feature2.drop_duplicates(subset=['user_id','item_city_id'],inplace=True)
shop_brand_feature2 = pd.read_csv('data/shop_brand_feature2.csv')
shop_brand_feature2.drop_duplicates(subset=['shop_id','item_brand_id'],inplace=True)
shop_item_feature2 = pd.read_csv('data/shop_item_feature2.csv')
shop_item_feature2.drop_duplicates(inplace=True)
shop_hour_feature2 = pd.read_csv('data/shop_hour_feature2.csv')
shop_hour_feature2.drop_duplicates(inplace=True)
user_shop_item_feature2 = pd.read_csv('data/user_shop_item_feature2.csv')
user_shop_item_feature2.drop_duplicates(inplace=True)
age_item_feature2 = pd.read_csv('data/age_item_feature2.csv')
age_item_feature2.drop_duplicates(subset=['user_age_level','item_id'],inplace=True)
star_item_feature2 = pd.read_csv('data/star_item_feature2.csv')
star_item_feature2.drop_duplicates(subset=['user_star_level','item_id'],inplace=True)
occupation_item_feature2 = pd.read_csv('data/occupation_item_feature2.csv')
occupation_item_feature2.drop_duplicates(subset=['user_occupation_id','item_id'],inplace=True)
age_occupation_item_feature2 = pd.read_csv('data/age_occupation_item_feature2.csv')
age_occupation_item_feature2.drop_duplicates(subset=['user_age_level','user_occupation_id','item_id'],inplace=True)
age_star_item_feature2 = pd.read_csv('data/age_star_item_feature2.csv')
age_star_item_feature2.drop_duplicates(subset=['user_age_level','user_star_level','item_id'],inplace=True)
occupation_star_item_feature2 = pd.read_csv('data/occupation_star_item_feature2.csv')
occupation_star_item_feature2.drop_duplicates(subset=['user_occupation_id','user_star_level','item_id'],inplace=True)
occupation_star_age_item_feature2 = pd.read_csv('data/occupation_star_age_item_feature2.csv')
occupation_star_age_item_feature2.drop_duplicates(subset=['user_occupation_id','user_star_level','user_age_level','item_id'],inplace=True)
age_cate_feature2 = pd.read_csv('data/age_cate_feature2.csv')
age_cate_feature2.drop_duplicates(subset=['user_age_level','cate2'],inplace=True)
star_cate_feature2 = pd.read_csv('data/star_cate_feature2.csv')
star_cate_feature2.drop_duplicates(subset=['user_star_level','cate2'],inplace=True)
occupation_cate_feature2 = pd.read_csv('data/occupation_cate_feature2.csv')
occupation_cate_feature2.drop_duplicates(subset=['user_occupation_id','cate2'],inplace=True)
item_brand_feature2 = pd.read_csv('data/item_brand_feature2.csv')
item_brand_feature2.drop_duplicates(subset=['item_id','item_brand_id'],inplace=True)
occupation_brand_feature2 = pd.read_csv('data/occupation_brand_feature2.csv')
occupation_brand_feature2.drop_duplicates(subset=['user_occupation_id','item_brand_id'],inplace=True)
star_brand_feature2 = pd.read_csv('data/star_brand_feature2.csv')
star_brand_feature2.drop_duplicates(subset=['user_star_level','item_brand_id'],inplace=True)

other_user_feature2 = pd.read_csv('data/other_user_feature2.csv')
other_user_feature2.drop_duplicates(inplace=True)
other_item_feature2 = pd.read_csv('data/other_item_feature2.csv')
other_item_feature2.drop_duplicates(inplace=True)
other_shop_feature2 = pd.read_csv('data/other_shop_feature2.csv')
other_shop_feature2.drop_duplicates(inplace=True)
other_brand_feature2 = pd.read_csv('data/other_brand_feature2.csv')
other_brand_feature2.drop_duplicates(inplace=True)
other_user_item_feature2 = pd.read_csv('data/other_user_item_feature2.csv')
other_user_item_feature2.drop_duplicates(inplace=True)
other_user_shop_feature2 = pd.read_csv('data/other_user_shop_feature2.csv')
other_user_shop_feature2.drop_duplicates(inplace=True)
other_user_hour_feature2 = pd.read_csv('data/other_user_hour_feature2.csv')
other_user_hour_feature2.drop_duplicates(inplace=True)
other_user_cate_feature2 = pd.read_csv('data/other_user_cate_feature2.csv')
other_user_cate_feature2.drop_duplicates(inplace=True)
other_item_shop_feature2 = pd.read_csv('data/other_item_shop_feature2.csv')
other_item_shop_feature2.drop_duplicates(inplace=True)
other_user_brand_feature2 = pd.read_csv('data/other_user_brand_feature2.csv')
other_user_brand_feature2.drop_duplicates(inplace=True)
# other_shop_brand_feature2 = pd.read_csv('data/other_shop_brand_feature2.csv')
# other_shop_brand_feature2.drop_duplicates(inplace=True)
other_user_shop_item_feature2 = pd.read_csv('data/other_user_shop_item_feature2.csv')
other_user_shop_item_feature2.drop_duplicates(inplace=True)
other_user_item_hour_feature2 = pd.read_csv('data/other_user_item_hour_feature2.csv')
other_user_item_hour_feature2.drop_duplicates(inplace=True)
other_user_price_feature2 = pd.read_csv('data/other_user_price_feature2.csv')
other_user_price_feature2.drop_duplicates(inplace=True)
other_user_collected_feature2 = pd.read_csv('data/other_user_collected_feature2.csv')
other_user_collected_feature2.drop_duplicates(inplace=True)
other_user_sales_feature2 = pd.read_csv('data/other_user_sales_feature2.csv')
other_user_sales_feature2.drop_duplicates(inplace=True)
other_shop_hour_feature2 = pd.read_csv('data/other_shop_hour_feature2.csv')
other_shop_hour_feature2.drop_duplicates(inplace=True)

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
# train2_p = pd.merge(train2_p,user_pv_feature2,on=['user_id','item_pv_level'],how='left')
train2_p = pd.merge(train2_p,user_context_feature2,on=['user_id','context_page_id'],how='left')
# train2_p = pd.merge(train2_p,user_hour_feature2,on=['user_id','hour'],how='left')
train2_p = pd.merge(train2_p,user_cate_feature2,on=['user_id','cate2'],how='left')
train2_p = pd.merge(train2_p,shop_item_feature2,on=['shop_id','item_id'],how='left')
# train2_p = pd.merge(train2_p,shop_hour_feature2,on=['shop_id','hour'],how='left')
train2_p = pd.merge(train2_p,user_shop_item_feature2,on=['user_id','shop_id','item_id'],how='left')
train2_p = pd.merge(train2_p,other_user_feature2,on=['user_id'],how='left')
train2_p = pd.merge(train2_p,other_item_feature2,on=['item_id'],how='left')
train2_p = pd.merge(train2_p,other_shop_feature2,on=['shop_id'],how='left')
train2_p = pd.merge(train2_p,other_brand_feature2,on=['item_brand_id'],how='left')
train2_p = pd.merge(train2_p,other_user_item_feature2,on=['user_id','item_id'],how='left')
train2_p = pd.merge(train2_p,other_user_shop_feature2,on=['user_id','shop_id'],how='left')
train2_p = pd.merge(train2_p,other_user_hour_feature2,on=['user_id','hour'],how='left')
train2_p = pd.merge(train2_p,other_item_shop_feature2,on=['shop_id','item_id'],how='left')
train2_p = pd.merge(train2_p,other_user_brand_feature2,on=['user_id','item_brand_id'],how='left')
# train2_p = pd.merge(train2_p,other_shop_brand_feature2,on=['shop_id','item_brand_id'],how='left')
train2_p = pd.merge(train2_p,other_user_shop_item_feature2,on=['user_id','shop_id','item_id'],how='left')
train2_p = pd.merge(train2_p,other_user_item_hour_feature2,on=['user_id','hour','item_id'],how='left')
train2_p = pd.merge(train2_p,other_user_price_feature2,on=['user_id','item_price_level'],how='left')
train2_p = pd.merge(train2_p,other_user_collected_feature2,on=['user_id','item_collected_level'],how='left')
# train2_p = pd.merge(train2_p,other_user_sales_feature2,on=['user_id','item_sales_level'],how='left')
train2_p = pd.merge(train2_p,other_user_cate_feature2,on=['user_id','cate2'],how='left')
# train2_p = pd.merge(train2_p,age_item_feature2,on=['user_age_level','item_id'],how='left')
train2_p = pd.merge(train2_p,star_item_feature2,on=['user_star_level','item_id'],how='left')
train2_p = pd.merge(train2_p,occupation_item_feature2,on=['user_occupation_id','item_id'],how='left')
# train2_p = pd.merge(train2_p,age_star_item_feature2,on=['user_age_level','user_star_level','item_id'],how='left')
# train2_p = pd.merge(train2_p,age_occupation_item_feature2,on=['user_age_level','user_occupation_id','item_id'],how='left')
train2_p = pd.merge(train2_p,occupation_star_item_feature2,on=['user_occupation_id','user_star_level','item_id'],how='left')
# train2_p = pd.merge(train2_p,occupation_star_age_item_feature2,on=['user_occupation_id','user_star_level','user_age_level','item_id'],how='left')
# train2_p = pd.merge(train2_p,age_cate_feature2,on=['user_age_level','cate2'],how='left')
train2_p = pd.merge(train2_p,star_cate_feature2,on=['user_star_level','cate2'],how='left')
train2_p = pd.merge(train2_p,occupation_cate_feature2,on=['user_occupation_id','cate2'],how='left')
train2_p = pd.merge(train2_p,item_brand_feature2,on=['item_id','item_brand_id'],how='left')
# train2_p = pd.merge(train2_p,occupation_brand_feature2,on=['user_occupation_id','item_brand_id'],how='left')
# train2_p = pd.merge(train2_p,star_brand_feature2,on=['item_brand_id','user_star_level'],how='left')
print(train2_p.shape)


train3_p = pd.read_csv('data/train3_p.csv')

cate = pd.get_dummies(train3_p['cate2'])
train3_p = pd.concat([train3_p,cate],axis=1)
user_occupation_id = pd.get_dummies(train3_p['user_occupation_id'])
train3_p = pd.concat([train3_p,user_occupation_id],axis=1)
# gender = pd.get_dummies(train3_p['user_gender_id'])
# train3_p = pd.concat([train3_p,gender],axis=1)

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
user_pv_feature3 = pd.read_csv('data/user_pv_feature3.csv')
user_pv_feature3.drop_duplicates(subset=['user_id','item_pv_level'],inplace=True)
user_hour_feature3 = pd.read_csv('data/user_hour_feature3.csv')
user_hour_feature3.drop_duplicates(subset=['user_id','hour'],inplace=True)
user_cate_feature3 = pd.read_csv('data/user_cate_feature3.csv')
user_cate_feature3.drop_duplicates(subset=['user_id','cate2'],inplace=True)
user_city_feature3 = pd.read_csv('data/user_city_feature3.csv')
user_city_feature3.drop_duplicates(subset=['user_id','item_city_id'],inplace=True)
shop_brand_feature3 = pd.read_csv('data/shop_brand_feature3.csv')
shop_brand_feature3.drop_duplicates(subset=['shop_id','item_brand_id'],inplace=True)
shop_item_feature3 = pd.read_csv('data/shop_item_feature3.csv')
shop_item_feature3.drop_duplicates(inplace=True)
shop_hour_feature3 = pd.read_csv('data/shop_hour_feature3.csv')
shop_hour_feature3.drop_duplicates(inplace=True)
user_shop_item_feature3 = pd.read_csv('data/user_shop_item_feature3.csv')
user_shop_item_feature3.drop_duplicates(inplace=True)
age_item_feature3 = pd.read_csv('data/age_item_feature3.csv')
age_item_feature3.drop_duplicates(subset=['user_age_level','item_id'],inplace=True)
star_item_feature3 = pd.read_csv('data/star_item_feature3.csv')
star_item_feature3.drop_duplicates(subset=['user_star_level','item_id'],inplace=True)
occupation_item_feature3 = pd.read_csv('data/occupation_item_feature3.csv')
occupation_item_feature3.drop_duplicates(subset=['user_occupation_id','item_id'],inplace=True)
age_occupation_item_feature3 = pd.read_csv('data/age_occupation_item_feature3.csv')
age_occupation_item_feature3.drop_duplicates(subset=['user_age_level','user_occupation_id','item_id'],inplace=True)
age_star_item_feature3 = pd.read_csv('data/age_star_item_feature3.csv')
age_star_item_feature3.drop_duplicates(subset=['user_age_level','user_star_level','item_id'],inplace=True)
occupation_star_item_feature3 = pd.read_csv('data/occupation_star_item_feature3.csv')
occupation_star_item_feature3.drop_duplicates(subset=['user_occupation_id','user_star_level','item_id'],inplace=True)
occupation_star_age_item_feature3 = pd.read_csv('data/occupation_star_age_item_feature3.csv')
occupation_star_age_item_feature3.drop_duplicates(subset=['user_occupation_id','user_star_level','user_age_level','item_id'],inplace=True)
age_cate_feature3 = pd.read_csv('data/age_cate_feature3.csv')
age_cate_feature3.drop_duplicates(subset=['user_age_level','cate2'],inplace=True)
star_cate_feature3 = pd.read_csv('data/star_cate_feature3.csv')
star_cate_feature3.drop_duplicates(subset=['user_star_level','cate2'],inplace=True)
occupation_cate_feature3 = pd.read_csv('data/occupation_cate_feature3.csv')
occupation_cate_feature3.drop_duplicates(subset=['user_occupation_id','cate2'],inplace=True)
item_brand_feature3 = pd.read_csv('data/item_brand_feature3.csv')
item_brand_feature3.drop_duplicates(subset=['item_id','item_brand_id'],inplace=True)
occupation_brand_feature3 = pd.read_csv('data/occupation_brand_feature3.csv')
occupation_brand_feature3.drop_duplicates(subset=['user_occupation_id','item_brand_id'],inplace=True)
star_brand_feature3 = pd.read_csv('data/star_brand_feature3.csv')
star_brand_feature3.drop_duplicates(subset=['user_star_level','item_brand_id'],inplace=True)

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
other_user_price_feature3 = pd.read_csv('data/other_user_price_feature3.csv')
other_user_price_feature3.drop_duplicates(inplace=True)
other_user_collected_feature3 = pd.read_csv('data/other_user_collected_feature3.csv')
other_user_collected_feature3.drop_duplicates(inplace=True)
other_user_sales_feature3 = pd.read_csv('data/other_user_sales_feature3.csv')
other_user_sales_feature3.drop_duplicates(inplace=True)
other_shop_hour_feature3 = pd.read_csv('data/other_shop_hour_feature3.csv')
other_shop_hour_feature3.drop_duplicates(inplace=True)
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
# train1_p = pd.merge(train1_p,user_pv_feature1,on=['user_id','item_pv_level'],how='left')
train3_p = pd.merge(train3_p,user_context_feature3,on=['user_id','context_page_id'],how='left')
# train1_p = pd.merge(train1_p,user_hour_feature1,on=['user_id','hour'],how='left')
train3_p = pd.merge(train3_p,user_cate_feature3,on=['user_id','cate2'],how='left')
train3_p = pd.merge(train3_p,shop_item_feature3,on=['shop_id','item_id'],how='left')
# train1_p = pd.merge(train1_p,shop_hour_feature1,on=['shop_id','hour'],how='left')
train3_p = pd.merge(train3_p,user_shop_item_feature3,on=['user_id','shop_id','item_id'],how='left')
train3_p = pd.merge(train3_p,other_user_feature3,on=['user_id'],how='left')
train3_p = pd.merge(train3_p,other_item_feature3,on=['item_id'],how='left')
train3_p = pd.merge(train3_p,other_shop_feature3,on=['shop_id'],how='left')
train3_p = pd.merge(train3_p,other_brand_feature3,on=['item_brand_id'],how='left')
train3_p = pd.merge(train3_p,other_user_item_feature3,on=['user_id','item_id'],how='left')
train3_p = pd.merge(train3_p,other_user_shop_feature3,on=['user_id','shop_id'],how='left')
train3_p = pd.merge(train3_p,other_user_hour_feature3,on=['user_id','hour'],how='left')
train3_p = pd.merge(train3_p,other_item_shop_feature3,on=['shop_id','item_id'],how='left')
train3_p = pd.merge(train3_p,other_user_brand_feature3,on=['user_id','item_brand_id'],how='left')
# train1_p = pd.merge(train1_p,other_shop_brand_feature1,on=['shop_id','item_brand_id'],how='left')
train3_p = pd.merge(train3_p,other_user_shop_item_feature3,on=['user_id','shop_id','item_id'],how='left')
train3_p = pd.merge(train3_p,other_user_item_hour_feature3,on=['user_id','hour','item_id'],how='left')
train3_p = pd.merge(train3_p,other_user_price_feature3,on=['user_id','item_price_level'],how='left')
train3_p = pd.merge(train3_p,other_user_collected_feature3,on=['user_id','item_collected_level'],how='left')
# train1_p = pd.merge(train1_p,other_user_sales_feature1,on=['user_id','item_sales_level'],how='left')
train3_p = pd.merge(train3_p,other_user_cate_feature3,on=['user_id','cate2'],how='left')
# train1_p = pd.merge(train1_p,age_item_feature1,on=['user_age_level','item_id'],how='left')
train3_p = pd.merge(train3_p,star_item_feature3,on=['user_star_level','item_id'],how='left')
train3_p = pd.merge(train3_p,occupation_item_feature3,on=['user_occupation_id','item_id'],how='left')
# train1_p = pd.merge(train1_p,age_star_item_feature1,on=['user_age_level','user_star_level','item_id'],how='left')
# train1_p = pd.merge(train1_p,age_occupation_item_feature1,on=['user_age_level','user_occupation_id','item_id'],how='left')
train3_p = pd.merge(train3_p,occupation_star_item_feature3,on=['user_occupation_id','user_star_level','item_id'],how='left')
# train1_p = pd.merge(train1_p,occupation_star_age_item_feature1,on=['user_occupation_id','user_star_level','user_age_level','item_id'],how='left')
# train1_p = pd.merge(train1_p,age_cate_feature1,on=['user_age_level','cate2'],how='left')
train3_p = pd.merge(train3_p,star_cate_feature3,on=['user_star_level','cate2'],how='left')
train3_p = pd.merge(train3_p,occupation_cate_feature3,on=['user_occupation_id','cate2'],how='left')
train3_p = pd.merge(train3_p,item_brand_feature3,on=['item_id','item_brand_id'],how='left')
# train3_p = pd.merge(train3_p,occupation_brand_feature3,on=['user_occupation_id','item_brand_id'],how='left')
# train3_p = pd.merge(train3_p,star_brand_feature3,on=['item_brand_id','user_star_level'],how='left')

print(train3_p.shape)

train3_pre = train3_p[['instance_id']]
# train2_pre = train2_p[['instance_id']]


drop_ele = [
            'instance_id','item_id','item_property_list','item_brand_id','item_pv_level','item_city_id',
            'user_id','user_occupation_id',
           'context_id','context_timestamp','predict_category_property','shop_id','cate2','cate3',
           'user_click_min','user_click_mean' , 'user_click_amtotal','user_am_click','collected_user_tcrate',
           'user_price_crate','user_brand_today_click','is_high_sale'
           #    'user_age_level', 'user_star_level', 'shop_review_num_level',
           # 'shop_star_level',
            ]
#  '2.27313E+16', '5.0966E+17', '1.96806E+18', '2.01198E+18', '2.43672E+18','user_occupation_id',
#            '3.20367E+18', '4.87972E+18', '5.75569E+18', '5.79935E+18', '-1', '2004', '2005','ubefore',

# ,'user_cate_buy', 'max_sale_hour', 'min_age', 'sale_am',, 'user_shop_click_buy_rate',
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

train12 = pd.concat([train1_p_x,train2_p_x],axis=0)
train12_y = train12.is_trade

train3_p = train3_p.drop(drop_ele,axis=1)

train2_p_x = train2_p_x.drop('is_trade',axis=1)
train1_p_x = train1_p_x.drop('is_trade',axis=1)
train12 = train12.drop('is_trade',axis=1)
train12_gdbt = train12


train1_p = xgb.DMatrix(train1_p_x, label=train1_p_y)
train2_p = xgb.DMatrix(train2_p_x, label=train2_p_y)
train12 = xgb.DMatrix(train12,label=train12_y)
train3_p = xgb.DMatrix(train3_p)
params = {'booster': 'gbtree',
          'objective': 'binary:logistic',
          'scale_pos_weight': 1,
          'eval_metric': 'logloss',
          'gamma': 0.1,
          'min_child_weight': 1.0,
          'max_depth': 4,
          'lambda': 15,
          'subsample': 0.7,
          'colsample_bytree': 0.7,
          # 'colsample_bylevel': 0.7,
          'eta': 0.008,
          'tree_method': 'exact',
          'seed': 0,
          'nthread': 4
          }
watchlist = [(train1_p,'train'),(train2_p,'val')]
model = xgb.train(params,train1_p,num_boost_round=3000,evals=watchlist,early_stopping_rounds=100)
# pre1 = model.predict(train2_p)
# train2_pre = pd.DataFrame(index=None)
# print(log_loss(train2_p_y,pre1))
# def my(a):
#     if a<0.5:
#         if a-0.01>=0:
#             a=a-0.01
#         else:
#             a=0
# train2_pre['pre'] = pre1
# train2_pre['pre'].apply(my)
# pre1 = train2_pre['pre']
# print(log_loss(train2_p_y,pre1))
# train2_pre['pre'] = pre1
# train2_pre.to_csv('train2_pre_xgb.csv',index=None)
# watchlist = [(train12,'train')]
# model = xgb.train(params,train12,num_boost_round=1320,evals=watchlist)
# train3_pre['predicted_score'] = model.predict(train3_p)
# train3_pre.to_csv('xgb_adv_pred.csv',sep=' ',index=None)



# train1_p_x.fillna(0,inplace=True)
# train2_p_x.fillna(0,inplace=True)
# train12.fillna(0,inplace=True)
# train3_p.fillna(0,inplace=True)
# clf1 = GradientBoostingClassifier(
# loss='deviance',  ##损失函数默认deviance  deviance具有概率输出的分类的偏差
# n_estimators=230, ##默认100 回归树个数 弱学习器个数
# learning_rate=0.01,  ##默认0.1学习速率/步长0.0-1.0的超参数  每个树学习前一个树的残差的步长
# max_depth=5,   ## 默认值为3每个回归树的深度  控制树的大小 也可用叶节点的数量max leaf nodes控制
# subsample=0.8,  ##树生成时对样本采样 选择子样本<1.0导致方差的减少和偏差的增加
# # min_impurity_decrease=1e-7, ##停止分裂叶子节点的阈值
# verbose=2,  ##打印输出 大于1打印每棵树的进度和性能
# warm_start=False, ##True在前面基础上增量训练(重设参数减少训练次数) False默认擦除重新训练
# random_state=0,  ##随机种子-方便重现
# )
# # clf1.fit(train1_p_x,train1_p_y)
# # pre = clf1.predict_proba(train2_p_x)
# clf1.fit(train12,train12_y)
# pre = clf1.predict_proba(train3_p)
# gbdt_adv_pre = pd.DataFrame(index=None)
# gbdt_adv_pre['instance_id'] = train3_pre['instance_id']
# gbdt_adv_pre['predicted_score'] = pre[:,1]
# gbdt_adv_pre.to_csv('gbdt_adv_pre.csv')

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