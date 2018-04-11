import pandas as pd
import xgboost as xgb
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import time
from datetime import date
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

train = pd.read_table('data/round1_ijcai_18_train_20180301.txt',delim_whitespace=True)
test = pd.read_table('data/round1_ijcai_18_test_a_20180301.txt',delim_whitespace=True)
train.user_gender_id = train.user_gender_id.replace(-1,0)
test.user_gender_id = test.user_gender_id.replace(-1,0)

train.item_brand_id = train.item_brand_id.replace(-1,train['item_brand_id'].mode())
test.item_brand_id = test.item_brand_id.replace(-1,test['item_brand_id'].mode())

train.item_city_id = train.item_city_id.replace(-1,train['item_city_id'].mode())
test.item_city_id = test.item_city_id.replace(-1,test['item_city_id'].mode())

train.loc[train.item_sales_level==-1,'item_sales_level']=pd.DataFrame(train[train['item_sales_level'].isin([-1])]['item_collected_level'],columns=['item_collected_level']).reset_index()['item_collected_level']
test.loc[test.item_sales_level==-1,'item_sales_level']=pd.DataFrame(test[test['item_sales_level'].isin([-1])]['item_collected_level'],columns=['item_collected_level']).reset_index()['item_collected_level']

train.user_gender_id = train.user_gender_id.replace(-1,train['user_gender_id'].mode())
test.user_gender_id = test.user_gender_id.replace(-1,test['user_gender_id'].mode())

train.user_age_level = train.user_age_level.replace(-1,train['user_age_level'].mode())
test.user_age_level = test.user_age_level.replace(-1,test['user_age_level'].mode())

train.user_occupation_id = train.user_occupation_id.replace(-1,train['user_occupation_id'].mode())
test.user_occupation_id = test.user_occupation_id.replace(-1,test['user_occupation_id'].mode())

train.user_star_level = train.user_star_level.replace(-1,train['user_star_level'].mode())
test.user_star_level = test.user_star_level.replace(-1,test['user_star_level'].mode())

train.shop_review_positive_rate = train.shop_review_positive_rate.replace(-1,1)
test.shop_review_positive_rate = test.shop_review_positive_rate.replace(-1,1)

train.shop_score_service = train.shop_score_service.replace(-1,1)
test.shop_score_service = test.shop_score_service.replace(-1,1)

train.shop_score_delivery = train.shop_score_delivery.replace(-1,1)
test.shop_score_delivery = test.shop_score_delivery.replace(-1,1)

train.shop_score_description = train.shop_score_description.replace(-1,1)
test.shop_score_description = test.shop_score_description.replace(-1,1)

# 首先将商品类目切分
def split_item_category(s):
    cate = s.split(';')
    if len(cate)==3:
        return cate
    elif len(cate)==2:
        cate.append('-1')
    return cate
train['cate']=train['item_category_list'].apply(split_item_category)
train['cate2'] = train['cate'].apply(lambda x:x[1])
train['cate3'] = train['cate'].apply(lambda x:x[2])
train['cate2'].replace('-1',method='ffill')
train['cate3'].replace('-1',method='ffill')
test['cate']=test['item_category_list'].apply(split_item_category)
test['cate2'] = test['cate'].apply(lambda x:x[1])
test['cate3'] = test['cate'].apply(lambda x:x[2])
train.drop(['cate','item_category_list'],axis=1,inplace=True)

test.drop(['cate','item_category_list'],axis=1,inplace=True)

# 将时间戳转化为时间格式
def timestamp_datetime(value):
    format = '%Y%m%d%H%M%S'
    value = time.localtime(value)
    ## 经过localtime转换后变成
    ## time.struct_time(tm_year=2012, tm_mon=3, tm_mday=28, tm_hour=6, tm_min=53, tm_sec=40, tm_wday=2, tm_yday=88, tm_isdst=0)
    # 最后再经过strftime函数转换为正常日期格式。
    dt = time.strftime(format, value)
    return dt
# 将unix时间戳转为正常格式
train['context_timestamp'] = train['context_timestamp'].apply(timestamp_datetime)
test['context_timestamp'] = test['context_timestamp'].apply(timestamp_datetime)

train['is_weekend'] = train['context_timestamp'].apply(lambda x:date(int(x[0:4]),int(x[4:6]),int(x[6:8])).weekday()+1).apply(lambda x:1 if x in (6,7) else 0)
test['is_weekend'] = test['context_timestamp'].apply(lambda x:date(int(x[0:4]),int(x[4:6]),int(x[6:8])).weekday()+1).apply(lambda x:1 if x in (6,7) else 0)

train['day']=train['context_timestamp'].apply(lambda x:int(x[6:8]))
train['hour']=train['context_timestamp'].apply(lambda x:int(x[8:10]))
train['is_am']=train['hour'].apply(lambda x:1 if x<12 else 0)

test['day']=test['context_timestamp'].apply(lambda x:int(x[6:8]))
test['hour']=test['context_timestamp'].apply(lambda x:int(x[8:10]))
test['is_am']=test['hour'].apply(lambda x:1 if x<12 else 0)

# train.drop('context_timestamp',axis=1,inplace=True)
# test.drop('context_timestamp',axis=1,inplace=True)


# 将double切换为float
train[['shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description']]=train[['shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description']].astype(float)
test[['shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description']]=test[['shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description']].astype(float)



# 将商品属性切分
# train['prop']=train['item_property_list'].apply(split_item_category)
# train['prop1'] = train['prop'].apply(lambda x:x[0])
# train['prop2'] = train['prop'].apply(lambda x:x[1])
# train['prop3'] = train['prop'].apply(lambda x:x[2])
# test['prop']=test['item_property_list'].apply(split_item_category)
# test['prop1'] = test['prop'].apply(lambda x:x[0])
# test['prop2'] = test['prop'].apply(lambda x:x[1])
# test['prop3'] = test['prop'].apply(lambda x:x[2])
#
# train.drop(['prop','item_property_list'],axis=1,inplace=True)
# test.drop(['prop','item_property_list'],axis=1,inplace=True)
# print(train[['prop3']].describe())


# 生成训练集和预测集
train2_f = train[(train.context_timestamp>='20180919000000')&(train.context_timestamp<='20180923240000')]
train2_p= train[(train.context_timestamp>='20180924000000')&(train.context_timestamp<='20180924240000')]

train1_f = train[(train.context_timestamp>='20180918000000')&(train.context_timestamp<='20180922240000')]
train1_p = train[(train.context_timestamp>='20180923000000')&(train.context_timestamp<='20180923240000')]

train3_f = train[(train.context_timestamp>='20180920000000')&(train.context_timestamp<='20180924240000')]
train3_p = test

train1_f.to_csv('data/train1_f.csv',index=None)
train1_p.to_csv('data/train1_p.csv',index=None)
train2_f.to_csv('data/train2_f.csv',index=None)
train2_p.to_csv('data/train2_p.csv',index=None)
train3_f.to_csv('data/train3_f.csv',index=None)
train3_p.to_csv('data/train3_p.csv',index=None)
