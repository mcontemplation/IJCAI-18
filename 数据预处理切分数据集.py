import pandas as pd
import xgboost as xgb
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import time
from datetime import date
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import gc

train = pd.read_table('C:/Users/37798/Desktop/[update] round2_ijcai_18_train_20180425/round2_train.txt',delim_whitespace=True)
test = pd.read_table('C:/Users/37798/Desktop/round2_ijcai_18_test_a_20180425/round2_ijcai_18_test_a_20180425.txt',delim_whitespace=True)

train.user_gender_id = train.user_gender_id.replace(-1,0)
test.user_gender_id = test.user_gender_id.replace(-1,0)


train.item_brand_id = train.item_brand_id.replace(-1,int(train['item_brand_id'].mode()))
test.item_brand_id = test.item_brand_id.replace(-1,int(test['item_brand_id'].mode()))



train.item_city_id = train.item_city_id.replace(-1,int(train['item_city_id'].mode()))
test.item_city_id = test.item_city_id.replace(-1,int(test['item_city_id'].mode()))


train.loc[train.item_sales_level==-1,'item_sales_level']=pd.DataFrame(train[train['item_sales_level'].isin([-1])]['item_collected_level'],columns=['item_collected_level']).reset_index()['item_collected_level']
test.loc[test.item_sales_level==-1,'item_sales_level']=pd.DataFrame(test[test['item_sales_level'].isin([-1])]['item_collected_level'],columns=['item_collected_level']).reset_index()['item_collected_level']


train.user_gender_id = train.user_gender_id.replace(-1,int(train['user_gender_id'].mode()))
test.user_gender_id = test.user_gender_id.replace(-1,int(test['user_gender_id'].mode()))


train.user_age_level = train.user_age_level.replace(-1,int(train['user_age_level'].mode()))
test.user_age_level = test.user_age_level.replace(-1,int(test['user_age_level'].mode()))


train.user_occupation_id = train.user_occupation_id.replace(-1,int(train['user_occupation_id'].mode()))
test.user_occupation_id = test.user_occupation_id.replace(-1,int(test['user_occupation_id'].mode()))


train.user_star_level = train.user_star_level.replace(-1,int(train['user_star_level'].mode()))
test.user_star_level = test.user_star_level.replace(-1,int(test['user_star_level'].mode()))

train.shop_review_positive_rate = train.shop_review_positive_rate.replace(-1,int(train[train.shop_review_positive_rate!=-1]['shop_review_positive_rate'].mode()))
test.shop_review_positive_rate = test.shop_review_positive_rate.replace(-1,int(test[test.shop_review_positive_rate!=-1]['shop_review_positive_rate'].mode()))


train.shop_score_service = train.shop_score_service.replace(-1,int(train[train.shop_score_service!=-1]['shop_score_service'].mode()))
test.shop_score_service = test.shop_score_service.replace(-1,int(test[test.shop_score_service!=-1]['shop_score_service'].mode()))


train.shop_score_delivery = train.shop_score_delivery.replace(-1,int(train[train.shop_score_delivery!=-1]['shop_score_delivery'].mode()))
test.shop_score_delivery = test.shop_score_delivery.replace(-1,int(test[test.shop_score_delivery!=-1]['shop_score_delivery'].mode()))

train.shop_score_description = train.shop_score_description.replace(-1,int(train[train.shop_score_description!=-1]['shop_score_description'].mode()))
test.shop_score_description = test.shop_score_description.replace(-1,int(test[test.shop_score_description!=-1]['shop_score_description'].mode()))


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
def ustar1(x):
    if(x==3000)|(x==3001)|(x==3008):
        return 'us1'
    if (x>=3002)&(x<=3007):
        return 'us2'
    return 'us3'
def uage1(x):
    if(x==1001)|(x==1000)|(x==1007):
        return 'ua1'
    if(x==1002)|(x==1005)|(x==1006):
        return 'ua2'
    return 'ua3'
def ipri1(x):
    if(x==0)|(x==1)|(x==2)|(x==3)|(x==9)|(x==10)|(x==11):
        return 'ip1'
    if(x==4)|(x==5)|(x==8):
        return 'ip2'
    return 'ip3'
def isale1(x):
    if(x==7.0)|(x==8.0)|(x==9.0)|(x==10.0)|(x==15.0)|(x==16.0):
        return 'is2'
    if(x==11.0)|(x==12.0)|(x==13.0)|(x==14.0):
        return 'is3'
    return 'is1'
def myime(x):
    if (x>=0)&(x<=6):
        return 'lingchen'
    elif(x>6)&(x<=18):
        return 'baitian'
    else:
        return 'yejian'
# train['ustar'] = train['user_star_level'].apply(ustar1)
# train['uage'] = train['user_age_level'].apply(uage1)
# train['iprice'] = train['item_price_level'].apply(ipri1)
# ustar = pd.get_dummies(train['ustar'])
# train = pd.concat([train,ustar],axis=1)
# uage = pd.get_dummies(train['uage'])
# train = pd.concat([train,uage],axis=1)
# iprice = pd.get_dummies(train['iprice'])
# train = pd.concat([train,iprice],axis=1)
train['timeduan'] = train['hour'].apply(myime)



# test['ustar'] = test['user_star_level'].apply(ustar1)
# test['uage'] = test['user_age_level'].apply(uage1)
# test['iprice'] = test['item_price_level'].apply(ipri1)
# ustar = pd.get_dummies(test['ustar'])
# test = pd.concat([test,ustar],axis=1)
# uage = pd.get_dummies(test['uage'])
# test = pd.concat([test,uage],axis=1)
# iprice = pd.get_dummies(test['iprice'])
# test = pd.concat([test,iprice],axis=1)
test['timeduan'] = test['hour'].apply(myime)

print(train.context_timestamp.min())
print(train.context_timestamp.max())
# 生成训练集和预测集
train1_f = train[(train.context_timestamp>='20180831000000')&(train.context_timestamp<='20180903240000')]
train1_p = train[(train.context_timestamp>='20180904120000')&(train.context_timestamp<='20180904240000')]
train1_f.to_csv('data/train1_f.csv',index=None)
train1_p.to_csv('data/train1_p.csv',index=None)
del train1_f
del train1_p
gc.collect()


train2_f = train[(train.context_timestamp>='20180901000000')&(train.context_timestamp<='20180904240000')]
train2_p= train[(train.context_timestamp>='20180905120000')&(train.context_timestamp<='20180905240000')]
train2_f.to_csv('data/train2_f.csv',index=None)
train2_p.to_csv('data/train2_p.csv',index=None)
del train2_f
del train2_p
gc.collect()

train4_f = train[(train.context_timestamp>='20180902000000')&(train.context_timestamp<='20180905240000')]
train4_p = train[(train.context_timestamp>='20180906120000')&(train.context_timestamp<='20180906240000')]
train4_f.to_csv('data/train4_f.csv',index=None)
train4_p.to_csv('data/train4_p.csv',index=None)
del train4_f
del train4_p
gc.collect()

# train5_f = train[(train.context_timestamp>='20180918000000')&(train.context_timestamp<='20180920240000')]
# train5_p = train[(train.context_timestamp>='20180921000000')&(train.context_timestamp<='20180921240000')]

train3_f = train[(train.context_timestamp>='20180903000000')&(train.context_timestamp<='20180906240000')]
train3_p = test
train3_f.to_csv('data/train3_f.csv',index=None)
train3_p.to_csv('data/train3_p.csv',index=None)
del train3_f
del train3_p
gc.collect()


t = train[(train.context_timestamp>='20180831000000')&(train.context_timestamp<='20180831240000')&(train.is_trade==1)]
print(t.shape[0])
t = train[(train.context_timestamp>='20180901000000')&(train.context_timestamp<='20180901240000')&(train.is_trade==1)]
print(t.shape[0])
t = train[(train.context_timestamp>='20180902000000')&(train.context_timestamp<='20180902240000')&(train.is_trade==1)]
print(t.shape[0])
t = train[(train.context_timestamp>='20180903000000')&(train.context_timestamp<='20180903240000')&(train.is_trade==1)]
print(t.shape[0])
t = train[(train.context_timestamp>='20180904000000')&(train.context_timestamp<='20180904240000')&(train.is_trade==1)]
print(t.shape[0])
t = train[(train.context_timestamp>='20180905000000')&(train.context_timestamp<='20180905240000')&(train.is_trade==1)]
print(t.shape[0])
t = train[(train.context_timestamp>='20180906000000')&(train.context_timestamp<='20180906240000')&(train.is_trade==1)]
print(t.shape[0])




# train5_f.to_csv('data/train5_f.csv',index=None)
# train5_p.to_csv('data/train5_p.csv',index=None)
# train.to_csv('data/traint.csv',index=None)


