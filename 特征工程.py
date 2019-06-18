# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:28:49 2019

@author: HuYue
"""

import pandas as pd
import numpy as np

def getDays(date1,date2):
    #得到两个日期之间相差的天数
    #参数：date1、date2，均为pandas的series序列
    #返回：天数的pandas-series序列
    
    index=np.where(date2=="\\N")[0]
    date2.iloc[index]="2020-01-01"
    date11=pd.to_datetime(date1)
    date21=pd.to_datetime(date2)
    days=(date11-date21).apply(lambda x:x.days)
    days=days.apply(lambda x: x if x>=0 else -1 )
    
    return days

def getRate(data,features):
    
    rate=data[['user_id',features]].groupby(features).count()/len(data)
    rate=rate.reset_index()
    rate.rename(columns={'user_id':'rate'}, inplace = True)
    
    return rate

def getUserInfo(train,user_info):
    
    #先剔除19年的数据
    user_info['insertdate']=pd.to_datetime(user_info['insertdate'])
    train['auditing_date']=pd.to_datetime(train['auditing_date'])
    user_info1=user_info.sort_values("insertdate").set_index('insertdate')
    user_info1=user_info1.truncate(after="2019-01").reset_index()
    
    #合并两个数据集，然后将日期不符合要求的剔除即可
    user_info_train=pd.merge(train,user_info, how='left', on=['user_id'])
    #print("len of user_info_train:{}".format(len(user_info_train)))
    #用户信息插入日期不能在用户借贷信息之后
    
    user_info_train1=user_info_train[user_info_train['auditing_date']>=user_info_train['insertdate']].reset_index()
    #print("len of user_info_train1:{}".format(len(user_info_train1)))
    #排序，取每个分组的最后一条数据，也就是最新的一条数据
    user_info_train1.sort_values(by=['user_id','listing_id'],inplace=True,ascending=True) 
    user_info_train2=user_info_train1.groupby(['user_id','listing_id']).apply(lambda i:i.iloc[-1])
    #print("len of user_info_train2:{}".format(len(user_info_train2)))
    return user_info_train2




#导入训练集
train=pd.read_csv("training/train.csv")
#导入训练集所需的用户基本信息表
user_info=pd.read_csv("training/user_info.csv")

#得到相差天数和相差天数所占的比例
train['days']=getDays(train['due_date'],train['repay_date'])
rate=getRate(train,'days')
train=pd.merge(train,rate, how='left', on=['days'])

#得到用户信息表与用户训练集合并的数据集
user_info_train=getUserInfo(train,user_info)
#%%user_info总共有186752个用户，部分有多条数据
#user_info[['user_id','age']].groupby('user_id').count()
#test=user_info[user_info['user_id']==924007]"

#男女的比例为0.66/0.33,这个数据这么好看，肯定是官方调整后的数据
#m=user_info_train[['user_id','gender']].groupby("gender").count()/len(user_info_train)
#plt=m.plot(kind='bar').get_figure()

def buildFeatures(data):
    #特征1：性别
    maplist={"男":0,"女":1}
    data['gender']=data['gender'].map(maplist)
    #特征2：年龄
    bins=[0,25,35,40,50,100]
    data['age_cut'] = pd.cut(user_info_train['age'], bins, labels=[1, 2, 3, 4, 5])
    #特征3：注册日期与标的成交日期相差月份
    data['reg_mon']=pd.to_datetime(data['reg_mon'])
    data['year']=(data['auditing_date']-data['reg_mon']).apply(lambda x:x.days)/365
    
    return data

user_info_train=buildFeatures(user_info_train)

#%%这个年龄的分布也太牛批了，25(左开右闭) 12000以上，25-35 8000个人以上 35-40 2000以上  40-50 2000以内 50左闭以上 很少了
#user_info_train.age.plot(kind='hist', bins=500),这个省份数据不知道怎么处理哎
#m=user_info_train[['id_province','days']].groupby("id_province").describe()
#m.plot(kind='bar')
#%%
#user_info_train.month.plot(kind='hist', bins=20)


#%%用户画像数据处理
user_taglist=pd.read_csv("training/user_taglist.csv")
user_info_train=user_info_train.drop(['insertdate'],axis=1)

#%%
def getUserITag(train,user_taglist):
    
    #先剔除19年的数据
    user_taglist['insertdate']=pd.to_datetime(user_taglist['insertdate'])
    user_taglist1=user_taglist.sort_values("insertdate").set_index('insertdate')
    user_taglist1=user_taglist1.truncate(after="2019-01").reset_index()
    
    #合并两个数据集，然后将日期不符合要求的剔除即可
    user_taglist_train=pd.merge(train,user_taglist, how='left', on=['user_id'])
    print("len of user_info_train:{}".format(len(user_info_train)))
    #用户信息插入日期不能在用户借贷信息之后
    
    user_taglist_train1=user_taglist_train[user_taglist_train['auditing_date']>=user_taglist_train['insertdate']].reset_index()
    print("len of user_info_train1:{}".format(len(user_taglist_train1)))
    #排序，取每个分组的最后一条数据，也就是最新的一条数据
    user_taglist_train1.sort_values(by=['user_id','listing_id'],inplace=True,ascending=True) 
    user_taglist_train2=user_taglist_train1.groupby(['user_id','listing_id']).apply(lambda i:i.iloc[-1])
    print("len of user_taglist_train2:{}".format(len(user_taglist_train2)))
    return user_taglist_train2

user_taglist_train=getUserITag(user_info_train,user_taglist)
