# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:44:05 2019

@author: huyue
"""
import pandas as pd
import numpy as np
from datetime import datetime
#保存一些方法的地方
def getFeatures(data):
    #平均提前多少天还款(如果不算逾期的话)
    data['days']=(data['due_date']-data['repay_date']).apply(lambda x:x.days)
    data['days']=data['days'].apply(lambda x: x if x>=0 else -1 )
    data['days1']=(data['auditing_date']-data['repay_date']).apply(lambda x:x.days)
    
    temp1=data[['user_id','auditing_date','days']]
    temp2=data[['user_id','auditing_date','repay_amt']]
    temp3=['user_id','auditing_date']
    data1=temp1.groupby(temp3).mean()
    

    #在下单前贷款次数
    data1['brrow_frequency']=temp1.groupby(temp3).count()
    #历史平均贷款金额
    data1['brrow_avg_money']=temp2.groupby(temp3).mean()
    #历史最大贷款金额
    data1['brrow_max_money']=temp2.groupby(temp3).max()
    #历史最小贷款金额
    data1['brrow_min_money']=temp2.groupby(temp3).min()
    
    
    #在要下单前一个月的贷款次数
    
    rule=(data['days1']<30)
    data1['month1_brrow_frequency']=temp1[rule].groupby(temp3).count()
    #历史平均贷款金额
    data1['month1_brrow_avg_money']=temp2[rule].groupby(temp3).mean()
    #历史最大贷款金额
    data1['month1_brrow_max_money']=temp2[rule].groupby(temp3).max()
    #历史最小贷款金额
    data1['month1_brrow_min_money']=temp2[rule].groupby(temp3).min()
    
    #在要下单前三个月的贷款次数
    rule=(data['days1']<90)
    data1['month3_brrow_frequency']=temp1[rule].groupby(temp3).count()
    #历史平均贷款金额
    data1['month3_brrow_avg_money']=temp2[rule].groupby(temp3).mean()
    #历史最大贷款金额
    data1['month3_brrow_max_money']=temp2[rule].groupby(temp3).max()
    #历史最小贷款金额
    data1['month3_brrow_min_money']=temp2[rule].groupby(temp3).min()

    
     #在要下单前一个月的还款次数
    rule=(data['days1']<30) & (data['days']!=-1)
    data1['month1_return_frequency']=temp1[rule].groupby(temp3).count()
    #历史平均还款金额
    data1['month1_return_avg_money']=temp2[rule].groupby(temp3).mean()
    #历史最大还款金额
    data1['month1_return_max_money']=temp2[rule].groupby(temp3).max()
    #历史最小还款金额
    data1['month1_return_min_money']=temp2[rule].groupby(temp3).min()
    
    #在要下单前三个月的还款次数
    rule=(data['days1']<90) & (data['days']!=-1)
    data1['month3_return_frequency']=temp1[rule].groupby(temp3).count()
    #历史平均还款金额
    data1['month3_return_avg_money']=temp2[rule].groupby(temp3).mean()
    #历史最大还款金额
    data1['month3_return_max_money']=temp2[rule].groupby(temp3).max()
    #历史最小还款金额
    data1['month3_return_min_money']=temp2[rule].groupby(temp3).min()
    
    
    
    
    #逾期的次数
    rule=(data['days']==-1)
    data1['overdue_frequency']=temp1[rule].groupby(temp3).count()
    #预期的平均金额
    data1['overdue_avg_money']=temp2[rule].groupby(temp3).mean()
    #逾期的最大金额
    data1['overdue__max_money']=temp2[rule].groupby(temp3).max()
    #逾期的最小金额month3_return
    data1['overdue__min_money']=temp2[rule].groupby(temp3).min()
    #data1['overdue_frequency'] = data1['overdue_frequency'].fillna(0)
    #逾期的比例
    data1['overdue_frequency_ratio']=data1['overdue_frequency']/data1['brrow_frequency']
    #计算还款日期的规律
    data['the_day_in_a_month']=data['repay_date'].dt.day
    data['the_day_in_a_week']=data['repay_date'].dt.dayofweek
    
 
    
    #当日还款次数
    rule=(data['days']==0)
    data1['day0_frequency']=temp1[rule].groupby(temp3).count()
    #当日还款平均金额
    data1['day0_avg_money']=temp2[rule].groupby(temp3).mean() 
    #当日还款最大金额
    data1['day0_max_money']=temp2[rule].groupby(temp3).max()
    #当日还款最小金额
    data1['day0_min_money']=temp2[rule].groupby(temp3).min()
    
    #还款日前三天还款的次数
    rule=(data['days']<3) & (data['days']>-1)
    data1['day3_frequency']=temp1[rule].groupby(temp3).count()
    #还款日三天还款的平均金额
    data1['day3_avg_money']=temp2[rule].groupby(temp3).mean()
    #还款日三天还款的最大金额
    data1['day3_max_money']=temp2[rule].groupby(temp3).max()
    #还款日三天还款的最小金额
    data1['day3_min_money']=temp2[rule].groupby(temp3).min()    
    
    #还款日前五天还款的次数
    rule=(data['days']<5) & (data['days']>-1)
    data1['day5_frequency']=temp1[rule].groupby(temp3).count()
    #还款日前五天还款的平均金额
    data1['day5_money']=temp2[rule].groupby(temp3).mean()
     #还款日五天还款的最大金额
    data1['day5_max_money']=temp2[rule].groupby(temp3).max()
    #还款日五天还款的最小金额
    data1['day5_min_money']=temp2[rule].groupby(temp3).min()
    
    
    
    #周五还款次数
    rule=(data['the_day_in_a_week']==4)
    data1['Friday_frequency']=data[['user_id','auditing_date','the_day_in_a_week']][rule].groupby(temp3).count()
    #平均周五还款金额
    data1['Friday_money']=temp2[rule].groupby(temp3).mean()
    #data1[['Friday_frequency','Friday_money']] = data1[['Friday_frequency','Friday_money']].fillna(0)
    
    #周末还款次数
    rule=(data['the_day_in_a_week']>4)
    data1['weekend_frequency']=data[['user_id','auditing_date','the_day_in_a_week']][rule].groupby(temp3).count()
    #平均周末还款金额
    data1['weekend_avg_money']=temp2[rule].groupby(temp3).mean()
    
     #一个月的前5天还款次数
    rule=(data['days1']>=0)&(data['days1']<5)
    data1['day0_repay_frequency']=temp1[rule].groupby(temp3).count()
    #一个月的前5天还款金额
    data1['day0_repay_frequency']=temp2[rule].groupby(temp3).count()
    
    
    
    #一个月的第5天还款次数
    rule=(data['days1']>=5)&(data['days1']<10)
    data1['day5_repay_frequency']=temp1[rule].groupby(temp3).count()
    #一个月的第5天还款金额
    data1['day5_repay_frequency']=temp2[rule].groupby(temp3).count()
    
    #一个月的第10天还款次数
    rule=(data['days1']>=10)&(data['days1']<15)
    data1['day10_repay_frequency']=temp1[rule].groupby(temp3).count()
    #一个月的第10天还款金额
    data1['day10_repay_frequency']=temp2[rule].groupby(temp3).count()
    
    
    #一个月的第15天还款次数
    rule=(data['days1']>=15)&(data['days1']<20)
    data1['day15_repay_frequency']=temp1[rule].groupby(temp3).count()
    #一个月的第10天还款金额
    data1['day15_repay_frequency']=temp2[rule].groupby(temp3).count()
    
    #一个月的第15天还款次数
    rule=(data['days1']>=20)&(data['days1']<25)
    data1['day20_repay_frequency']=temp1[rule].groupby(temp3).count()
    #一个月的第10天还款金额
    data1['day20_repay_frequency']=temp2[rule].groupby(temp3).count()
    
     #一个月的第25天还款次数
    rule=(data['days1']>=25)
    data1['day25_repay_frequency']=temp1[rule].groupby(temp3).count()
    #一个月的第10天还款金额
    data1['day25_repay_frequency']=temp2[rule].groupby(temp3).count()
    
    data1=data1.reset_index()
    data1=data1.fillna(0)
    return data1

#删除数据穿越的记录
def getProData(data):
    data['due_date']=pd.to_datetime(data['due_date'])
    data['repay_date']=pd.to_datetime( data['repay_date'])
    data['auditing_date']=pd.to_datetime( data['auditing_date'])
    tim=datetime.strptime("2200-01-01", "%Y-%m-%d")
    index1=((data['repay_date']!=tim)&(data['repay_date']<data['auditing_date']))
    index2=((data['repay_date']==tim)&(data['due_date']<data['auditing_date']))|index1
    data1=data[index2]
    return data1

#对训练数据集Y的处理
def getTrain(train):
    index=np.where(train['repay_date']=="\\N")[0]
    train['repay_date'].iloc[index]="2020-01-01"
    index=np.where(train['repay_amt']=="\\N")[0]
    train['repay_amt'].iloc[index]=0
    train['repay_date']=pd.to_datetime(train['repay_date'])
    train['auditing_date']=pd.to_datetime(train['auditing_date'])
    train['due_date']=pd.to_datetime(train['due_date'])
    train['train_days']=(train['due_date']-train['repay_date']).apply(lambda x:x.days)
    train['train_days']=train['train_days'].apply(lambda x: x if x>=0 else -1 )
    bins=[-2,-0.5,4,31]
    train['y']=pd.cut(train['train_days'], bins, labels=[0, 1, 2])
    return train

def getRatio(train):
    ratio=train[['user_id','train_days']].groupby("train_days").count()/len(train)
    ratio=ratio.reset_index()
    bins=[-2,-0.5,0.5,31]
    ratio['y']=pd.cut(ratio['train_days'], bins, labels=[0, 1, 2])
    ratio.rename(columns={"user_id":"ratio"}, inplace = True)
    return ratio


def buildFeatures(data):
    #特征1：性别
    maplist={"男":0,"女":1}
    data['gender']=data['gender'].map(maplist)
    #特征2：年龄
    bins=[0,25,35,40,50,100]
    data['age_cut'] = pd.cut(data['age'], bins, labels=[1, 2, 3, 4, 5])
    #特征3：注册日期与标的成交日期相差月份
    data['reg_mon']=pd.to_datetime(data['reg_mon'])
    data['reg_year']=(data['auditing_date']-data['reg_mon']).apply(lambda x:x.days)/365
    data= data.drop(['reg_mon','age'],axis=1)
    return data

def getUserInfo(train,user_info):
    
    #先剔除19年的数据
    user_info['insertdate']=pd.to_datetime(user_info['insertdate'])
    train['auditing_date']=pd.to_datetime(train['auditing_date'])
    user_info1=user_info.sort_values("insertdate").set_index('insertdate')
    user_info1=user_info1.truncate(after="2019-01").reset_index()
    
    #合并两个数据集，然后将日期不符合要求的剔除即可
    user_info_train=pd.merge(train,user_info, how='left', on=['user_id'])
    print("len of user_info_train:{}".format(len(user_info_train)))
    #用户信息插入日期不能在用户借贷信息之后
    
    user_info_train1=user_info_train[user_info_train['auditing_date']>=user_info_train['insertdate']]
    print("len of user_info_train1:{}".format(len(user_info_train1)))
    #排序，取每个分组的最后一条数据，也就是最新的一条数据
    user_info_train1.sort_values(by=['user_id','listing_id','insertdate'],inplace=True,ascending=True) 
    #user_info_train2=user_info_train1.groupby(['user_id','listing_id']).apply(lambda i:i.iloc[-1])
    user_info_train2=user_info_train1.drop_duplicates(["user_id","listing_id"],keep="last")
    print("len of user_info_train2:{}".format(len(user_info_train2)))
    
    #构造特征
    data=user_info_train2[['user_id', 'listing_id','auditing_date','reg_mon', 'gender', 'age']]
    data1=buildFeatures(data)
    return data1





#读入用户历史数据集
user_info=pd.read_csv("training/user_info.csv")
#读入训练集
train=pd.read_csv("training/train.csv")

