# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:08:09 2019

@author: huyue
"""
#专门用于处理用户标签数据集的
import pandas as pd

#读入用户标签数据集
user_taglist=pd.read_csv("training/user_taglist.csv")
#%%
num=[0]*6000
num_list=[num]*len(user_taglist)
#%%
num_list1=pd.DataFrame(num_list)

def getTag(loc):
    user=user_taglist.loc[3,"user_id"]
    insertdate=user_taglist.loc[3,"insertdate"]
    tags=user_taglist.loc[3,"taglist"]
    taglist=tags.split("|")
    for tag in taglist:
        num_list[(num_list['num_list']==user) & (num_list['insertdate']==insertdate)][tag]=1
#%%先剔除19年的数据
# =============================================================================
# user_taglist['insertdate']=pd.to_datetime(user_taglist['insertdate'])
# user_taglist1=user_taglist.sort_values("insertdate").set_index('insertdate')
# user_taglist1=user_taglist1.truncate(after="2019-01").reset_index()
# 
# #%%再进行分词
# m=user_taglist1.loc[3,"taglist"]
# m.split("|")
# def 
# #%%
# df=user_taglist1['taglist'].str.split("|",expand=True).stack()
# df1=df.reset_index(level=1,drop=True).rename("tag")
# #%%
# df2=df1.drop_duplicates()
# #%%
# df2.plot(kind="hist")
# 
# #%%
# df3=df1.groupby("tag").count()
# 
# =============================================================================
#%%