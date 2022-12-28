#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 19:56:26 2022

@author: fabioribeiro
"""

import seaborn as sns
sns.set_theme(style="whitegrid")
# import data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats.mstats import winsorize



PER_data = pd.read_excel(r'/Users/fabioribeiro/Documents/Master/QARM2/project/data.xlsx', 'PER')
PBR_data = pd.read_excel(r'/Users/fabioribeiro/Documents/Master/QARM2/project/data.xlsx', 'Pricebook')
FCF_data = pd.read_excel(r'/Users/fabioribeiro/Documents/Master/QARM2/project/data4.xlsx', sheet_name='FCF',index_col=0).T
GPM_data = pd.read_excel(r'/Users/fabioribeiro/Documents/Master/QARM2/project/data.xlsx', sheet_name = 'Gross profit margin',parse_dates = True) #Gross profit margin 
ROA_data = pd.read_excel(r'/Users/fabioribeiro/Documents/Master/QARM2/project/data.xlsx', sheet_name = 'Sheet4',parse_dates = True) #Return on Assets
TA_data = pd.read_excel(r'/Users/fabioribeiro/Documents/Master/QARM2/project/data.xlsx', sheet_name = 'Sheet9',parse_dates = True) #Total Assets

GPM_data.set_index(GPM_data.iloc[:,0], inplace=True)
GPM_data.pop("Name")
GPM_data =GPM_data.astype(float)

ROA_data.set_index(ROA_data.iloc[:,0], inplace=True)
ROA_data.pop("Name")
ROA_data =ROA_data.astype(float)



TA_data=TA_data.replace(0, 'nan')
TA_data.set_index(TA_data.iloc[:,0], inplace=True)
TA_data.pop("Name")
TA_data =TA_data.astype(float)



new_header = PER_data.iloc[0] #grab the first row for the header
PER_data = PER_data[1:] #take the data less the header row
PER_data.columns = new_header

PBR_data = PBR_data[1:] #take the data less the header row
PBR_data.columns = new_header


PER_data =pd.DataFrame.transpose(PER_data)
PBR_data =pd.DataFrame.transpose(PBR_data)

new_header = PER_data.iloc[0] #grab the first row for the header
PER_data = PER_data[1:] #take the data less the header row
PER_data.columns = new_header

PBR_data = PBR_data[1:] #take the data less the header row
PBR_data.columns = new_header

PER_data =PER_data.astype(float)
PBR_data=PBR_data.astype(float)
PER_data1 = PER_data
PER_data = PER_data.pow(-1)#inverse the value of our data since we wanna long the lowest value
PBR_data = PBR_data.pow(-1)
FCF_data=FCF_data.loc[PER_data.index]
FCF_data=FCF_data.loc[:,PER_data.columns]




PER_data = np.where(PER_data.isnull(), np.nan, winsorize(np.ma.masked_invalid(PER_data),limits=(0.01,0.01)))
PBR_data = np.where(PBR_data.isnull(), np.nan, winsorize(np.ma.masked_invalid(PBR_data),limits=(0.01,0.01)))


PER_data = pd.DataFrame(PER_data)
PBR_data = pd.DataFrame(PBR_data)

PER_data.set_index(PER_data1.index, inplace=True)
PER_data.columns = new_header
PBR_data.set_index(PER_data1.index, inplace=True)
PBR_data.columns = new_header



Price_data = pd.read_excel(r'/Users/fabioribeiro/Documents/Master/QARM2/project/Price_FCF.xlsx', sheet_name='Price',index_col=0)
Price_data = Price_data.T
Price_data = Price_data.resample('MS').first()
#Price_data.shape
returns = Price_data.pct_change().dropna(how="all")
returns = returns.T
returns = returns.loc[:, :'2022-10-31']
returns.fillna(0, inplace=True)
returns=returns.loc[PER_data.index]


Price_data1=Price_data.T
Price_data1=Price_data1.loc[PER_data.index]
Price_data1=Price_data1.loc[:,PER_data.columns]
FCF_data=FCF_data/Price_data1
FCF_data = np.where(FCF_data.isnull(), np.nan, winsorize(np.ma.masked_invalid(FCF_data),limits=(0.01,0.01)))
FCF_data = pd.DataFrame(FCF_data)
FCF_data.set_index(PER_data1.index, inplace=True)
FCF_data.columns = new_header

GPM_data = np.where(GPM_data.isnull(), np.nan, winsorize(np.ma.masked_invalid(GPM_data),limits=(0.01,0.01)))
ROA_data = np.where(ROA_data.isnull(), np.nan, winsorize(np.ma.masked_invalid(ROA_data),limits=(0.01,0.01)))
TA_data = np.where(TA_data.isnull(), np.nan, winsorize(np.ma.masked_invalid(TA_data),limits=(0.01,0.01)))


GPM_data = pd.DataFrame(GPM_data)
TA_data = pd.DataFrame(TA_data)
ROA_data = pd.DataFrame(ROA_data)

GPM_data.set_index(PER_data1.columns, inplace=True)
GPM_data.columns = PER_data1.index
ROA_data.set_index(PER_data1.columns, inplace=True)
ROA_data.columns = PER_data1.index
TA_data.set_index(PER_data1.columns, inplace=True)
TA_data.columns = PER_data1.index


GPM_data=GPM_data.T
ROA_data=ROA_data.T
TA_data = TA_data.T
# calculate average std for our data to get zscore to properly rank them

class zscore(object):
    """create a zscore for the serie"""
    def __init__(self, data):
        self.data=data
        
        
        
        
    
    def score(self):
        mean =np.zeros((1,261))
        mean = pd.DataFrame(mean)
        mean.columns=new_header
        for i in range(len(self.data.T)):
            mean.iloc[0,i]=self.data.iloc[:,i].mean()
        std=np.zeros((1,261))
        std = pd.DataFrame(std)
        std.columns = new_header
        for i in range(len(self.data.T)):
            std.iloc[0,i] = self.data.iloc[:,i].std()
        zscore=np.zeros((502,261))
        zscore = pd.DataFrame(zscore)
        zscore.set_index(PER_data.index, inplace=True)
        zscore.columns = new_header
        for i in range(len(self.data.T)):
            for j in range(len(self.data)):
                zscore.iloc[j,i] = (self.data.iloc[j,i]-mean.iloc[0,i])/std.iloc[0,i]
        self.zscore=zscore
        return zscore

        
PER=zscore(PER_data)        
PER_zscore = PER.score()

PBR=zscore(PBR_data)
PBR_zscore=PBR.score()

FCF=zscore(FCF_data)
FCF_zscore =FCF.score()

value_zscore=np.zeros((502,261))
value_zscore = pd.DataFrame(value_zscore)
value_zscore.set_index(PBR_data.index, inplace=True)
value_zscore.columns = new_header
for i in range(261):
    for j in range(502):
        value_zscore.iloc[j,i] = FCF_zscore.iloc[j,i]+PER_zscore.iloc[j,i]+PBR_zscore.iloc[j,i]


GPM=zscore(GPM_data)
GPM_zscore=GPM.score()

ROA=zscore(ROA_data)
ROA_zscore=ROA.score()

TA = zscore(TA_data)
TA_zscore = TA.score()


quality_zscore=np.zeros((502,261))
quality_zscore = pd.DataFrame(quality_zscore)
quality_zscore.set_index(PER_data.index, inplace=True)
quality_zscore.columns = new_header
for i in range(261):
    for j in range(502):
        quality_zscore.iloc[j,i] = GPM_zscore.iloc[j,i]+ROA_zscore.iloc[j,i]+TA_zscore.iloc[j,i]
        




integ_zscore=np.zeros((502,261))
integ_zscore = pd.DataFrame(integ_zscore)
integ_zscore.set_index(PER_data.index, inplace=True)
integ_zscore.columns = new_header
for i in range(261):
    for j in range(502):
        integ_zscore.iloc[j,i] = 0.5*quality_zscore.iloc[j,i]+ 0.5*value_zscore.iloc[j,i]


integ_rank=np.zeros((502,261))
integ_rank = pd.DataFrame(integ_rank)
integ_rank.set_index(PBR_data.index, inplace=True)
integ_rank.columns = new_header
for i in range(261):
    integ_rank.iloc[:,i] = integ_zscore.iloc[:,i].rank()

integ_Top_Up_30 = []
for i in range(252):
    integ_Top_Up_30.append(integ_rank.iloc[:,i].nlargest(30))


integ_Top_Down_30 = []
for i in range(252):
    integ_Top_Down_30.append(integ_rank.iloc[:,i].nsmallest(30))
    
#Equally Weighted integrated without tc
iW_up=np.zeros((30,252))
iW_down = np.zeros((30,252))
iW_up = pd.DataFrame(iW_up)
iW_down  = pd.DataFrame(iW_down)
i_returns_asset=np.zeros((30,252))
i_returns_asset = pd.DataFrame(i_returns_asset)
i_returns_port=np.zeros((1,252))
i_returns_port = pd.DataFrame(i_returns_port)
i_cum_returns=np.zeros((1,252))
i_cum_returns = pd.DataFrame(i_cum_returns)

for i in range (252):
    iTop_Up_30names = integ_Top_Up_30[i].keys()
    iTop_Down_30names = integ_Top_Down_30[i].keys()
    for j in range (30):
        iW_up.iloc[j,i]=1/30
        iW_down.iloc[j,i]=1/30
        
        i_returns_asset.iloc[j,i]= iW_up.iloc[j,i]*returns.loc[iTop_Up_30names].iloc[j,i+1]-(iW_down.iloc[j,i]*returns.loc[iTop_Down_30names].iloc[j,i+1])
                                                                                
    i_returns_port.iloc[0,i]=i_returns_asset.iloc[:,i].sum()
    
    
    if i==0 :
        i_cum_returns.iloc[0,i]=1+i_returns_port.iloc[0,i]
        
    else :
        i_cum_returns.iloc[0,i]=(1+i_returns_port.iloc[0,i])*i_cum_returns.iloc[0,i-1]
        
            
plt.plot(i_cum_returns.iloc[0,:])

#rank weighted integrated portfolio without tc
iW_up=np.zeros((30,252))
iW_down = np.zeros((30,252))
iW_up = pd.DataFrame(iW_up)
iW_down  = pd.DataFrame(iW_down)
i_returns_asset=np.zeros((30,252))
i_returns_asset = pd.DataFrame(i_returns_asset)
i_returns_port=np.zeros((1,252))
i_returns_port = pd.DataFrame(i_returns_port)
i_cum_returns=np.zeros((1,252))
i_cum_returns = pd.DataFrame(i_cum_returns)

for i in range (252):
    iTop_Up_30names = integ_Top_Up_30[i].keys()
    iTop_Down_30names = integ_Top_Down_30[i].keys()
    for j in range (30):
        iW_up.iloc[j,i]=(j+1)/(30*(30+1)/2)
        iW_down.iloc[j,i]=((30-j))/(30*(30+1)/2)
        
        i_returns_asset.iloc[j,i]= iW_up.iloc[j,i]*returns.loc[iTop_Up_30names].iloc[j,i+1]-(iW_down.iloc[j,i]*returns.loc[iTop_Down_30names].iloc[j,i+1])
                                                                                
    i_returns_port.iloc[0,i]=i_returns_asset.iloc[:,i].sum()
    
    
    if i==0 :
        i_cum_returns.iloc[0,i]=1+i_returns_port.iloc[0,i]
        
    else :
        i_cum_returns.iloc[0,i]=(1+i_returns_port.iloc[0,i])*i_cum_returns.iloc[0,i-1]
        
            
plt.plot(i_cum_returns.iloc[0,:])


#adding transaction costs

TO_i=np.zeros((1,251))
TO_i = pd.DataFrame(TO_i)
for i in range (251):
    
    
    for j in range (30):
        if integ_Top_Up_30[i+1].keys()[j] in integ_Top_Up_30[i].keys():
            TO_i.iloc[0,i]=TO_i.iloc[0,i]
        else:
            TO_i.iloc[0,i]=TO_i.iloc[0,i]+0.08
            
        if integ_Top_Down_30[i+1].keys()[j] in integ_Top_Down_30[i].keys():
            TO_i.iloc[0,i]=TO_i.iloc[0,i]
        else:
            TO_i.iloc[0,i]=TO_i.iloc[0,i]+0.08
TO_i.mean(axis=1)*12


TC_i=0.002*TO_i

TC_i.insert(0,"",0)


TC_i.columns=FCF_data.iloc[:,0:252].columns

TO1_i=np.zeros((1,251))
TO1_i = pd.DataFrame(TO1_i)
for i in range (251):
    my_list_up=integ_Top_Up_30[i].keys()    
    my_list_up=my_list_up.tolist()
    my_list_down=integ_Top_Down_30[i].keys()    
    my_list_down=my_list_down.tolist()        
    
    for j in range (25):
        if my_list_up.count(integ_Top_Up_30[i+1].keys()[j])>0 and integ_Top_Up_30[i+1].keys()[j]==integ_Top_Up_30[i].keys()[j]:
            TO1_i.iloc[0,i]=TO1_i.iloc[0,i]
        elif my_list_up.count(integ_Top_Up_30[i+1].keys()[j])>0 and integ_Top_Up_30[i+1].keys()[j]!=integ_Top_Up_30[i].keys()[j]:
            TO1_i.iloc[0,i]=TO1_i.iloc[0,i]+(iW_up.iloc[my_list_up.index(integ_Top_Up_30[i+1].keys()[j]),i+1]-iW_up.iloc[j,i])*2
        else:
            TO1_i.iloc[0,i]=TO1_i.iloc[0,i]+iW_up.iloc[j,i]
            
            
        if my_list_down.count(integ_Top_Down_30[i+1].keys()[j])>0 and integ_Top_Down_30[i+1].keys()[j]==integ_Top_Down_30[i].keys()[j]:
            TO1_i.iloc[0,i]=TO_i.iloc[0,i]
        elif my_list_down.count(integ_Top_Down_30[i+1].keys()[j])>0 and integ_Top_Down_30[i+1].keys()[j]!=integ_Top_Down_30[i].keys()[j]:
            TO1_i.iloc[0,i]=TO1_i.iloc[0,i]+(iW_down.iloc[my_list_down.index(integ_Top_Down_30[i+1].keys()[j]),i+1]-iW_down.iloc[j,i])*2
        else:
            TO1_i.iloc[0,i]=TO_i.iloc[0,i]+iW_down.iloc[j,i]

TC1_i=0.002*TO1_i
TC1_i.insert(0,"",0)
TC1_i.columns=FCF_data.iloc[:,0:252].columns

TO1_i.mean(axis=1)*12

#Equally Weighted integrated with transaction costs
iW_up=np.zeros((30,252))
iW_down = np.zeros((30,252))
iW_up = pd.DataFrame(iW_up)
iW_down  = pd.DataFrame(iW_down)
i_returns_asset=np.zeros((30,252))
i_returns_asset = pd.DataFrame(i_returns_asset)
i_returns_port=np.zeros((1,252))
i_returns_port = pd.DataFrame(i_returns_port)
i_cum_returns=np.zeros((1,252))
i_cum_returns = pd.DataFrame(i_cum_returns)
returns_long_ai =np.zeros((30,252))
returns_long_ai = pd.DataFrame(returns_long_ai)
returns_longi =np.zeros((1,252))
returns_longi = pd.DataFrame(returns_longi)
returns_short_ai =np.zeros((30,252))
returns_short_ai = pd.DataFrame(returns_short_ai)
returns_shorti =np.zeros((1,252))
returns_shorti = pd.DataFrame(returns_shorti)
cum_returns_longi=np.zeros((1,252))
cum_returns_longi = pd.DataFrame(cum_returns_longi)
cum_returns_shorti=np.zeros((1,252))
cum_returns_shorti = pd.DataFrame(cum_returns_shorti)
for i in range (252):
    iTop_Up_30names = integ_Top_Up_30[i].keys()
    iTop_Down_30names = integ_Top_Down_30[i].keys()
    for j in range (30):
        iW_up.iloc[j,i]=1/30
        iW_down.iloc[j,i]=1/30
        
        i_returns_asset.iloc[j,i]= iW_up.iloc[j,i]*returns.loc[iTop_Up_30names].iloc[j,i+1]-iW_down.iloc[j,i]*returns.loc[iTop_Down_30names].iloc[j,i+1]
        returns_long_ai.iloc[j,i]= iW_up.iloc[j,i]*returns.loc[iTop_Up_30names].iloc[j,i+1]
        returns_short_ai.iloc[j,i]= -1*iW_up.iloc[j,i]*returns.loc[iTop_Down_30names].iloc[j,i+1]                                                                             
    i_returns_port.iloc[0,i]=i_returns_asset.iloc[:,i].sum()-TC_i.iloc[0,i]
    returns_longi.iloc[0,i]=returns_long_ai.iloc[:,i].sum()-TC_i.iloc[0,i]/2
    returns_shorti.iloc[0,i]=returns_short_ai.iloc[:,i].sum()-TC_i.iloc[0,i]/2
    if i==0 :
        i_cum_returns.iloc[0,i]=1+i_returns_port.iloc[0,i]
        cum_returns_longi.iloc[0,i]=1+returns_longi.iloc[0,i]
        cum_returns_shorti.iloc[0,i]=1+returns_shorti.iloc[0,i]
    else :
        i_cum_returns.iloc[0,i]=(1+i_returns_port.iloc[0,i])*i_cum_returns.iloc[0,i-1]
        cum_returns_longi.iloc[0,i]=(1+returns_longi.iloc[0,i])*cum_returns_longi.iloc[0,i-1]
        cum_returns_shorti.iloc[0,i]=(1+returns_shorti.iloc[0,i])*cum_returns_shorti.iloc[0,i-1]      
plt.plot(i_cum_returns.iloc[0,:])


#stats for integrated EW
from scipy.stats import skew, kurtosis
import pandas_datareader as web
risk_free = web.DataReader('DTB3', 'fred', start='2001-1-1', end='2021-12-01')
risk_free = risk_free.resample('MS').first()/100
print(risk_free)

risk_free.columns = ['RF']
return_df= i_returns_port.T
return_df.index=risk_free.index
excessReturn = return_df.sub(risk_free['RF'], axis=0)
geometric_mean = ((1+return_df).cumprod().iloc[-1]**(1/len(return_df))-1)
meanReturn = return_df.mean()*12
volReturn = return_df.std()*(12**0.5)
SR = excessReturn.mean()*12/volReturn
skewness=return_df.skew()
kurtosis=return_df.kurtosis()

esohfséfu










        