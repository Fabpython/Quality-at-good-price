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
MC_data = pd.read_excel(r'/Users/fabioribeiro/Documents/Master/QARM2/project/data4.xlsx', sheet_name='MC',index_col=0).T
GPM_data = pd.read_excel(r'/Users/fabioribeiro/Documents/Master/QARM2/project/data.xlsx', sheet_name = 'Gross profit margin',parse_dates = True) #Gross profit margin 
ROA_data = pd.read_excel(r'/Users/fabioribeiro/Documents/Master/QARM2/project/data.xlsx', sheet_name = 'Sheet4',parse_dates = True) #Return on Assets
OTH_data = pd.read_excel(r'/Users/fabioribeiro/Documents/Master/QARM2/project/data.xlsx', sheet_name = 'Sheet5',parse_dates = True) #Increase/Decrease OTH Accruals
WC_data = pd.read_excel(r'/Users/fabioribeiro/Documents/Master/QARM2/project/data.xlsx', sheet_name = 'Sheet6',parse_dates = True) #Working capital
DPS_data = pd.read_excel(r'/Users/fabioribeiro/Documents/Master/QARM2/project/data.xlsx', sheet_name = 'Sheet7',parse_dates = True) #Dividend earnout per share
DE_data = pd.read_excel(r'/Users/fabioribeiro/Documents/Master/QARM2/project/data.xlsx', sheet_name = 'DER',parse_dates = True) #Debt to common equity
TA_data = pd.read_excel(r'/Users/fabioribeiro/Documents/Master/QARM2/project/data.xlsx', sheet_name = 'Sheet9',parse_dates = True) #Total Assets

GPM_data.set_index(GPM_data.iloc[:,0], inplace=True)
GPM_data.pop("Name")
GPM_data =GPM_data.astype(float)

ROA_data.set_index(ROA_data.iloc[:,0], inplace=True)
ROA_data.pop("Name")
ROA_data =ROA_data.astype(float)


DE_data=DE_data.replace(0, 'nan')
DE_data.set_index(DE_data.iloc[:,0], inplace=True)
DE_data.pop("Name")
DE_data =DE_data.astype(float)

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
MC_data=MC_data.astype(float)
PER_data1 = PER_data
PER_data = PER_data.pow(-1)#inverse the value of our data since we wanna long the lowest value
PBR_data = PBR_data.pow(-1)
FCF_data=FCF_data.loc[PER_data.index]
FCF_data=FCF_data.loc[:,PER_data.columns]
MC_data=MC_data.loc[PER_data.index]
MC_data=MC_data.loc[:,PER_data.columns]



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

        
         
    
              
   
        
    
        
        

            
    

    
    
    
    
    
PER_average=np.zeros((1,261))
PER_average = pd.DataFrame(PER_average)
PER_average.columns = new_header
for i in range(261):
    PER_average.iloc[0,i] = PER_data.iloc[:,i].mean()

PER_std=np.zeros((1,261))
PER_std = pd.DataFrame(PER_std)
PER_std.columns = new_header
for i in range(261):
    PER_std.iloc[0,i] = PER_data.iloc[:,i].std()

PER_zscore=np.zeros((502,261))
PER_zscore = pd.DataFrame(PER_zscore)
PER_zscore.set_index(PER_data.index, inplace=True)
PER_zscore.columns = new_header
for i in range(261):
    for j in range(502):
        PER_zscore.iloc[j,i] = (PER_data.iloc[j,i]-PER_average.iloc[0,i])/PER_std.iloc[0,i]



PBR_average=np.zeros((1,261))
PBR_average = pd.DataFrame(PBR_average)
PBR_average.columns = new_header
for i in range(261):
    PBR_average.iloc[0,i] = PBR_data.iloc[:,i].mean()

PBR_std=np.zeros((1,261))
PBR_std = pd.DataFrame(PBR_std)
PBR_std.columns = new_header
for i in range(261):
    PBR_std.iloc[0,i] = PBR_data.iloc[:,i].std()

PBR_zscore=np.zeros((502,261))
PBR_zscore = pd.DataFrame(PBR_zscore)
PBR_zscore.set_index(PBR_data.index, inplace=True)
PBR_zscore.columns = new_header
for i in range(261):
    for j in range(502):
        PBR_zscore.iloc[j,i] = (PBR_data.iloc[j,i]-PBR_average.iloc[0,i])/PBR_std.iloc[0,i]
    
FCF_average=np.zeros((1,261))
FCF_average = pd.DataFrame(FCF_average)
FCF_average.columns = new_header
for i in range(261):
    FCF_average.iloc[0,i] = FCF_data.iloc[:,i].mean()

FCF_std=np.zeros((1,261))
FCF_std = pd.DataFrame(FCF_std)
FCF_std.columns = new_header
for i in range(261):
    FCF_std.iloc[0,i] = FCF_data.iloc[:,i].std()


FCF_zscore=np.zeros((502,261))
FCF_zscore = pd.DataFrame(FCF_zscore)
FCF_zscore.set_index(PBR_data.index, inplace=True)
FCF_zscore.columns = new_header
for i in range(261):
    for j in range(502):
        FCF_zscore.iloc[j,i] = (FCF_data.iloc[j,i]-FCF_average.iloc[0,i])/FCF_std.iloc[0,i]







value_zscore=np.zeros((502,261))
value_zscore = pd.DataFrame(value_zscore)
value_zscore.set_index(PBR_data.index, inplace=True)
value_zscore.columns = new_header
for i in range(261):
    for j in range(502):
        value_zscore.iloc[j,i] = FCF_zscore.iloc[j,i]+PER_zscore.iloc[j,i]+PBR_zscore.iloc[j,i]
        
value_rank=np.zeros((502,261))
value_rank = pd.DataFrame(value_rank)
value_rank.set_index(PER_data.index, inplace=True)
value_rank.columns = new_header
for i in range(261):
    value_rank.iloc[:,i] = value_zscore.iloc[:,i].rank()



#Store the best 25 firms and worse 25 firms for each month
Top_Up_25 = []
for i in range(252):
    Top_Up_25.append(value_rank.iloc[:,i].sort_values().dropna()[-25:])


Top_Down_25 = []
for i in range(252):
    Top_Down_25.append(value_rank.iloc[:,i].sort_values().dropna()[0:25])
    

    

#Compute return




#Rf = web.DataReader("DTB3", "fred",)
#risk_free = web.DataReader('DTB3', 'fred', start='2001-1-1', end='2021-12-31')
#risk_free = risk_free.resample('MS').first()/1200
#risk_free.columns = ['RF']
#risk_free = risk_free.values.tolist()




    
W_up=np.zeros((25,252))
W_down = np.zeros((25,252))
W_up = pd.DataFrame(W_up)
W_down  = pd.DataFrame(W_down)
returns_asset=np.zeros((25,252))
returns_asset = pd.DataFrame(returns_asset)
returns_port=np.zeros((1,252))
returns_port = pd.DataFrame(returns_port)
cum_returns=np.zeros((1,252))
cum_returns = pd.DataFrame(cum_returns)
returns_long_a =np.zeros((25,252))
returns_long_a = pd.DataFrame(returns_long_a)
returns_long =np.zeros((25,252))
returns_long = pd.DataFrame(returns_long)
returns_short_a =np.zeros((25,252))
returns_short_a = pd.DataFrame(returns_short_a)
returns_short =np.zeros((25,252))
returns_short = pd.DataFrame(returns_short)
cum_returns_long=np.zeros((1,252))
cum_returns_long = pd.DataFrame(cum_returns_long)
cum_returns_short=np.zeros((1,252))
cum_returns_short = pd.DataFrame(cum_returns_short)
for i in range (252):
    Top_Up_25names = Top_Up_25[i].keys()
    Top_Down_25names = Top_Down_25[i].keys()
    for j in range (25):
        W_up.iloc[j,i]=1/25
        W_down.iloc[j,i]=1/25
        
        returns_asset.iloc[j,i]= W_up.iloc[j,i]*returns.loc[Top_Up_25names].iloc[j,i+1]-W_down.iloc[j,i]*returns.loc[Top_Down_25names].iloc[j,i+1]
        returns_long_a.iloc[j,i]= W_up.iloc[j,i]*returns.loc[Top_Up_25names].iloc[j,i+1]
        returns_short_a.iloc[j,i]=-W_down.iloc[j,i]*returns.loc[Top_Down_25names].iloc[j,i+1]                                                                              
    returns_port.iloc[0,i]=returns_asset.iloc[:,i].sum()
    returns_long.iloc[0,i]=returns_long_a.iloc[:,i].sum()
    returns_short.iloc[0,i]=returns_short_a.iloc[:,i].sum()
    if i==0 :
        cum_returns.iloc[0,i]=1+returns_port.iloc[0,i]
        cum_returns_long.iloc[0,i]=1+returns_long.iloc[0,i]
        cum_returns_short.iloc[0,i]=1+returns_short.iloc[0,i]
        
    else :
        cum_returns.iloc[0,i]=(1+returns_port.iloc[0,i])*cum_returns.iloc[0,i-1]
        cum_returns_long.iloc[0,i]=(1+returns_long.iloc[0,i])*cum_returns_long.iloc[0,i-1]  
        cum_returns_short.iloc[0,i]=(1+returns_short.iloc[0,i])*cum_returns_short.iloc[0,i-1]          


    
plt.plot(cum_returns.iloc[0,:])
plt.plot(cum_returns_long.iloc[0,:])
plt.plot(cum_returns_short.iloc[0,:])


W_up_exp=np.zeros((25,252))
W_down_exp = np.zeros((25,261))
W_up_exp = pd.DataFrame(W_up_exp)
W_down_exp  = pd.DataFrame(W_down_exp)
returns_asset1=np.zeros((25,252))
returns_asset1 = pd.DataFrame(returns_asset1)
returns_port1=np.zeros((1,252))
returns_port1 = pd.DataFrame(returns_port1)
cum_returns1=np.zeros((1,252))
cum_returns1 = pd.DataFrame(cum_returns1)
for i in range (252):
    Top_Up_25names = Top_Up_25[i].keys()
    Top_Down_25names = Top_Down_25[i].keys()
    for j in range (25):
        W_up_exp.iloc[j,i]=np.exp(value_zscore.loc[Top_Up_25names].iloc[j,i])/(np.exp(value_zscore.loc[Top_Up_25names].iloc[:,i]).sum())
        W_down_exp.iloc[j,i]=np.exp(value_zscore.loc[Top_Down_25names].iloc[j,i]**-1)/(np.exp(value_zscore.loc[Top_Down_25names].iloc[:,i]**-1).sum())
        returns_asset1.iloc[j,i]= W_up_exp.iloc[j,i]*returns.loc[Top_Up_25names].iloc[j,i+1]-W_down_exp.iloc[j,i]*returns.loc[Top_Down_25names].iloc[j,i+1]                                                                              
    returns_port1.iloc[0,i]=returns_asset1.iloc[:,i].sum()
    if i==0 :
        cum_returns1.iloc[0,i]=1+returns_port1.iloc[0,i]
        
    else :
        cum_returns1.iloc[0,i]=(1+returns_port1.iloc[0,i])*cum_returns1.iloc[0,i-1]
 
plt.plot(cum_returns1.iloc[0,:])

W_up_exp=np.zeros((25,252))
W_down_exp = np.zeros((25,261))
W_up_exp = pd.DataFrame(W_up_exp)
W_down_exp  = pd.DataFrame(W_down_exp)
returns_asset2=np.zeros((25,252))
returns_asset2 = pd.DataFrame(returns_asset2)
returns_port2=np.zeros((1,252))
returns_port2 = pd.DataFrame(returns_port2)
cum_returns2=np.zeros((1,252))
cum_returns2 = pd.DataFrame(cum_returns2)
returns_long_a2 =np.zeros((25,252))
returns_long_a2 = pd.DataFrame(returns_long_a2)
returns_long2 =np.zeros((25,252))
returns_long2 = pd.DataFrame(returns_long2)
returns_short_a2 =np.zeros((25,252))
returns_short_a2 = pd.DataFrame(returns_short_a2)
returns_short2 =np.zeros((25,252))
returns_short2 = pd.DataFrame(returns_short2)
cum_returns_long2=np.zeros((1,252))
cum_returns_long2 = pd.DataFrame(cum_returns_long2)
cum_returns_short2=np.zeros((1,252))
cum_returns_short2 = pd.DataFrame(cum_returns_short2)
for i in range (252):
    Top_Up_25names = Top_Up_25[i].keys()
    Top_Down_25names = Top_Down_25[i].keys()
    for j in range (25):
        W_up_exp.iloc[j,i]=(j+1)/(25*(25+1)/2)
        W_down_exp.iloc[j,i]=((25-j))/(25*(25+1)/2)
        returns_asset2.iloc[j,i]= W_up_exp.iloc[j,i]*returns.loc[Top_Up_25names].iloc[j,i+1]-W_down_exp.iloc[j,i]*returns.loc[Top_Down_25names].iloc[j,i+1]
        returns_long_a2.iloc[j,i]= W_up.iloc[j,i]*returns.loc[Top_Up_25names].iloc[j,i+1]
        returns_short_a2.iloc[j,i]=-W_down.iloc[j,i]*returns.loc[Top_Down_25names].iloc[j,i+1]                                                                               
    returns_port2.iloc[0,i]=returns_asset2.iloc[:,i].sum()
    returns_long2.iloc[0,i]=returns_long_a2.iloc[:,i].sum()
    returns_short2.iloc[0,i]=returns_short_a2.iloc[:,i].sum()
    if i==0 :
        cum_returns2.iloc[0,i]=1+returns_port2.iloc[0,i]
        cum_returns_long2.iloc[0,i]=1+returns_long2.iloc[0,i]
        cum_returns_short2.iloc[0,i]=1+returns_short2.iloc[0,i]
    else :
        cum_returns2.iloc[0,i]=(1+returns_port2.iloc[0,i])*cum_returns2.iloc[0,i-1]
        cum_returns_long2.iloc[0,i]=(1+returns_long2.iloc[0,i])*cum_returns_long2.iloc[0,i-1]  
        cum_returns_short2.iloc[0,i]=(1+returns_short2.iloc[0,i])*cum_returns_short2.iloc[0,i-1]
 
plt.plot(cum_returns2.iloc[0,:]) 
plt.plot(cum_returns_long2.iloc[0,:])
plt.plot(cum_returns_short2.iloc[0,:])  

plt.plot(returns_long2.iloc[0,:])
plt.plot(returns_short2.iloc[0,:])


benchmark_data = pd.read_excel(r'/Users/fabioribeiro/Documents/Master/QARM2/project/benchmark.xlsx', sheet_name='benchmark',index_col=0)
benchmark_data=benchmark_data.loc[PER_data.iloc[:,0:252].columns]
benchmark_cum = (1+benchmark_data).cumprod()
plt.plot(benchmark_cum)

cum_returns=cum_returns.T
cum_returns1=cum_returns1.T
cum_returns2=cum_returns2.T

cum_returns2.set_index(benchmark_cum.index, inplace=True)
cum_returns1.set_index(benchmark_cum.index, inplace=True)
cum_returns.set_index(benchmark_cum.index, inplace=True)

plt.plot(benchmark_cum, color='b', label='benchmark')
plt.plot( cum_returns, color='r', label='equally weighted')
plt.plot( cum_returns1, color='g', label='zscore weighted')
plt.plot( cum_returns2, color='y', label='rank weighted')
plt.legend();

#Farma-French factors 


import pandas_datareader as web
import statsmodels.api as sm

ff_df = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start='2001-2-1', end='2022-1-1')[0]
ff_df = ff_df[['Mkt-RF', 'SMB', 'HML']]
ff_df.index = returns.iloc[:,0:252].columns
ff_df

x = ff_df/100
y = returns_port.T
y.index=returns.iloc[:,0:252].columns

model = sm.OLS(y, x, missing='drop')
results = model.fit()
print(results.summary())




#Transaction costs


TO=np.zeros((1,251))
TO = pd.DataFrame(TO)
for i in range (251):
    
    
    for j in range (25):
        if Top_Up_25[i+1].keys()[j] in Top_Up_25[i].keys():
            TO.iloc[0,i]=TO.iloc[0,i]
        else:
            TO.iloc[0,i]=TO.iloc[0,i]+0.08
            
        if Top_Down_25[i+1].keys()[j] in Top_Down_25[i].keys():
            TO.iloc[0,i]=TO.iloc[0,i]
        else:
            TO.iloc[0,i]=TO.iloc[0,i]+0.08

TO.mean(axis=1)*12

my_list=Top_Up_25[i+1].keys()    
my_list=my_list.tolist()    
my_list.index(Top_Up_25[i].keys()[1])

my_list.count(Top_Up_25[i+1].keys()[j])

TO1=np.zeros((1,251))
TO1 = pd.DataFrame(TO1)
for i in range (251):
    my_list_up=Top_Up_25[i].keys()    
    my_list_up=my_list_up.tolist()
    my_list_down=Top_Down_25[i].keys()    
    my_list_down=my_list_down.tolist()        
    
    for j in range (25):
        if my_list_up.count(Top_Up_25[i+1].keys()[j])>0 and Top_Up_25[i+1].keys()[j]==Top_Up_25[i].keys()[j]:
            TO1.iloc[0,i]=TO1.iloc[0,i]
        elif my_list_up.count(Top_Up_25[i+1].keys()[j])>0 and Top_Up_25[i+1].keys()[j]!=Top_Up_25[i].keys()[j]:
            TO1.iloc[0,i]=TO1.iloc[0,i]+abs((W_up_exp.iloc[my_list_up.index(Top_Up_25[i+1].keys()[j]),i+1]-W_up_exp.iloc[j,i])*2)
        else:
            TO1.iloc[0,i]=TO1.iloc[0,i]+W_up_exp.iloc[j,i]
            
            
        if my_list_down.count(Top_Down_25[i+1].keys()[j])>0 and Top_Down_25[i+1].keys()[j]==Top_Down_25[i].keys()[j]:
            TO1.iloc[0,i]=TO.iloc[0,i]
        elif my_list_down.count(Top_Down_25[i+1].keys()[j])>0 and Top_Down_25[i+1].keys()[j]!=Top_Down_25[i].keys()[j]:
            TO1.iloc[0,i]=TO1.iloc[0,i]+abs((W_down_exp.iloc[my_list_down.index(Top_Down_25[i+1].keys()[j]),i+1]-W_down_exp.iloc[j,i])*2)
        else:
            TO1.iloc[0,i]=TO.iloc[0,i]+W_down_exp.iloc[j,i]
            
TO1.mean(axis=1)*12

TC1=0.002*TO1
TC=0.002*TO

TC.insert(0,"",0)
TC1.insert(0,"",0)

TC.columns=returns_port.columns
TC1.columns=returns_port.columns




W_up=np.zeros((25,252))
W_down = np.zeros((25,252))
W_up = pd.DataFrame(W_up)
W_down  = pd.DataFrame(W_down)
returns_asset=np.zeros((25,252))
returns_asset = pd.DataFrame(returns_asset)
returns_port=np.zeros((1,252))
returns_port = pd.DataFrame(returns_port)
cum_returns=np.zeros((1,252))
cum_returns = pd.DataFrame(cum_returns)
for i in range (252):
    Top_Up_25names = Top_Up_25[i].keys()
    Top_Down_25names = Top_Down_25[i].keys()
    for j in range (25):
        W_up.iloc[j,i]=1/25
        W_down.iloc[j,i]=1/25
        
        returns_asset.iloc[j,i]= W_up.iloc[j,i]*returns.loc[Top_Up_25names].iloc[j,i+1]-W_down.iloc[j,i]*returns.loc[Top_Down_25names].iloc[j,i+1]                                                                              
    returns_port.iloc[0,i]=returns_asset.iloc[:,i].sum() - TC.iloc[0,i]
    if i==0 :
        cum_returns.iloc[0,i]=1+returns_port.iloc[0,i]
    else :
        cum_returns.iloc[0,i]=(1+returns_port.iloc[0,i])*cum_returns.iloc[0,i-1]        

from scipy.stats import skew, kurtosis
import pandas_datareader as web
risk_free = web.DataReader('DTB3', 'fred', start='2001-1-1', end='2021-12-01')
risk_free = risk_free.resample('MS').first()/1200
print(risk_free)

risk_free.columns = ['RF']
return_df= returns_port.T
return_df.index=risk_free.index
excessReturn = return_df.sub(risk_free['RF'], axis=0)
geometric_mean = ((1+return_df).cumprod().iloc[-1]**(1/len(return_df))-1)
meanReturn = return_df.mean()*12
volReturn = return_df.std()*(12**0.5)
SR = excessReturn.mean()*12/volReturn
skewness=return_df.skew()
kurtosis=return_df.kurtosis()
    


W_up_exp=np.zeros((25,252))
W_down_exp = np.zeros((25,261))
W_up_exp = pd.DataFrame(W_up_exp)
W_down_exp  = pd.DataFrame(W_down_exp)
returns_asset2=np.zeros((25,252))
returns_asset2 = pd.DataFrame(returns_asset2)
returns_port2=np.zeros((1,252))
returns_port2 = pd.DataFrame(returns_port2)
cum_returns2=np.zeros((1,252))
cum_returns2 = pd.DataFrame(cum_returns2)
for i in range (252):
    Top_Up_25names = Top_Up_25[i].keys()
    Top_Down_25names = Top_Down_25[i].keys()
    for j in range (25):
        W_up_exp.iloc[j,i]=(j+1)/(25*(25+1)/2)
        W_down_exp.iloc[j,i]=((25-j))/(25*(25+1)/2)
        returns_asset2.iloc[j,i]= W_up_exp.iloc[j,i]*returns.loc[Top_Up_25names].iloc[j,i+1]-W_down_exp.iloc[j,i]*returns.loc[Top_Down_25names].iloc[j,i+1]                                                                              
    returns_port2.iloc[0,i]=returns_asset2.iloc[:,i].sum()-TC1.iloc[0,i]
    if i==0 :
        cum_returns2.iloc[0,i]=1+returns_port2.iloc[0,i]
    else :
        cum_returns2.iloc[0,i]=(1+returns_port2.iloc[0,i])*cum_returns2.iloc[0,i-1]
 
plt.plot(cum_returns2.iloc[0,:])   

risk_free.columns = ['RF']
return_df= returns_port2.T
return_df.index=risk_free.index
excessReturn = return_df.sub(risk_free['RF'], axis=0)
geometric_mean = ((1+return_df).cumprod().iloc[-1]**(1/len(return_df))-1)
meanReturn = return_df.mean()*12
volReturn = return_df.std()*(12**0.5)
SR = excessReturn.mean()*12/volReturn
skewness=return_df.skew()
kurtosis=return_df.kurtosis()


benchmark_data = pd.read_excel(r'/Users/fabioribeiro/Documents/Master/QARM2/project/benchmark.xlsx', sheet_name='benchmark',index_col=0)
benchmark_data=benchmark_data.loc[PER_data.iloc[:,0:252].columns]
benchmark_cum = (1+benchmark_data).cumprod()
plt.plot(benchmark_cum)

cum_returns=cum_returns.T
cum_returns1=cum_returns2.T


cum_returns1.set_index(benchmark_cum.index, inplace=True)

cum_returns.set_index(benchmark_cum.index, inplace=True)

plt.plot(benchmark_cum, color='b', label='benchmark')
plt.plot( cum_returns, color='r', label='equally weighted')
plt.plot( cum_returns1, color='g', label='rank weighted')

plt.legend();


#cum returns on the 5 last years
cum_returns_5years = (1+returns_port.iloc[0,192:]).cumprod()
cum_returns_5years1 = (1+returns_port2.iloc[0,192:]).cumprod()
benchmark_cum_5years =  (1+benchmark_data.iloc[192:,0]).cumprod()

cum_returns_5years=cum_returns_5years.T
cum_returns_5years1=cum_returns_5years1.T

cum_returns_5years=pd.DataFrame(cum_returns_5years)
cum_returns_5years1=pd.DataFrame(cum_returns_5years1)

cum_returns_5years1.set_index(benchmark_cum.iloc[192:,0].index, inplace=True)

cum_returns_5years.set_index(benchmark_cum.iloc[192:,0].index, inplace=True)

plt.plot(benchmark_cum_5years, color='b', label='benchmark')
plt.plot( cum_returns_5years, color='r', label='equally weighted')
plt.plot( cum_returns_5years1, color='g', label='rank weighted')
plt.legend();







import pandas_datareader as web
import statsmodels.api as sm

ff_df = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start='2001-2-1', end='2022-1-1')[0]
ff_df = ff_df[['Mkt-RF', 'SMB', 'HML']]
ff_df.index = returns.iloc[:,0:252].columns
ff_df

x = ff_df/100
y = returns_port.T
y.index=returns.iloc[:,0:252].columns

model = sm.OLS(y, x, missing='drop')
results = model.fit()
print(results.summary())







from numpy.linalg import inv
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import statsmodels.api as sm
from scipy.stats.mstats import winsorize
import pandas_datareader as web
import statsmodels.api as sm
import time



##QUALITY
GPM_data = np.where(GPM_data.isnull(), np.nan, winsorize(np.ma.masked_invalid(GPM_data),limits=(0.01,0.01)))
DE_data = np.where(DE_data.isnull(), np.nan, winsorize(np.ma.masked_invalid(DE_data),limits=(0.01,0.01)))
ROA_data = np.where(ROA_data.isnull(), np.nan, winsorize(np.ma.masked_invalid(ROA_data),limits=(0.01,0.01)))
TA_data = np.where(TA_data.isnull(), np.nan, winsorize(np.ma.masked_invalid(TA_data),limits=(0.01,0.01)))


GPM_data = pd.DataFrame(GPM_data)
DE_data = pd.DataFrame(DE_data)
TA_data = pd.DataFrame(TA_data)
ROA_data = pd.DataFrame(ROA_data)

GPM_data.set_index(PER_data1.columns, inplace=True)
GPM_data.columns = PER_data1.index
ROA_data.set_index(PER_data1.columns, inplace=True)
ROA_data.columns = PER_data1.index
TA_data.set_index(PER_data1.columns, inplace=True)
TA_data.columns = PER_data1.index
DE_data.set_index(PER_data1.columns, inplace=True)
DE_data.columns = PER_data1.index



GPM_average_byfirm=np.zeros((1,261))
GPM_average_byfirm = pd.DataFrame(GPM_average_byfirm)
GPM_average_byfirm.columns = GPM_data.T.columns
for i in range(261):
    GPM_average_byfirm.iloc[0,i] = GPM_data.T.iloc[:,i].mean()

GPM_std=np.zeros((1,261))
GPM_std = pd.DataFrame(GPM_std)
GPM_std.columns = GPM_data.T.columns
for i in range(261):
    GPM_std.iloc[0,i] = GPM_data.T.iloc[:,i].std()

GPM_zscore=np.zeros((502,261))
GPM_zscore = pd.DataFrame(GPM_zscore)
GPM_zscore.set_index(GPM_data.T.index, inplace=True)
GPM_zscore.columns = GPM_data.T.columns
for i in range(261):
    for j in range(502):
        GPM_zscore.iloc[j,i] = (GPM_data.T.iloc[j,i]-GPM_average_byfirm.iloc[0,i])/GPM_std.iloc[0,i]

ROA_average=np.zeros((1,261))
ROA_average = pd.DataFrame(ROA_average)
ROA_average.columns = ROA_data.T.columns
for i in range(261):
    ROA_average.iloc[0,i] = ROA_data.T.iloc[:,i].mean()


ROA_std=np.zeros((1,261))
ROA_std = pd.DataFrame(ROA_std)
ROA_std.columns = ROA_data.T.columns
for i in range(261):
    ROA_std.iloc[0,i] = ROA_data.T.iloc[:,i].std()

ROA_zscore=np.zeros((502,261))
ROA_zscore = pd.DataFrame(ROA_zscore)
ROA_zscore.set_index(ROA_data.T.index, inplace=True)
ROA_zscore.columns = ROA_data.T.columns
for i in range(261):
    for j in range(502):
        ROA_zscore.iloc[j,i] = (ROA_data.T.iloc[j,i]-ROA_average.iloc[0,i])/ROA_std.iloc[0,i]

ROE_data = pd.read_excel(r'/Users/fabioribeiro/Documents/Master/QARM2/project/data.xlsx', sheet_name = 'Sheet4', index_col=0, parse_dates=True).iloc[:-1,]
ROE_data = ROE_data.T
ROE_data = ROE_data.reindex(columns = PER_data1.columns)
ROE_data = np.where(ROE_data.isnull(), np.nan, winsorize(np.ma.masked_invalid(ROE_data),limits=(0.01,0.01)))
ROE_data = pd.DataFrame(ROE_data)
ROE_data.set_index(PER_data1.index, inplace=True)
ROE_data.columns = new_header

ROE_average=np.zeros((1,261))
ROE_average = pd.DataFrame(ROE_average)
ROE_average.columns = new_header
for i in range(261):
    ROE_average.iloc[0,i] = ROE_data.iloc[:,i].mean()

ROE_std=np.zeros((1,261))
ROE_std = pd.DataFrame(ROE_std)
ROE_std.columns = new_header
for i in range(261):
    ROE_std.iloc[0,i] = ROE_data.iloc[:,i].std()

ROE_zscore=np.zeros((502,261))
ROE_zscore = pd.DataFrame(ROE_zscore)
ROE_zscore.set_index(ROE_data.index, inplace=True)
ROE_zscore.columns = new_header
for i in range(261):
    for j in range(502):
        ROE_zscore.iloc[j,i] = (ROE_data.iloc[j,i]-ROE_average.iloc[0,i])/ROE_std.iloc[0,i]

DE_data = DE_data.pow(-1)
DE_average=np.zeros((1,261))
DE_average = pd.DataFrame(DE_average)
DE_average.columns = DE_data.T.columns
for i in range(261):
    DE_average.iloc[0,i] = DE_data.T.iloc[:,i].mean()


DE_std=np.zeros((1,261))
DE_std = pd.DataFrame(DE_std)
DE_std.columns = DE_data.T.columns
for i in range(261):
    DE_std.iloc[0,i] = DE_data.T.iloc[:,i].std()

DE_zscore=np.zeros((502,261))
DE_zscore = pd.DataFrame(DE_zscore)
DE_zscore.set_index(DE_data.T.index, inplace=True)
DE_zscore.columns = DE_data.T.columns
for i in range(261):
    for j in range(502):
        DE_zscore.iloc[j,i] = (DE_data.T.iloc[j,i]-DE_average.iloc[0,i])/DE_std.iloc[0,i]


TA_data = TA_data.pow(-1)
TA_average=np.zeros((1,261))
TA_average = pd.DataFrame(TA_average)
TA_average.columns = TA_data.T.columns
for i in range(261):
    TA_average.iloc[0,i] = TA_data.T.iloc[:,i].mean()
    



TA_std=np.zeros((1,261))
TA_std = pd.DataFrame(TA_std)
TA_std.columns = TA_data.T.columns
for i in range(261):
    TA_std.iloc[0,i] = TA_data.T.iloc[:,i].std()

TA_zscore=np.zeros((502,261))
TA_zscore = pd.DataFrame(TA_zscore)
TA_zscore.set_index(TA_data.T.index, inplace=True)
TA_zscore.columns = TA_data.T.columns
for i in range(261):
    for j in range(502):
        TA_zscore.iloc[j,i] = (TA_data.T.iloc[j,i]-TA_average.iloc[0,i])/TA_std.iloc[0,i]

quality_zscore=np.zeros((502,261))
quality_zscore = pd.DataFrame(quality_zscore)
quality_zscore.set_index(PER_data.index, inplace=True)
quality_zscore.columns = new_header
for i in range(261):
    for j in range(502):
        quality_zscore.iloc[j,i] = GPM_zscore.iloc[j,i]+ROA_zscore.iloc[j,i]+TA_zscore.iloc[j,i]
        
quality_rank=np.zeros((502,261))
quality_rank = pd.DataFrame(quality_rank)
quality_rank.set_index(PBR_data.index, inplace=True)
quality_rank.columns = new_header
for i in range(261):
    quality_rank.iloc[:,i] = quality_zscore.iloc[:,i].rank()





quality_Top_Up_25 = []
for i in range(252):
    quality_Top_Up_25.append(quality_rank.iloc[:,i].sort_values().dropna()[-25:])


quality_Top_Down_25 = []
for i in range(252):
    quality_Top_Down_25.append(quality_rank.iloc[:,i].sort_values().dropna()[0:25])



#Equally Weighted
qW_up=np.zeros((25,252))
qW_down = np.zeros((25,252))
qW_up = pd.DataFrame(qW_up)
qW_down  = pd.DataFrame(qW_down)
q_returns_asset=np.zeros((25,252))
q_returns_asset = pd.DataFrame(q_returns_asset)
q_returns_port=np.zeros((1,252))
q_returns_port = pd.DataFrame(q_returns_port)
q_cum_returns=np.zeros((1,252))
q_cum_returns = pd.DataFrame(q_cum_returns)
for i in range (252):
    qTop_Up_25names = quality_Top_Up_25[i].keys()
    qTop_Down_25names = quality_Top_Down_25[i].keys()
    for j in range (25):
        qW_up.iloc[j,i]=1/25
        qW_down.iloc[j,i]=1/25
        
        q_returns_asset.iloc[j,i]= qW_up.iloc[j,i]*returns.loc[qTop_Up_25names].iloc[j,i+1]-qW_down.iloc[j,i]*returns.loc[qTop_Down_25names].iloc[j,i+1]                                                                              
    q_returns_port.iloc[0,i]=q_returns_asset.iloc[:,i].sum()
    if i==0 :
        q_cum_returns.iloc[0,i]=1+q_returns_port.iloc[0,i]
    else :
        q_cum_returns.iloc[0,i]=(1+q_returns_port.iloc[0,i])*q_cum_returns.iloc[0,i-1]
        #q_cum_returns.iloc[0,i] = q_returns_port.iloc[0,i] + q_cum_returns.iloc[0,i-1]      

plt.plot(q_cum_returns.iloc[0,:])


#Z-score weighted
#Z-score weighted
qW_up_exp=np.zeros((25,252))
qW_down_exp = np.zeros((25,261))
qW_up_exp = pd.DataFrame(qW_up_exp)
qW_down_exp  = pd.DataFrame(qW_down_exp)
q_returns_asset1=np.zeros((25,252))
q_returns_asset1 = pd.DataFrame(q_returns_asset1)
q_returns_port1=np.zeros((1,252))
q_returns_port1 = pd.DataFrame(q_returns_port1)
q_cum_returns1=np.zeros((1,252))
q_cum_returns1 = pd.DataFrame(q_cum_returns1)
for i in range (252):
    qTop_Up_25names = quality_Top_Up_25[i].keys()
    qTop_Down_25names = quality_Top_Down_25[i].keys()
    for j in range (25):
        qW_up_exp.iloc[j,i]=np.exp(quality_zscore.loc[qTop_Up_25names].iloc[j,i])/(np.exp(quality_zscore.loc[qTop_Up_25names].iloc[:,i]).sum())
        qW_down_exp.iloc[j,i]=np.exp(quality_zscore.loc[qTop_Down_25names].iloc[j,i]**-1)/(np.exp(quality_zscore.loc[qTop_Down_25names].iloc[:,i]**-1).sum())
        q_returns_asset1.iloc[j,i]= qW_up_exp.iloc[j,i]*returns.loc[qTop_Up_25names].iloc[j,i+1]-qW_down_exp.iloc[j,i]*returns.loc[qTop_Down_25names].iloc[j,i+1]                                                                              
    q_returns_port1.iloc[0,i]=q_returns_asset1.iloc[:,i].sum()
    if i==0 :
        q_cum_returns1.iloc[0,i]=1+q_returns_port1.iloc[0,i]
    else :
        q_cum_returns1.iloc[0,i]=(1+q_returns_port1.iloc[0,i])*q_cum_returns1.iloc[0,i-1]
        #q_cum_returns1.iloc[0,i] = q_returns_port1.iloc[0,i] + q_cum_returns1.iloc[0,i-1]
 
plt.plot(q_cum_returns1.iloc[0,:])

#rank weighted
qW_up_exp=np.zeros((25,252))
qW_down_exp = np.zeros((25,261))
qW_up_exp = pd.DataFrame(qW_up_exp)
qW_down_exp  = pd.DataFrame(qW_down_exp)
q_returns_asset2=np.zeros((25,252))
q_returns_asset2 = pd.DataFrame(q_returns_asset2)
q_returns_port2=np.zeros((1,252))
q_returns_port2 = pd.DataFrame(q_returns_port2)
q_cum_returns2=np.zeros((1,252))
q_cum_returns2 = pd.DataFrame(q_cum_returns2)
test = np.zeros((1,252))
test = pd.DataFrame(test)
for i in range (252):
    qTop_Up_25names = quality_Top_Up_25[i].keys()
    qTop_Down_25names = quality_Top_Down_25[i].keys()
    for j in range (25):
        qW_up_exp.iloc[j,i]=(j+1)/(25*(25+1)/2)
        qW_down_exp.iloc[j,i]=((25-j))/(25*(25+1)/2)
        q_returns_asset2.iloc[j,i]= qW_up_exp.iloc[j,i]*returns.loc[qTop_Up_25names].iloc[j,i+1]-qW_down_exp.iloc[j,i]*returns.loc[qTop_Down_25names].iloc[j,i+1]                                                                              
    q_returns_port2.iloc[0,i]=q_returns_asset2.iloc[:,i].sum()
    if i==0 :
        q_cum_returns2.iloc[0,i]=1+q_returns_port2.iloc[0,i]
    else :
        q_cum_returns2.iloc[0,i]=(1+q_returns_port2.iloc[0,i])*q_cum_returns2.iloc[0,i-1]
        #q_cum_returns2.iloc[0,i] = q_returns_port.iloc[0,i] + q_cum_returns.iloc[0,i-1]

plt.plot(q_cum_returns2.iloc[0,:])




q_cum_returns=q_cum_returns.T
q_cum_returns1=q_cum_returns1.T
q_cum_returns2=q_cum_returns2.T

q_cum_returns2.set_index(benchmark_cum.index, inplace=True)
q_cum_returns1.set_index(benchmark_cum.index, inplace=True)
q_cum_returns.set_index(benchmark_cum.index, inplace=True)

plt.plot(benchmark_cum, color='b', label='benchmark')
plt.plot( q_cum_returns, color='r', label='equally weighted')
plt.plot( q_cum_returns1, color='g', label='zscore weighted')
plt.plot( q_cum_returns2, color='y', label='rank weighted')
plt.legend();

#fama french exposition to the factor
ff_df1 = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2001-2-1', end='2022-1-1')[0]
ff_df1 = ff_df1[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
ff_df1.index = returns.iloc[:,0:252].columns

x = ff_df1/100
y = q_returns_port.T
y.index=returns.iloc[:,0:252].columns

model = sm.OLS(y, x, missing='drop')
results = model.fit()
print(results.summary())


#adding transaction costs
TO_q=np.zeros((1,251))
TO_q = pd.DataFrame(TO_q)
for i in range (251):
    
    
    for j in range (25):
        if quality_Top_Up_25[i+1].keys()[j] in quality_Top_Up_25[i].keys():
            TO_q.iloc[0,i]=TO_q.iloc[0,i]
        else:
            TO_q.iloc[0,i]=TO_q.iloc[0,i]+0.08
            
        if quality_Top_Down_25[i+1].keys()[j] in quality_Top_Down_25[i].keys():
            TO_q.iloc[0,i]=TO_q.iloc[0,i]
        else:
            TO_q.iloc[0,i]=TO_q.iloc[0,i]+0.08
TO_q.mean(axis=1)*12

my_list=quality_Top_Up_25[i+1].keys()    
my_list=my_list.tolist()    
my_list.index(quality_Top_Up_25[i].keys()[1])

my_list.count(quality_Top_Up_25[i+1].keys()[j])

TO1_q=np.zeros((1,251))
TO1_q = pd.DataFrame(TO1)
for i in range (251):
    my_list_up=quality_Top_Up_25[i].keys()    
    my_list_up=my_list_up.tolist()
    my_list_down=quality_Top_Down_25[i].keys()    
    my_list_down=my_list_down.tolist()        
    
    for j in range (25):
        if my_list_up.count(quality_Top_Up_25[i+1].keys()[j])>0 and quality_Top_Up_25[i+1].keys()[j]==quality_Top_Up_25[i].keys()[j]:
            TO1_q.iloc[0,i]=TO1_q.iloc[0,i]
        elif my_list_up.count(quality_Top_Up_25[i+1].keys()[j])>0 and quality_Top_Up_25[i+1].keys()[j]!=quality_Top_Up_25[i].keys()[j]:
            TO1_q.iloc[0,i]=TO1_q.iloc[0,i]+abs((qW_up_exp.iloc[my_list_up.index(quality_Top_Up_25[i+1].keys()[j]),i+1]-qW_up_exp.iloc[j,i])*2)
        else:
            TO1_q.iloc[0,i]=TO1_q.iloc[0,i]+qW_up_exp.iloc[j,i]
            
            
        if my_list_down.count(quality_Top_Down_25[i+1].keys()[j])>0 and quality_Top_Down_25[i+1].keys()[j]==quality_Top_Down_25[i].keys()[j]:
            TO1_q.iloc[0,i]=TO_q.iloc[0,i]
        elif my_list_down.count(quality_Top_Down_25[i+1].keys()[j])>0 and quality_Top_Down_25[i+1].keys()[j]!=quality_Top_Down_25[i].keys()[j]:
            TO1_q.iloc[0,i]=TO1_q.iloc[0,i]+abs((qW_down_exp.iloc[my_list_down.index(quality_Top_Down_25[i+1].keys()[j]),i+1]-qW_down_exp.iloc[j,i])*2)
        else:
            TO1_q.iloc[0,i]=TO_q.iloc[0,i]+qW_down_exp.iloc[j,i]
TO1_q.mean(axis=1)*12

TC1_q=0.002*TO1_q
TC_q=0.002*TO_q

TC_q.insert(0,"",0)
TC1_q.insert(0,"",0)

TC_q.columns=returns_port.columns
TC1_q.columns=returns_port.columns

#Equally Weighted with transaction cost
qW_up=np.zeros((25,252))
qW_down = np.zeros((25,252))
qW_up = pd.DataFrame(qW_up)
qW_down  = pd.DataFrame(qW_down)
q_returns_asset=np.zeros((25,252))
q_returns_asset = pd.DataFrame(q_returns_asset)
q_returns_port=np.zeros((1,252))
q_returns_port = pd.DataFrame(q_returns_port)
q_cum_returns=np.zeros((1,252))
q_cum_returns = pd.DataFrame(q_cum_returns)
for i in range (252):
    qTop_Up_25names = quality_Top_Up_25[i].keys()
    qTop_Down_25names = quality_Top_Down_25[i].keys()
    for j in range (25):
        qW_up.iloc[j,i]=1/25
        qW_down.iloc[j,i]=1/25
        
        q_returns_asset.iloc[j,i]= qW_up.iloc[j,i]*returns.loc[qTop_Up_25names].iloc[j,i+1]-qW_down.iloc[j,i]*returns.loc[qTop_Down_25names].iloc[j,i+1]                                                                              
    q_returns_port.iloc[0,i]=q_returns_asset.iloc[:,i].sum()-TC_q.iloc[0,i]
    if i==0 :
        q_cum_returns.iloc[0,i]=1+q_returns_port.iloc[0,i]
    else :
        q_cum_returns.iloc[0,i]=(1+q_returns_port.iloc[0,i])*q_cum_returns.iloc[0,i-1]
        #q_cum_returns.iloc[0,i] = q_returns_port.iloc[0,i] + q_cum_returns.iloc[0,i-1]      

plt.plot(q_cum_returns.iloc[0,:])


#rank weighted with transaction cost
qW_up_exp=np.zeros((25,252))
qW_down_exp = np.zeros((25,261))
qW_up_exp = pd.DataFrame(qW_up_exp)
qW_down_exp  = pd.DataFrame(qW_down_exp)
q_returns_asset2=np.zeros((25,252))
q_returns_asset2 = pd.DataFrame(q_returns_asset2)
q_returns_port2=np.zeros((1,252))
q_returns_port2 = pd.DataFrame(q_returns_port2)
q_cum_returns2=np.zeros((1,252))
q_cum_returns2 = pd.DataFrame(q_cum_returns2)
test = np.zeros((1,252))
test = pd.DataFrame(test)
for i in range (252):
    qTop_Up_25names = quality_Top_Up_25[i].keys()
    qTop_Down_25names = quality_Top_Down_25[i].keys()
    for j in range (25):
        qW_up_exp.iloc[j,i]=(j+1)/(25*(25+1)/2)
        qW_down_exp.iloc[j,i]=((25-j))/(25*(25+1)/2)
        q_returns_asset2.iloc[j,i]= qW_up_exp.iloc[j,i]*returns.loc[qTop_Up_25names].iloc[j,i+1]-qW_down_exp.iloc[j,i]*returns.loc[qTop_Down_25names].iloc[j,i+1]                                                                              
    q_returns_port2.iloc[0,i]=q_returns_asset2.iloc[:,i].sum()-TC1_q.iloc[0,i]
    if i==0 :
        q_cum_returns2.iloc[0,i]=1+q_returns_port2.iloc[0,i]
    else :
        q_cum_returns2.iloc[0,i]=(1+q_returns_port2.iloc[0,i])*q_cum_returns2.iloc[0,i-1]
        #q_cum_returns2.iloc[0,i] = q_returns_port.iloc[0,i] + q_cum_returns.iloc[0,i-1]

plt.plot(q_cum_returns2.iloc[0,:])

q_cum_returns=q_cum_returns.T

q_cum_returns2=q_cum_returns2.T

q_cum_returns2.set_index(benchmark_cum.index, inplace=True)
q_cum_returns1.set_index(benchmark_cum.index, inplace=True)
q_cum_returns.set_index(benchmark_cum.index, inplace=True)

plt.plot(benchmark_cum, color='b', label='benchmark')
plt.plot( q_cum_returns, color='r', label='equally weighted')

plt.plot( q_cum_returns2, color='y', label='rank weighted')
plt.legend();


#cum returns on the 5 last years for quality we can see it is outperforming the benchmark
q_cum_returns_5years = (1+q_returns_port.iloc[0,192:]).cumprod()
q_cum_returns_5years1 = (1+q_returns_port2.iloc[0,192:]).cumprod()
benchmark_cum_5years =  (1+benchmark_data.iloc[192:,0]).cumprod()

q_cum_returns_5years=q_cum_returns_5years.T
q_cum_returns_5years1=q_cum_returns_5years1.T

q_cum_returns_5years=pd.DataFrame(q_cum_returns_5years)
q_cum_returns_5years1=pd.DataFrame(q_cum_returns_5years1)

q_cum_returns_5years1.set_index(benchmark_cum.iloc[192:,0].index, inplace=True)

q_cum_returns_5years.set_index(benchmark_cum.iloc[192:,0].index, inplace=True)

plt.plot(benchmark_cum_5years, color='b', label='benchmark')
plt.plot( q_cum_returns_5years, color='r', label='equally weighted')
plt.plot( q_cum_returns_5years1, color='g', label='rank weighted')
plt.legend();







ff_df1 = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2001-2-1', end='2022-1-1')[0]
ff_df1 = ff_df1[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
ff_df1.index = returns.iloc[:,0:252].columns

x = ff_df1/100
y = q_returns_port.T
y.index=returns.iloc[:,0:252].columns

model = sm.OLS(y, x, missing='drop')
results = model.fit()
print(results.summary())


#Calculating the correlation between value and quality portfolios by looking at the returns
#Negative correlation is a good sign of why quality is a good hedge to value portfolios
df = pd.concat([returns_port.T, returns_port1.T, returns_port2.T, q_returns_port.T, q_returns_port1.T, q_returns_port1.T], axis = 1)
df.columns = ['Value EQ', 'Value Zscore', 'Value rank','Quality EQ', 'Quality Zscore', 'Quality rank']
corr_df = df.corr(method='pearson')
corr_df
#What is quality paper - 'portfolios based on company profitability have negative correlations with value portfolios'
#What is quality paper - 'portfolios based on company profitability have negative correlations with value portfolios'


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

integ_Top_Up_25 = []
for i in range(252):
    integ_Top_Up_25.append(integ_rank.iloc[:,i].nlargest(25))


integ_Top_Down_25 = []
for i in range(252):
    integ_Top_Down_25.append(integ_rank.iloc[:,i].nsmallest(25))










#Equally Weighted integrated without tc
iW_up=np.zeros((25,252))
iW_down = np.zeros((25,252))
iW_up = pd.DataFrame(iW_up)
iW_down  = pd.DataFrame(iW_down)
i_returns_asset=np.zeros((25,252))
i_returns_asset = pd.DataFrame(i_returns_asset)
i_returns_port=np.zeros((1,252))
i_returns_port = pd.DataFrame(i_returns_port)
i_cum_returns=np.zeros((1,252))
i_cum_returns = pd.DataFrame(i_cum_returns)

for i in range (252):
    iTop_Up_25names = integ_Top_Up_25[i].keys()
    iTop_Down_25names = integ_Top_Down_25[i].keys()
    for j in range (25):
        iW_up.iloc[j,i]=1/25
        iW_down.iloc[j,i]=1/25
        
        i_returns_asset.iloc[j,i]= iW_up.iloc[j,i]*returns.loc[iTop_Up_25names].iloc[j,i+1]-(iW_down.iloc[j,i]*returns.loc[iTop_Down_25names].iloc[j,i+1])
                                                                                
    i_returns_port.iloc[0,i]=i_returns_asset.iloc[:,i].sum()
    
    
    if i==0 :
        i_cum_returns.iloc[0,i]=1+i_returns_port.iloc[0,i]
        
    else :
        i_cum_returns.iloc[0,i]=(1+i_returns_port.iloc[0,i])*i_cum_returns.iloc[0,i-1]
        
            
plt.plot(i_cum_returns.iloc[0,:])

#rank weighted integrated portfolio without tc
iW_up=np.zeros((25,252))
iW_down = np.zeros((25,252))
iW_up = pd.DataFrame(iW_up)
iW_down  = pd.DataFrame(iW_down)
i_returns_asset=np.zeros((25,252))
i_returns_asset = pd.DataFrame(i_returns_asset)
i_returns_port=np.zeros((1,252))
i_returns_port = pd.DataFrame(i_returns_port)
i_cum_returns=np.zeros((1,252))
i_cum_returns = pd.DataFrame(i_cum_returns)

for i in range (252):
    iTop_Up_25names = integ_Top_Up_25[i].keys()
    iTop_Down_25names = integ_Top_Down_25[i].keys()
    for j in range (25):
        iW_up.iloc[j,i]=(j+1)/(25*(25+1)/2)
        iW_down.iloc[j,i]=((25-j))/(25*(25+1)/2)
        
        i_returns_asset.iloc[j,i]= iW_up.iloc[j,i]*returns.loc[iTop_Up_25names].iloc[j,i+1]-(iW_down.iloc[j,i]*returns.loc[iTop_Down_25names].iloc[j,i+1])
                                                                                
    i_returns_port.iloc[0,i]=i_returns_asset.iloc[:,i].sum()
    
    
    if i==0 :
        i_cum_returns.iloc[0,i]=1+i_returns_port.iloc[0,i]
        
    else :
        i_cum_returns.iloc[0,i]=(1+i_returns_port.iloc[0,i])*i_cum_returns.iloc[0,i-1]
        
            
plt.plot(i_cum_returns.iloc[0,:])





i_cum_returns = i_cum_returns.T
i_cum_returns.set_index(benchmark_cum.index, inplace=True)
plt.plot(benchmark_cum, color='b', label='benchmark')
plt.plot(i_cum_returns, color='r', label='Integrated')
plt.plot(cum_returns, color='g', label='Value')
plt.plot(q_cum_returns, color='y', label='Quality')
plt.legend()

x = ff_df1/100
y = i_returns_port.T
y.index=returns.iloc[:,0:252].columns

model = sm.OLS(y, x, missing='drop')
results = model.fit()
print(results.summary())

#adding transaction costs
TO_i=np.zeros((1,251))
TO_i = pd.DataFrame(TO_i)
for i in range (251):
    
    
    for j in range (25):
        if integ_Top_Up_25[i+1].keys()[j] in integ_Top_Up_25[i].keys():
            TO_i.iloc[0,i]=TO_i.iloc[0,i]
        else:
            TO_i.iloc[0,i]=TO_i.iloc[0,i]+0.08
            
        if integ_Top_Down_25[i+1].keys()[j] in integ_Top_Down_25[i].keys():
            TO_i.iloc[0,i]=TO_i.iloc[0,i]
        else:
            TO_i.iloc[0,i]=TO_i.iloc[0,i]+0.08
TO_i.mean(axis=1)*12


TC_i=0.002*TO_i

TC_i.insert(0,"",0)


TC_i.columns=returns_port.columns

TO1_i=np.zeros((1,251))
TO1_i = pd.DataFrame(TO1)
for i in range (251):
    my_list_up=integ_Top_Up_25[i].keys()    
    my_list_up=my_list_up.tolist()
    my_list_down=integ_Top_Down_25[i].keys()    
    my_list_down=my_list_down.tolist()        
    
    for j in range (25):
        if my_list_up.count(integ_Top_Up_25[i+1].keys()[j])>0 and integ_Top_Up_25[i+1].keys()[j]==integ_Top_Up_25[i].keys()[j]:
            TO1_i.iloc[0,i]=TO1_i.iloc[0,i]
        elif my_list_up.count(integ_Top_Up_25[i+1].keys()[j])>0 and integ_Top_Up_25[i+1].keys()[j]!=integ_Top_Up_25[i].keys()[j]:
            TO1_i.iloc[0,i]=TO1_i.iloc[0,i]+(iW_up.iloc[my_list_up.index(integ_Top_Up_25[i+1].keys()[j]),i+1]-iW_up.iloc[j,i])*2
        else:
            TO1_i.iloc[0,i]=TO1_i.iloc[0,i]+iW_up.iloc[j,i]
            
            
        if my_list_down.count(integ_Top_Down_25[i+1].keys()[j])>0 and integ_Top_Down_25[i+1].keys()[j]==integ_Top_Down_25[i].keys()[j]:
            TO1_i.iloc[0,i]=TO_i.iloc[0,i]
        elif my_list_down.count(integ_Top_Down_25[i+1].keys()[j])>0 and integ_Top_Down_25[i+1].keys()[j]!=integ_Top_Down_25[i].keys()[j]:
            TO1_i.iloc[0,i]=TO1_i.iloc[0,i]+(iW_down.iloc[my_list_down.index(integ_Top_Down_25[i+1].keys()[j]),i+1]-iW_down.iloc[j,i])*2
        else:
            TO1_i.iloc[0,i]=TO_i.iloc[0,i]+iW_down.iloc[j,i]

TC1_i=0.002*TO1_i
TC1_i.insert(0,"",0)
TC1_i.columns=returns_port.columns

TO1_i.mean(axis=1)*12

#Equally Weighted integrated with transaction costs
iW_up=np.zeros((25,252))
iW_down = np.zeros((25,252))
iW_up = pd.DataFrame(iW_up)
iW_down  = pd.DataFrame(iW_down)
i_returns_asset=np.zeros((25,252))
i_returns_asset = pd.DataFrame(i_returns_asset)
i_returns_port=np.zeros((1,252))
i_returns_port = pd.DataFrame(i_returns_port)
i_cum_returns=np.zeros((1,252))
i_cum_returns = pd.DataFrame(i_cum_returns)
returns_long_ai =np.zeros((25,252))
returns_long_ai = pd.DataFrame(returns_long_ai)
returns_longi =np.zeros((1,252))
returns_longi = pd.DataFrame(returns_longi)
returns_short_ai =np.zeros((25,252))
returns_short_ai = pd.DataFrame(returns_short_ai)
returns_shorti =np.zeros((1,252))
returns_shorti = pd.DataFrame(returns_shorti)
cum_returns_longi=np.zeros((1,252))
cum_returns_longi = pd.DataFrame(cum_returns_longi)
cum_returns_shorti=np.zeros((1,252))
cum_returns_shorti = pd.DataFrame(cum_returns_shorti)
for i in range (252):
    iTop_Up_25names = integ_Top_Up_25[i].keys()
    iTop_Down_25names = integ_Top_Down_25[i].keys()
    for j in range (25):
        iW_up.iloc[j,i]=1/25
        iW_down.iloc[j,i]=1/25
        
        i_returns_asset.iloc[j,i]= iW_up.iloc[j,i]*returns.loc[iTop_Up_25names].iloc[j,i+1]-iW_down.iloc[j,i]*returns.loc[iTop_Down_25names].iloc[j,i+1]
        returns_long_ai.iloc[j,i]= iW_up.iloc[j,i]*returns.loc[iTop_Up_25names].iloc[j,i+1]
        returns_short_ai.iloc[j,i]= -1*iW_up.iloc[j,i]*returns.loc[iTop_Down_25names].iloc[j,i+1]                                                                             
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
risk_free = risk_free.resample('MS').first()/1200
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

#long short decomposition

cum_returns_longi.columns = benchmark_data.T.columns
plt.plot(cum_returns_longi.iloc[0,:])

cum_returns_shorti.columns = benchmark_data.T.columns
plt.plot(cum_returns_shorti.iloc[0,:])

plt.plot(returns_longi.iloc[0,:])
plt.plot(returns_shorti.iloc[0,:])
average = returns_longi.mean(axis=1)
volat = returns_longi.std(axis=1)

#rank Weighted integrated with transaction costs
iW_up=np.zeros((25,252))
iW_down = np.zeros((25,252))
iW_up = pd.DataFrame(iW_up)
iW_down  = pd.DataFrame(iW_down)
i_returns_asset1=np.zeros((25,252))
i_returns_asset1 = pd.DataFrame(i_returns_asset1)
i_returns_port1=np.zeros((1,252))
i_returns_port1 = pd.DataFrame(i_returns_port1)
i_cum_returns1=np.zeros((1,252))
i_cum_returns1 = pd.DataFrame(i_cum_returns1)
returns_long_ai1 =np.zeros((25,252))
returns_long_ai1 = pd.DataFrame(returns_long_ai1)
returns_longi1 =np.zeros((1,252))
returns_longi1 = pd.DataFrame(returns_longi1)
returns_short_ai1 =np.zeros((25,252))
returns_short_ai1 = pd.DataFrame(returns_short_ai1)
returns_shorti1 =np.zeros((1,252))
returns_shorti1 = pd.DataFrame(returns_shorti1)
cum_returns_longi1=np.zeros((1,252))
cum_returns_longi1 = pd.DataFrame(cum_returns_longi1)
cum_returns_shorti1=np.zeros((1,252))
cum_returns_shorti1 = pd.DataFrame(cum_returns_shorti1)
for i in range (252):
    iTop_Up_25names = integ_Top_Up_25[i].keys()
    iTop_Down_25names = integ_Top_Down_25[i].keys()
    for j in range (25):
        iW_up.iloc[j,i]=(j+1)/(25*(25+1)/2)
        iW_down.iloc[j,i]=((25-j))/(25*(25+1)/2)
        
        
        i_returns_asset.iloc[j,i]= iW_up.iloc[j,i]*returns.loc[iTop_Up_25names].iloc[j,i+1]-iW_down.iloc[j,i]*returns.loc[iTop_Down_25names].iloc[j,i+1]
        returns_long_ai1.iloc[j,i]= iW_up.iloc[j,i]*returns.loc[iTop_Up_25names].iloc[j,i+1]
        returns_short_ai1.iloc[j,i]= -1*iW_up.iloc[j,i]*returns.loc[iTop_Down_25names].iloc[j,i+1]                                                                            
    i_returns_port.iloc[0,i]=i_returns_asset.iloc[:,i].sum()-TC1_i.iloc[0,i]
    returns_longi1.iloc[0,i]=returns_long_ai1.iloc[:,i].sum()-TC1_i.iloc[0,i]/2
    returns_shorti1.iloc[0,i]=returns_short_ai1.iloc[:,i].sum()-TC1_i.iloc[0,i]/2
    if i==0 :
        i_cum_returns.iloc[0,i]=1+i_returns_port.iloc[0,i]
        cum_returns_longi1.iloc[0,i]=1+returns_longi1.iloc[0,i]
        cum_returns_shorti1.iloc[0,i]=1+returns_shorti1.iloc[0,i]
    else :
        i_cum_returns.iloc[0,i]=(1+i_returns_port.iloc[0,i])*i_cum_returns.iloc[0,i-1]
        cum_returns_longi1.iloc[0,i]=(1+returns_longi1.iloc[0,i])*cum_returns_longi1.iloc[0,i-1]
        cum_returns_shorti1.iloc[0,i]=(1+returns_shorti1.iloc[0,i])*cum_returns_shorti1.iloc[0,i-1]        
plt.plot(i_cum_returns.iloc[0,:])

cum_returns_longi1.columns = benchmark_data.T.columns
plt.plot(cum_returns_longi1.iloc[0,:])
cum_returns_shorti1.columns = benchmark_data.T.columns
plt.plot(cum_returns_shorti1.iloc[0,:])

plt.plot(returns_longi1.iloc[0,:])
plt.plot(returns_shorti1.iloc[0,:])


risk_free.columns = ['RF']
return_df= returns_longi1.T
return_df.index=risk_free.index
excessReturn = return_df.sub(risk_free['RF'], axis=0)
geometric_mean = ((1+return_df).cumprod().iloc[-1]**(1/len(return_df))-1)
meanReturn = return_df.mean()*12
volReturn = return_df.std()*(12**0.5)
SR = excessReturn.mean()*12/volReturn
skewness=return_df.skew()
kurtosis=return_df.kurtosis()



x = ff_df1/100
y = i_returns_port.T
y.index=returns.iloc[:,0:252].columns

model = sm.OLS(y, x, missing='drop')
results = model.fit()
print(results.summary())


## MIX PORTFOLIO

w = 0.8 #percentage of value stocks in the portfolio 
mix_Top_Up_25 = []
for i in range(252):
    mix_Top_Up_25.append(value_rank.iloc[:,i].nlargest(int(w*25))+quality_rank.iloc[:,i].nlargest(30-int(w*25))) 
#Quand je mets 25 ça me donne pas toujours un len(mix_Top_up_25) = 25   

mix_Top_Down_25 = []
for i in range(252):
    mix_Top_Down_25.append(value_rank.iloc[:,i].nsmallest(int(w*25))+quality_rank.iloc[:,i].nsmallest(30-int(w*25)))
#Quand je mets 25 ça me donne pas toujours un len(mix_Down_up_25) = 25    


#Equally Weighted
mW_up=np.zeros((25,252))
mW_down = np.zeros((25,252))
mW_up = pd.DataFrame(mW_up)
mW_down  = pd.DataFrame(mW_down)
m_returns_asset=np.zeros((25,252))
m_returns_asset = pd.DataFrame(m_returns_asset)
m_returns_port=np.zeros((1,252))
m_returns_port = pd.DataFrame(m_returns_port)
m_cum_returns=np.zeros((1,252))
m_cum_returns = pd.DataFrame(m_cum_returns)
for i in range (252):
    mTop_Up_25names = mix_Top_Up_25[i].keys()
    mTop_Down_25names = mix_Top_Down_25[i].keys()
    for j in range (25):
        mW_up.iloc[j,i]=1/25
        mW_down.iloc[j,i]=1/25
        
        m_returns_asset.iloc[j,i]= mW_up.iloc[j,i]*returns.loc[mTop_Up_25names].iloc[j,i+1]-mW_down.iloc[j,i]*returns.loc[mTop_Down_25names].iloc[j,i+1]                                                                              
    m_returns_port.iloc[0,i]=m_returns_asset.iloc[:,i].sum()
    if i==0 :
        m_cum_returns.iloc[0,i]=1+m_returns_port.iloc[0,i]
    else :
        #m_cum_returns.iloc[0,i]=(1+m_returns_port.iloc[0,i])*m_cum_returns.iloc[0,i-1]
        m_cum_returns.iloc[0,0] = 1
        m_cum_returns.iloc[0,i] = m_returns_port.iloc[0,i] + m_cum_returns.iloc[0,i-1]        
plt.plot(m_cum_returns.iloc[0,:])

m_cum_returns = m_cum_returns.T
m_cum_returns.set_index(benchmark_cum.index, inplace=True)
plt.figure(figsize=(10,6), dpi=80)
plt.plot(benchmark_cum, color='b', label='benchmark')
plt.plot(i_cum_returns, color='r', label='Integrated')
plt.plot(cum_returns, color='g', label='Value')
plt.plot(q_cum_returns, color='y', label='Quality')
plt.plot(m_cum_returns, color='c', label='Mixed')
plt.legend()

x = ff_df1/100
y = m_returns_port.T
y.index=returns.iloc[:,0:252].columns

model = sm.OLS(y, x, missing='drop')
results = model.fit()
print(results.summary())


#Equally Weighted
mW_up=np.zeros((25,252))
mW_down = np.zeros((25,252))
mW_up = pd.DataFrame(mW_up)
mW_down  = pd.DataFrame(mW_down)
m_returns_asset=np.zeros((25,252))
m_returns_asset = pd.DataFrame(m_returns_asset)
m_returns_port=np.zeros((1,252))
m_returns_port = pd.DataFrame(m_returns_port)
m_cum_returns=np.zeros((1,252))
m_cum_returns = pd.DataFrame(m_cum_returns)
for i in range (252):
    mTop_Up_25names = mix_Top_Up_25[i].keys()
    mTop_Down_25names = mix_Top_Down_25[i].keys()
    for j in range (25):
        mW_up.iloc[j,i]=1/25
        mW_down.iloc[j,i]=1/25
        
        m_returns_asset.iloc[j,i]= mW_up.iloc[j,i]*returns.loc[mTop_Up_25names].iloc[j,i+1]-mW_down.iloc[j,i]*returns.loc[mTop_Down_25names].iloc[j,i+1]                                                                              
    m_returns_port.iloc[0,i]=m_returns_asset.iloc[:,i].sum()
    if i==0 :
        m_cum_returns.iloc[0,i]=1+m_returns_port.iloc[0,i]
    else :
        #m_cum_returns.iloc[0,i]=(1+m_returns_port.iloc[0,i])*m_cum_returns.iloc[0,i-1]
        m_cum_returns.iloc[0,0] = 1
        m_cum_returns.iloc[0,i] = m_returns_port.iloc[0,i] + m_cum_returns.iloc[0,i-1]        
plt.plot(m_cum_returns.iloc[0,:])

mix_zscore=np.zeros((502,261))
mix_zscore = pd.DataFrame(mix_zscore)
mix_zscore.set_index(PER_data.index, inplace=True)
mix_zscore.columns = new_header
for i in range(261):
    for j in range(502):
        mix_zscore.iloc[j,i] = 0.5*quality_zscore.iloc[j,i]+0.5*value_zscore.iloc[j,i]


mix_rank=np.zeros((502,261))
mix_rank = pd.DataFrame(mix_rank)
mix_rank.set_index(PBR_data.index, inplace=True)
mix_rank.columns = new_header
for i in range(261):
    mix_rank.iloc[:,i] = mix_zscore.iloc[:,i].rank()
    
mix_Top_Up_25 = []
for i in range(252):
    mix_Top_Up_25.append(mix_rank.iloc[:,i].nlargest(25))


mix_Top_Down_25 = []
for i in range(252):
    mix_Top_Down_25.append(mix_rank.iloc[:,i].nsmallest(25))

#Equally Weighted
mW_up=np.zeros((25,252))
mW_down = np.zeros((25,252))
mW_up = pd.DataFrame(W_up)
mW_down  = pd.DataFrame(W_down)
m_returns_asset=np.zeros((25,252))
m_returns_asset = pd.DataFrame(m_returns_asset)
m_returns_port=np.zeros((1,252))
m_returns_port = pd.DataFrame(m_returns_port)
m_cum_returns=np.zeros((1,252))
m_cum_returns = pd.DataFrame(m_cum_returns)
for i in range (252):
    mTop_Up_25names = mix_Top_Up_25[i].keys()
    mTop_Down_25names = mix_Top_Down_25[i].keys()
    for j in range (25):
        mW_up.iloc[j,i]=1/25
        mW_down.iloc[j,i]=1/25
        
        m_returns_asset.iloc[j,i]= mW_up.iloc[j,i]*returns.loc[mTop_Up_25names].iloc[j,i+1]-mW_down.iloc[j,i]*returns.loc[mTop_Down_25names].iloc[j,i+1]                                                                              
    m_returns_port.iloc[0,i]=m_returns_asset.iloc[:,i].sum()
    
    if i==0 :
        m_cum_returns.iloc[0,i]=1+m_returns_port.iloc[0,i]
    else :
        m_cum_returns.iloc[0,i]=(1+m_returns_port.iloc[0,i])*m_cum_returns.iloc[0,i-1]        

plt.plot(m_cum_returns.iloc[0,:])    

test=np.zeros((1,252))
test = pd.DataFrame(test)
for i in range (252):
    if i==0 :
        test.iloc[0,i]=1+m_returns_port.iloc[0,i]
    else :
        test.iloc[0,i]=(1+m_returns_port.iloc[0,i])*test.iloc[0,i-1]  


m_cum_returns2 = m_cum_returns.T
m_cum_returns2.set_index(benchmark_cum.index, inplace=True)
plt.plot(benchmark_cum, color='b', label='benchmark')
plt.plot( m_cum_returns2, color='r', label='Mix')
plt.plot( cum_returns, color='g', label='Value')
plt.plot( q_cum_returns, color='y', label='Quality')
plt.legend()




ff_df1 = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2001-2-1', end='2022-1-1')[0]
ff_df1 = ff_df1[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
ff_df1.index = returns.iloc[:,0:252].columns

x = ff_df1/100
y = m_returns_port.T
y.index=returns.iloc[:,0:252].columns

model = sm.OLS(y, x, missing='drop')
results = model.fit()
print(results.summary())



Top_Up_25_quarterly = []
for i in range(0,252,3):
    Top_Up_25_quarterly.append(value_rank.iloc[:,i].sort_values().dropna()[-25:])

Top_Down_25_quarterly = []
for i in range(0,252,3):
    Top_Down_25_quarterly.append(value_rank.iloc[:,i].sort_values().dropna()[0:25])


#quarterly
#Equally Weighted
quarterW_up=np.zeros((25,84))
quarterW_down = np.zeros((25,84))
quarterW_up = pd.DataFrame(quarterW_up)
quarterW_down  = pd.DataFrame(quarterW_down)
quarter_returns_asset=np.zeros((25,84))
quarter_returns_asset = pd.DataFrame(quarter_returns_asset)
quarter_returns_port=np.zeros((1,84))
quarter_returns_port = pd.DataFrame(quarter_returns_port)
quarter_cum_returns=np.zeros((1,84))
quarter_cum_returns = pd.DataFrame(quarter_cum_returns)
for i in range (84):
    quarter_Top_Up_25names = Top_Up_25_quarterly[i].keys()
    quarter_Top_Down_25names = Top_Down_25_quarterly[i].keys()
    for j in range (25):
        quarterW_up.iloc[j,i]=1/25
        quarterW_down.iloc[j,i]=1/25
        
        quarter_returns_asset.iloc[j,i]= quarterW_up.iloc[j,i]*(returns.loc[quarter_Top_Up_25names].iloc[j,3*i+1]+returns.loc[quarter_Top_Up_25names].iloc[j,3*i+2]+returns.loc[quarter_Top_Up_25names].iloc[j,3*i+3])-quarterW_down.iloc[j,i]*(returns.loc[quarter_Top_Down_25names].iloc[j,3*i+1]+returns.loc[quarter_Top_Down_25names].iloc[j,3*i+2]+returns.loc[quarter_Top_Down_25names].iloc[j,3*i+3])                                                                              
    quarter_returns_port.iloc[0,i]=quarter_returns_asset.iloc[:,i].sum()
    if i==0 :
        quarter_cum_returns.iloc[0,i]=1+quarter_returns_port.iloc[0,i]
    else :
        quarter_cum_returns.iloc[0,i]=(1+quarter_returns_port.iloc[0,i])*quarter_cum_returns.iloc[0,i-1]       

plt.plot(quarter_cum_returns.iloc[0,:])

#Z-score weighted
quarterW_up_exp=np.zeros((25,84))
quarterW_down_exp = np.zeros((25,84))
quarterW_up_exp = pd.DataFrame(quarterW_up_exp)
quarterW_down_exp  = pd.DataFrame(quarterW_down_exp)
quarter_returns_asset1=np.zeros((25,84))
quarter_returns_asset1 = pd.DataFrame(quarter_returns_asset1)
quarter_returns_port1=np.zeros((1,84))
quarter_returns_port1 = pd.DataFrame(quarter_returns_port1)
quarter_cum_returns1=np.zeros((1,84))
quarter_cum_returns1 = pd.DataFrame(quarter_cum_returns1)
for i in range (84):
    quarter_Top_Up_25names = Top_Up_25_quarterly[i].keys()
    quarter_Top_Down_25names = Top_Down_25_quarterly[i].keys()
    for j in range (25):
        quarterW_up_exp.iloc[j,i]=np.exp(value_zscore.loc[quarter_Top_Up_25names].iloc[j,3*i])/(np.exp(value_zscore.loc[quarter_Top_Up_25names].iloc[:,3*i]).sum())
        quarterW_down_exp.iloc[j,i]=np.exp(value_zscore.loc[quarter_Top_Down_25names].iloc[j,3*i]**-1)/(np.exp(value_zscore.loc[quarter_Top_Down_25names].iloc[:,3*i]**-1).sum())
        quarter_returns_asset1.iloc[j,i]= quarterW_up_exp.iloc[j,i]*(returns.loc[quarter_Top_Up_25names].iloc[j,3*i+1]+returns.loc[quarter_Top_Up_25names].iloc[j,3*i+2]+returns.loc[quarter_Top_Up_25names].iloc[j,3*i+3])- quarterW_down_exp.iloc[j,i]*(returns.loc[quarter_Top_Down_25names].iloc[j,3*i+1]+returns.loc[quarter_Top_Down_25names].iloc[j,3*i+2]+returns.loc[quarter_Top_Down_25names].iloc[j,3*i+3])                                                                              
    quarter_returns_port1.iloc[0,i]=quarter_returns_asset1.iloc[:,i].sum()
    if i==0 :
        quarter_cum_returns1.iloc[0,i]=1+quarter_returns_port1.iloc[0,i]
    else :
        quarter_cum_returns1.iloc[0,i]=(1+quarter_returns_port1.iloc[0,i])*quarter_cum_returns1.iloc[0,i-1]
 
plt.plot(quarter_cum_returns1.iloc[0,:])

#rank weighted
quarterW_up_exp=np.zeros((25,84))
quarterW_down_exp = np.zeros((25,84))
quarterW_up_exp = pd.DataFrame(quarterW_up_exp)
quarterW_down_exp  = pd.DataFrame(quarterW_down_exp)
quarter_returns_asset2=np.zeros((25,84))
quarter_returns_asset2 = pd.DataFrame(quarter_returns_asset2)
quarter_returns_port2=np.zeros((1,84))
quarter_returns_port2 = pd.DataFrame(quarter_returns_port2)
quarter_cum_returns2=np.zeros((1,84))
quarter_cum_returns2 = pd.DataFrame(quarter_cum_returns2)
for i in range (84):
    quarter_Top_Up_25names = Top_Up_25_quarterly[i].keys()
    quarter_Top_Down_25names = Top_Down_25_quarterly[i].keys()
    for j in range (25):
        quarterW_up_exp.iloc[j,i]=(j+1)/(25*(25+1)/2)
        quarterW_down_exp.iloc[j,i]=((25-j))/(25*(25+1)/2)
        quarter_returns_asset2.iloc[j,i]= quarterW_up_exp.iloc[j,i]*(returns.loc[quarter_Top_Up_25names].iloc[j,3*i+1]+returns.loc[quarter_Top_Up_25names].iloc[j,3*i+2]+returns.loc[quarter_Top_Up_25names].iloc[j,3*i+3])-quarterW_down_exp.iloc[j,i]*(returns.loc[quarter_Top_Down_25names].iloc[j,3*i+1]+returns.loc[quarter_Top_Down_25names].iloc[j,3*i+2]+returns.loc[quarter_Top_Down_25names].iloc[j,3*i+3])                                                                              
    quarter_returns_port2.iloc[0,i]=quarter_returns_asset2.iloc[:,i].sum()
    if i==0 :
        quarter_cum_returns2.iloc[0,i]=1+quarter_returns_port2.iloc[0,i]
    else :
        quarter_cum_returns2.iloc[0,i]=(1+quarter_returns_port2.iloc[0,i])*quarter_cum_returns2.iloc[0,i-1]
 
plt.plot(quarter_cum_returns2.iloc[0,:])

quarter = pd.date_range(start='2001-1-1', end='2022-1-1', freq = 'QS', closed = 'right')

quarter_cum_returns=quarter_cum_returns.T
quarter_cum_returns1=quarter_cum_returns1.T
quarter_cum_returns2=quarter_cum_returns2.T

quarter_cum_returns2.set_index(quarter, inplace=True)
quarter_cum_returns1.set_index(quarter, inplace=True)
quarter_cum_returns.set_index(quarter, inplace=True)

plt.plot(benchmark_cum, color='b', label='benchmark')
plt.plot(quarter_cum_returns, color='r', label='equally weighted')
#plt.plot(quarter_cum_returns1, color='g', label='zscore weighted')
plt.plot(quarter_cum_returns2, color='y', label='rank weighted')
plt.legend()

##Quality
qTop_Up_25_quarterly = []
for i in range(0,252,3):
    qTop_Up_25_quarterly.append(quality_rank.iloc[:,i].sort_values().dropna()[-25:])

qTop_Down_25_quarterly = []
for i in range(0,252,3):
    qTop_Down_25_quarterly.append(quality_rank.iloc[:,i].sort_values().dropna()[0:25])

#Equally Weighted
qqW_up=np.zeros((25,84))
qqW_down = np.zeros((25,84))
qqW_up = pd.DataFrame(qqW_up)
qqW_down  = pd.DataFrame(qqW_down)
qq_returns_asset=np.zeros((25,84))
qq_returns_asset = pd.DataFrame(qq_returns_asset)
qq_returns_port=np.zeros((1,84))
qq_returns_port = pd.DataFrame(qq_returns_port)
qq_cum_returns=np.zeros((1,84))
qq_cum_returns = pd.DataFrame(qq_cum_returns)
for i in range (84):
    qqTop_Up_25names = qTop_Up_25_quarterly[i].keys()
    qqTop_Down_25names = qTop_Down_25_quarterly[i].keys()
    for j in range (25):
        qqW_up.iloc[j,i]=1/25
        qW_down.iloc[j,i]=1/25
        
        qq_returns_asset.iloc[j,i]= qqW_up.iloc[j,i]*(returns.loc[qTop_Up_25names].iloc[j,3*i+1]+returns.loc[qTop_Up_25names].iloc[j,3*i+2]+returns.loc[qTop_Up_25names].iloc[j,3*i+3])-qW_down.iloc[j,i]*(returns.loc[qTop_Up_25names].iloc[j,3*i+1]+returns.loc[qTop_Up_25names].iloc[j,3*i+2]+returns.loc[qTop_Up_25names].iloc[j,3*i+3])                                                                             
    qq_returns_port.iloc[0,i]=q_returns_asset.iloc[:,i].sum()
    if i==0 :
        qq_cum_returns.iloc[0,i]=1+qq_returns_port.iloc[0,i]
    else :
        qq_cum_returns.iloc[0,i]=(1+q_returns_port.iloc[0,i])*q_cum_returns.iloc[0,i-1]    

plt.plot(qq_cum_returns.iloc[0,:])

#Z-score weighted
qqW_up_exp=np.zeros((25,84))
qqW_down_exp = np.zeros((25,84))
qqW_up_exp = pd.DataFrame(qqW_up_exp)
qqW_down_exp  = pd.DataFrame(qqW_down_exp)
qq_returns_asset1=np.zeros((25,84))
qq_returns_asset1 = pd.DataFrame(qq_returns_asset1)
qq_returns_port1=np.zeros((1,84))
qq_returns_port1 = pd.DataFrame(qq_returns_port1)
qq_cum_returns1=np.zeros((1,84))
qq_cum_returns1 = pd.DataFrame(qq_cum_returns1)
for i in range (84):
    qTop_Up_25names = quality_Top_Up_25[i].keys()
    qTop_Down_25names = quality_Top_Down_25[i].keys()
    for j in range (25):
        qqW_up_exp.iloc[j,i]=np.exp(quality_zscore.loc[qTop_Up_25names].iloc[j,3*i])/(np.exp(quality_zscore.loc[qTop_Up_25names].iloc[:,3*i]).sum())
        qW_down_exp.iloc[j,i]=np.exp(quality_zscore.loc[qTop_Down_25names].iloc[j,3*i]**-1)/(np.exp(quality_zscore.loc[qTop_Down_25names].iloc[:,3*i]**-1).sum())
        qq_returns_asset1.iloc[j,i]= qqW_up_exp.iloc[j,i]*(returns.loc[qTop_Up_25names].iloc[j,3*i+1]+returns.loc[qTop_Up_25names].iloc[j,3*i+2]+returns.loc[qTop_Up_25names].iloc[j,3*i+3])-qqW_down_exp.iloc[j,i]*(returns.loc[qTop_Down_25names].iloc[j,3*i+1]+returns.loc[qTop_Down_25names].iloc[j,3*i+2]+returns.loc[qTop_Down_25names].iloc[j,3*i+3])                                                                              
    qq_returns_port1.iloc[0,i]=qq_returns_asset1.iloc[:,i].sum()
    if i==0 :
        qq_cum_returns1.iloc[0,i]=1+qq_returns_port1.iloc[0,i]
    else :
        qq_cum_returns1.iloc[0,i]=(1+qq_returns_port1.iloc[0,i])*qq_cum_returns1.iloc[0,i-1]
 
plt.plot(qq_cum_returns1.iloc[0,:])

#rank weighted
qqW_up_exp=np.zeros((25,84))
qqW_down_exp = np.zeros((25,84))
qqW_up_exp = pd.DataFrame(qqW_up_exp)
qqW_down_exp  = pd.DataFrame(qqW_down_exp)
qq_returns_asset2=np.zeros((25,84))
qq_returns_asset2 = pd.DataFrame(qq_returns_asset2)
qq_returns_port2=np.zeros((1,84))
qq_returns_port2 = pd.DataFrame(qq_returns_port2)
qq_cum_returns2=np.zeros((1,84))
qq_cum_returns2 = pd.DataFrame(qq_cum_returns2)
for i in range (84):
    qTop_Up_25names = quality_Top_Up_25[i].keys()
    qTop_Down_25names = quality_Top_Down_25[i].keys()
    for j in range (25):
        qqW_up_exp.iloc[j,i]=(j+1)/(25*(25+1)/2)
        qqW_down_exp.iloc[j,i]=((25-j))/(25*(25+1)/2)
        qq_returns_asset2.iloc[j,i]= qqW_up_exp.iloc[j,i]*(returns.loc[qTop_Up_25names].iloc[j,3*i+1]+returns.loc[qTop_Up_25names].iloc[j,3*i+2]+returns.loc[qTop_Up_25names].iloc[j,3*i+3])-qqW_down_exp.iloc[j,i]*(returns.loc[qTop_Down_25names].iloc[j,3*i+1]+returns.loc[qTop_Down_25names].iloc[j,3*i+2]+returns.loc[qTop_Down_25names].iloc[j,3*i+3])                                                                              
    qq_returns_port2.iloc[0,i]=qq_returns_asset2.iloc[:,i].sum()
    if i==0 :
        qq_cum_returns2.iloc[0,i]=1+qq_returns_port2.iloc[0,i]
    else :
        qq_cum_returns2.iloc[0,i]=(1+qq_returns_port2.iloc[0,i])*qq_cum_returns2.iloc[0,i-1]

plt.plot(qq_cum_returns2.iloc[0,:])

qq_cum_returns=qq_cum_returns.T
qq_cum_returns1=qq_cum_returns1.T
qq_cum_returns2=qq_cum_returns2.T

qq_cum_returns2.set_index(quarter, inplace=True)
qq_cum_returns1.set_index(quarter, inplace=True)
qq_cum_returns.set_index(quarter, inplace=True)

plt.plot(benchmark_cum, color='b', label='benchmark')
plt.plot(qq_cum_returns, color='r', label='equally weighted')
plt.plot(qq_cum_returns1, color='g', label='zscore weighted')
plt.plot(qq_cum_returns2, color='y', label='rank weighted')
plt.legend()

#Integrated

quarter_integ_Top_Up_25 = []
for i in range(0,252,3):
    quarter_integ_Top_Up_25.append(integ_rank.iloc[:,i].nlargest(25))


quarter_integ_Top_Down_25 = []
for i in range(0,252,3):
    quarter_integ_Top_Down_25.append(integ_rank.iloc[:,i].nsmallest(25))

#Equally Weighted
iqW_up=np.zeros((25,84))
iqW_down = np.zeros((25,84))
iqW_up = pd.DataFrame(iqW_up)
iqW_down  = pd.DataFrame(iqW_down)
iq_returns_asset=np.zeros((25,84))
iq_returns_asset = pd.DataFrame(iq_returns_asset)
iq_returns_port=np.zeros((1,84))
iq_returns_port = pd.DataFrame(iq_returns_port)
iq_cum_returns=np.zeros((1,84))
iq_cum_returns = pd.DataFrame(iq_cum_returns)
for i in range (84):
    iqTop_Up_25names = quarter_integ_Top_Up_25[i].keys()
    iqTop_Down_25names = quarter_integ_Top_Down_25[i].keys()
    for j in range (25):
        iqW_up.iloc[j,i]=1/25
        iqW_down.iloc[j,i]=1/25
        
        iq_returns_asset.iloc[j,i]= iqW_up.iloc[j,i]*(returns.loc[iqTop_Up_25names].iloc[j,3*i+1]+returns.loc[iqTop_Up_25names].iloc[j,3*i+2]+returns.loc[iqTop_Up_25names].iloc[j,3*i+3]).iloc[j,i+1]-iW_down.iloc[j,i]*(returns.loc[iqTop_Down_25names].iloc[j,3*i+1]+returns.loc[iqTop_Down_25names].iloc[j,3*i+2]+returns.loc[iqTop_Down_25names].iloc[j,3*i+3])                                                                              
    iq_returns_port.iloc[0,i]=iq_returns_asset.iloc[:,i].sum()
    if i==0 :
        iq_cum_returns.iloc[0,i]=1+iq_returns_port.iloc[0,i]
    else :
        iq_cum_returns.iloc[0,i]=(1+iq_returns_port.iloc[0,i])*iq_cum_returns.iloc[0,i-1]        
plt.plot(iq_cum_returns.iloc[0,:])



#sensitivity analysis all of the test will be on the final integrated portfolio
#first we try by increasing transaction fees



TO_iplus=np.zeros((1,251))
TO_iplus = pd.DataFrame(TO_iplus)
for i in range (251):
    
    
    for j in range (25):
        if integ_Top_Up_25[i+1].keys()[j] in integ_Top_Up_25[i].keys():
            TO_iplus.iloc[0,i]=TO_iplus.iloc[0,i]
        else:
            TO_iplus.iloc[0,i]=TO_iplus.iloc[0,i]+0.08
            
        if integ_Top_Down_25[i+1].keys()[j] in integ_Top_Down_25[i].keys():
            TO_iplus.iloc[0,i]=TO_iplus.iloc[0,i]
        else:
            TO_iplus.iloc[0,i]=TO_iplus.iloc[0,i]+0.08



TC_iplus=0.008*TO_iplus

TC_iplus.insert(0,"",0)


TC_iplus.columns=returns_port.columns



#equally wweighted integrated port with increase tc
iW_up=np.zeros((25,252))
iW_down = np.zeros((25,252))
iW_up = pd.DataFrame(iW_up)
iW_down  = pd.DataFrame(iW_down)
i_returns_asset=np.zeros((25,252))
i_returns_asset = pd.DataFrame(i_returns_asset)
i_returns_port=np.zeros((1,252))
i_returns_port = pd.DataFrame(i_returns_port)
i_cum_returns=np.zeros((1,252))
i_cum_returns = pd.DataFrame(i_cum_returns)

for i in range (252):
    iTop_Up_25names = integ_Top_Up_25[i].keys()
    iTop_Down_25names = integ_Top_Down_25[i].keys()
    for j in range (25):
        iW_up.iloc[j,i]=1/25
        iW_down.iloc[j,i]=1/25
        
        i_returns_asset.iloc[j,i]= iW_up.iloc[j,i]*returns.loc[iTop_Up_25names].iloc[j,i+1]-(iW_down.iloc[j,i]*returns.loc[iTop_Down_25names].iloc[j,i+1])
                                                                                
    i_returns_port.iloc[0,i]=i_returns_asset.iloc[:,i].sum()-TC_iplus.iloc[0,i]
    
    
    if i==0 :
        i_cum_returns.iloc[0,i]=1+i_returns_port.iloc[0,i]
        
    else :
        i_cum_returns.iloc[0,i]=(1+i_returns_port.iloc[0,i])*i_cum_returns.iloc[0,i-1]
        
            
plt.plot(i_cum_returns.iloc[0,:])

from scipy.stats import skew, kurtosis
import pandas_datareader as web
risk_free = web.DataReader('DTB3', 'fred', start='2001-1-1', end='2021-12-01')
risk_free = risk_free.resample('MS').first()/1200
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


#quarterly part
quarter_integ_Top_Up_25 = []
for i in range(0,252,3):
    quarter_integ_Top_Up_25.append(integ_rank.iloc[:,i].nlargest(25))


quarter_integ_Top_Down_25 = []
for i in range(0,252,3):
    quarter_integ_Top_Down_25.append(integ_rank.iloc[:,i].nsmallest(25))

#Equally Weighted integrated quarterly without tc
iqW_up=np.zeros((25,84))
iqW_down = np.zeros((25,84))
iqW_up = pd.DataFrame(iqW_up)
iqW_down  = pd.DataFrame(iqW_down)
iq_returns_asset=np.zeros((25,84))
iq_returns_asset = pd.DataFrame(iq_returns_asset)
iq_returns_port=np.zeros((1,84))
iq_returns_port = pd.DataFrame(iq_returns_port)
iq_cum_returns1=np.zeros((1,84))
iq_cum_returns1 = pd.DataFrame(iq_cum_returns1)
for i in range (84):
    iqTop_Up_25names = quarter_integ_Top_Up_25[i].keys()
    iqTop_Down_25names = quarter_integ_Top_Down_25[i].keys()
    for j in range (25):
        iqW_up.iloc[j,i]=1/25
        iqW_down.iloc[j,i]=1/25
        
        iq_returns_asset.iloc[j,i]= iqW_up.iloc[j,i]*(returns.loc[iqTop_Up_25names].iloc[j,3*i+1]+returns.loc[iqTop_Up_25names].iloc[j,3*i+2]+returns.loc[iqTop_Up_25names].iloc[j,3*i+3])-iqW_down.iloc[j,i]*(returns.loc[iqTop_Down_25names].iloc[j,3*i+1]+returns.loc[iqTop_Down_25names].iloc[j,3*i+2]+returns.loc[iqTop_Down_25names].iloc[j,3*i+3])                                                                              
    iq_returns_port.iloc[0,i]=iq_returns_asset.iloc[:,i].sum()
    if i==0 :
        iq_cum_returns1.iloc[0,i]=1+iq_returns_port.iloc[0,i]
    else :
        iq_cum_returns1.iloc[0,i]=(1+iq_returns_port.iloc[0,i])*iq_cum_returns1.iloc[0,i-1]        
plt.plot(iq_cum_returns1.iloc[0,:])


TO_i=np.zeros((1,83))
TO_i = pd.DataFrame(TO_i)
for i in range (83):
    
    
    for j in range (25):
        if quarter_integ_Top_Up_25[i+1].keys()[j] in quarter_integ_Top_Up_25[i].keys():
            TO_i.iloc[0,i]=TO_i.iloc[0,i]
        else:
            TO_i.iloc[0,i]=TO_i.iloc[0,i]+0.08
            
        if quarter_integ_Top_Down_25[i+1].keys()[j] in quarter_integ_Top_Down_25[i].keys():
            TO_i.iloc[0,i]=TO_i.iloc[0,i]
        else:
            TO_i.iloc[0,i]=TO_i.iloc[0,i]+0.08



TC_i=0.008*TO_i

TC_i.insert(0,"",0)

#EW quarterly integrated with increased tc
iqW_up=np.zeros((25,84))
iqW_down = np.zeros((25,84))
iqW_up = pd.DataFrame(iqW_up)
iqW_down  = pd.DataFrame(iqW_down)
iq_returns_asset=np.zeros((25,84))
iq_returns_asset = pd.DataFrame(iq_returns_asset)
iq_returns_port=np.zeros((1,84))
iq_returns_port = pd.DataFrame(iq_returns_port)
iq_cum_returns1=np.zeros((1,84))
iq_cum_returns1 = pd.DataFrame(iq_cum_returns1)
for i in range (84):
    iqTop_Up_25names = quarter_integ_Top_Up_25[i].keys()
    iqTop_Down_25names = quarter_integ_Top_Down_25[i].keys()
    for j in range (25):
        iqW_up.iloc[j,i]=1/25
        iqW_down.iloc[j,i]=1/25
        
        iq_returns_asset.iloc[j,i]= iqW_up.iloc[j,i]*(returns.loc[iqTop_Up_25names].iloc[j,3*i+1]+returns.loc[iqTop_Up_25names].iloc[j,3*i+2]+returns.loc[iqTop_Up_25names].iloc[j,3*i+3])-iqW_down.iloc[j,i]*(returns.loc[iqTop_Down_25names].iloc[j,3*i+1]+returns.loc[iqTop_Down_25names].iloc[j,3*i+2]+returns.loc[iqTop_Down_25names].iloc[j,3*i+3])                                                                              
    iq_returns_port.iloc[0,i]=iq_returns_asset.iloc[:,i].sum()-TC_i.iloc[0,i]
    if i==0 :
        iq_cum_returns1.iloc[0,i]=1+iq_returns_port.iloc[0,i]
    else :
        iq_cum_returns1.iloc[0,i]=(1+iq_returns_port.iloc[0,i])*iq_cum_returns1.iloc[0,i-1]        
plt.plot(iq_cum_returns1.iloc[0,:])

from scipy.stats import skew, kurtosis
import pandas_datareader as web
risk_free = web.DataReader('DTB3', 'fred', start='2001-1-1', end='2021-12-01')
risk_free = risk_free.resample('QS').first()/1200
print(risk_free)

risk_free.columns = ['RF']
return_df= iq_returns_port.T
return_df.index=risk_free.index
excessReturn = return_df.sub(risk_free['RF'], axis=0)
geometric_mean = ((1+return_df).cumprod().iloc[-1]**(1/len(return_df))-1)
meanReturn = return_df.mean()*4
volReturn = return_df.std()*(4**0.5)
SR = excessReturn.mean()*4/volReturn
skewness=return_df.skew()
kurtosis=return_df.kurtosis()








#sensitivity part 2 increased amount of stocks in portfolio

integ_Top_Up_50 = []
for i in range(0,252):
    integ_Top_Up_50.append(integ_rank.iloc[:,i].nlargest(50))


integ_Top_Down_50 = []
for i in range(0,252):
    integ_Top_Down_50.append(integ_rank.iloc[:,i].nsmallest(50))


iW_up=np.zeros((50,252))
iW_down = np.zeros((50,252))
iW_up = pd.DataFrame(iW_up)
iW_down  = pd.DataFrame(iW_down)
i_returns_asset=np.zeros((50,252))
i_returns_asset = pd.DataFrame(i_returns_asset)
i_returns_port=np.zeros((1,252))
i_returns_port = pd.DataFrame(i_returns_port)
i_cum_returns=np.zeros((1,252))
i_cum_returns = pd.DataFrame(i_cum_returns)

for i in range (252):
    iTop_Up_50names = integ_Top_Up_50[i].keys()
    iTop_Down_50names = integ_Top_Down_50[i].keys()
    for j in range (25):
        iW_up.iloc[j,i]=1/50
        iW_down.iloc[j,i]=1/50
        
        i_returns_asset.iloc[j,i]= iW_up.iloc[j,i]*returns.loc[iTop_Up_50names].iloc[j,i+1]-(iW_down.iloc[j,i]*returns.loc[iTop_Down_50names].iloc[j,i+1])
                                                                                
    i_returns_port.iloc[0,i]=i_returns_asset.iloc[:,i].sum()
    
    
    if i==0 :
        i_cum_returns.iloc[0,i]=1+i_returns_port.iloc[0,i]
        
    else :
        i_cum_returns.iloc[0,i]=(1+i_returns_port.iloc[0,i])*i_cum_returns.iloc[0,i-1]
        
            
plt.plot(i_cum_returns.iloc[0,:])



TO_i50=np.zeros((1,251))
TO_i50 = pd.DataFrame(TO_i50)
for i in range (251):
    
    
    for j in range (50):
        if integ_Top_Up_50[i+1].keys()[j] in integ_Top_Up_50[i].keys():
            TO_i50.iloc[0,i]=TO_i50.iloc[0,i]
        else:
            TO_i50.iloc[0,i]=TO_i50.iloc[0,i]+0.04
            
        if integ_Top_Down_50[i+1].keys()[j] in integ_Top_Down_50[i].keys():
            TO_i50.iloc[0,i]=TO_i50.iloc[0,i]
        else:
            TO_i50.iloc[0,i]=TO_i50.iloc[0,i]+0.04

TO_i50.mean(axis=1)*12

TC_i50=0.002*TO_i50

TC_i50.insert(0,"",0)


TC_i50.columns=returns_port.columns


iW_up=np.zeros((50,252))
iW_down = np.zeros((50,252))
iW_up = pd.DataFrame(iW_up)
iW_down  = pd.DataFrame(iW_down)
i_returns_asset=np.zeros((50,252))
i_returns_asset = pd.DataFrame(i_returns_asset)
i_returns_port=np.zeros((1,252))
i_returns_port = pd.DataFrame(i_returns_port)
i_cum_returns=np.zeros((1,252))
i_cum_returns = pd.DataFrame(i_cum_returns)

for i in range (252):
    iTop_Up_50names = integ_Top_Up_50[i].keys()
    iTop_Down_50names = integ_Top_Down_50[i].keys()
    for j in range (25):
        iW_up.iloc[j,i]=1/50
        iW_down.iloc[j,i]=1/50
        
        i_returns_asset.iloc[j,i]= iW_up.iloc[j,i]*returns.loc[iTop_Up_50names].iloc[j,i+1]-(iW_down.iloc[j,i]*returns.loc[iTop_Down_50names].iloc[j,i+1])
                                                                                
    i_returns_port.iloc[0,i]=i_returns_asset.iloc[:,i].sum()-TC_i50.iloc[0,i]
    
    
    if i==0 :
        i_cum_returns.iloc[0,i]=1+i_returns_port.iloc[0,i]
        
    else :
        i_cum_returns.iloc[0,i]=(1+i_returns_port.iloc[0,i])*i_cum_returns.iloc[0,i-1]
        
            
plt.plot(i_cum_returns.iloc[0,:])

from scipy.stats import skew, kurtosis
import pandas_datareader as web
risk_free = web.DataReader('DTB3', 'fred', start='2001-1-1', end='2021-12-01')
risk_free = risk_free.resample('MS').first()/1200
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


