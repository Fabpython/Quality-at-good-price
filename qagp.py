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
i_returns_port1=np.zeros((1,252))
i_returns_port1 = pd.DataFrame(i_returns_port1)
i_cum_returns1=np.zeros((1,252))
i_cum_returns1 = pd.DataFrame(i_cum_returns1)

for i in range (252):
    iTop_Up_30names = integ_Top_Up_30[i].keys()
    iTop_Down_30names = integ_Top_Down_30[i].keys()
    for j in range (30):
        iW_up.iloc[j,i]=(j+1)/(30*(30+1)/2)
        iW_down.iloc[j,i]=((30-j))/(30*(30+1)/2)
        
        i_returns_asset.iloc[j,i]= iW_up.iloc[j,i]*returns.loc[iTop_Up_30names].iloc[j,i+1]-(iW_down.iloc[j,i]*returns.loc[iTop_Down_30names].iloc[j,i+1])
                                                                                
    i_returns_port1.iloc[0,i]=i_returns_asset.iloc[:,i].sum()
    
    
    if i==0 :
        i_cum_returns1.iloc[0,i]=1+i_returns_port1.iloc[0,i]
        
    else :
        i_cum_returns1.iloc[0,i]=(1+i_returns_port1.iloc[0,i])*i_cum_returns1.iloc[0,i-1]
        
            
plt.plot(i_cum_returns1.iloc[0,:])


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


TC_i=0.004*TO_i

TC_i.insert(0,"",0)


TC_i.columns=FCF_data.iloc[:,0:252].columns

TO1_i=np.zeros((1,251))
TO1_i = pd.DataFrame(TO1_i)
for i in range (251):
    my_list_up=integ_Top_Up_30[i].keys()    
    my_list_up=my_list_up.tolist()
    my_list_down=integ_Top_Down_30[i].keys()    
    my_list_down=my_list_down.tolist()        
    
    for j in range (30):
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

TC1_i=0.004*TO1_i
TC1_i.insert(0,"",0)
TC1_i.columns=FCF_data.iloc[:,0:252].columns



#Equally Weighted integrated with transaction costs

i_returns_port.columns = TC_i.columns
i_returns_port = i_returns_port-TC_i
for i in range (252):    
    if i==0 :
        i_cum_returns.iloc[0,i]=1+i_returns_port.iloc[0,i]
        
    else :
        i_cum_returns.iloc[0,i]=(1+i_returns_port.iloc[0,i])*i_cum_returns.iloc[0,i-1]
plt.plot(i_cum_returns.iloc[0,:])

#rank weighted with transaction costs


i_returns_port1.columns = TC_i.columns
i_returns_port1 = i_returns_port-TC1_i
for i in range (252):    
    if i==0 :
        i_cum_returns1.iloc[0,i]=1+i_returns_port1.iloc[0,i]
        
    else :
        i_cum_returns1.iloc[0,i]=(1+i_returns_port1.iloc[0,i])*i_cum_returns1.iloc[0,i-1]
plt.plot(i_cum_returns1.iloc[0,:])


                      
                      

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
turnover = TO_i.mean(axis=1)*12

risk_free.columns = ['RF']
return_df1= i_returns_port1.T
return_df1.index=risk_free.index
excessReturn1 = return_df1.sub(risk_free['RF'], axis=0)
geometric_mean1 = ((1+return_df1).cumprod().iloc[-1]**(1/len(return_df1))-1)
meanReturn1 = return_df1.mean()*12
volReturn1 = return_df1.std()*(12**0.5)
SR1 = excessReturn1.mean()*12/volReturn1
skewness1=return_df1.skew()
kurtosis1=return_df1.kurtosis()
turnover1 = TO1_i.mean(axis=1)*12


import scipy
from scipy.optimize import minimize

# Optimization parameters
numberOfAssets=30
e = np.ones((numberOfAssets, 1))

# set the constraints
Aeq = e.T
beq = 1
eq_constraint = lambda x: np.dot(Aeq, x) - beq
cons = ({'type': 'eq', 'fun': eq_constraint})
lub = (0, 1)
bnds = ((lub, ) * numberOfAssets)

    
x0 = np.ones((numberOfAssets, 1)) / numberOfAssets

Sigma = np.cov(returns.iloc[:,i:60+1], bias=True)

def ERCfunc(w, Sigma):
    x = 0
    R = np.dot(Sigma, w)
    for i in range(len(w)):
        for j in range(len(w)):
            x = x + (w[i]*R[i] - w[j]*R[j])**2
    return x


#divRatioERC = (wERC.T @ sigmaVec) / (wERC.T @ Sigma @ wERC) ** 0.5
#print(f"\nDiversification ratio = {divRatioERC[0,0]:.2f}")

#ERC integrated portfolio without tc
wERCup=np.zeros((30,192))
wERCdown = np.zeros((30,192))
wERCup = pd.DataFrame(wERCup)
wERCdown  = pd.DataFrame(wERCdown)
i_returns_asset=np.zeros((30,192))
i_returns_asset = pd.DataFrame(i_returns_asset)
i_returns_port2=np.zeros((1,192))
i_returns_port2 = pd.DataFrame(i_returns_port2)
i_cum_returns2=np.zeros((1,192))
i_cum_returns2 = pd.DataFrame(i_cum_returns2)
numberOfAssets=30
for i in range (192):
    iTop_Up_30names = integ_Top_Up_30[i+60].keys()
    iTop_Down_30names = integ_Top_Down_30[i+60].keys()
    SigmaUp = np.cov(returns.loc[iTop_Up_30names].iloc[:,i:60+i], bias=True)
    SigmaDown = np.cov(returns.loc[iTop_Down_30names].iloc[:,i:60+i], bias=True)
    funERC = lambda x: ERCfunc(x, SigmaUp)
    # run the optimization
    resERC = minimize(funERC, x0, method='SLSQP', tol=1e-8, bounds=bnds, constraints=cons)
    wERCup.iloc[:,i] = resERC.x.reshape((numberOfAssets, 1))
    funERC = lambda x: ERCfunc(x, SigmaDown)
    # run the optimization
    resERC = minimize(funERC, x0, method='SLSQP', tol=1e-8, bounds=bnds, constraints=cons)
    wERCdown.iloc[:,i] = resERC.x.reshape((numberOfAssets, 1))
    
    for j in range (30):
        
        
        i_returns_asset.iloc[j,i]= wERCup.iloc[j,i]*returns.loc[iTop_Up_30names].iloc[j,i+61]-(wERCdown.iloc[j,i]*returns.loc[iTop_Down_30names].iloc[j,i+61])
                                                                                
    i_returns_port2.iloc[0,i]=i_returns_asset.iloc[:,i].sum()
    if i==0 :
        i_cum_returns2.iloc[0,i]=1+i_returns_port2.iloc[0,i]
        
    else :
        i_cum_returns2.iloc[0,i]=(1+i_returns_port2.iloc[0,i])*i_cum_returns2.iloc[0,i-1]
        
plt.plot(i_cum_returns2.iloc[0,:])
    
TO2_i=np.zeros((1,191))
TO2_i = pd.DataFrame(TO2_i)
for i in range (191):
    my_list_up=integ_Top_Up_30[i+60].keys()    
    my_list_up=my_list_up.tolist()
    my_list_down=integ_Top_Down_30[i+60].keys()    
    my_list_down=my_list_down.tolist()        
    
    for j in range (30):
        if my_list_up.count(integ_Top_Up_30[i+61].keys()[j])>0 and integ_Top_Up_30[i+61].keys()[j]==integ_Top_Up_30[i+60].keys()[j]:
            TO2_i.iloc[0,i]=TO2_i.iloc[0,i]
        elif my_list_up.count(integ_Top_Up_30[i+61].keys()[j])>0 and integ_Top_Up_30[i+61].keys()[j]!=integ_Top_Up_30[i+60].keys()[j]:
            TO2_i.iloc[0,i]=TO2_i.iloc[0,i]+(wERCup.iloc[my_list_up.index(integ_Top_Up_30[i+61].keys()[j]),i]-wERCup.iloc[j,i])*2
        else:
            TO2_i.iloc[0,i]=TO2_i.iloc[0,i]+wERCup.iloc[j,i]
            
            
        if my_list_down.count(integ_Top_Down_30[i+61].keys()[j])>0 and integ_Top_Down_30[i+61].keys()[j]==integ_Top_Down_30[i+60].keys()[j]:
            TO2_i.iloc[0,i]=TO2_i.iloc[0,i]
        elif my_list_down.count(integ_Top_Down_30[i+61].keys()[j])>0 and integ_Top_Down_30[i+61].keys()[j]!=integ_Top_Down_30[i+60].keys()[j]:
            TO2_i.iloc[0,i]=TO2_i.iloc[0,i]+(wERCdown.iloc[my_list_down.index(integ_Top_Down_30[i+61].keys()[j]),i]-wERCdown.iloc[j,i])*2
        else:
            TO2_i.iloc[0,i]=TO2_i.iloc[0,i]+wERCdown.iloc[j,i]

TC2_i=0.004*TO2_i
TC2_i.insert(0,"",0)
TC2_i.columns=FCF_data.iloc[:,60:252].columns  


i_returns_port2.columns = TC2_i.columns
i_returns_port2 = i_returns_port2-TC2_i
for i in range (192):    
    if i==0 :
        i_cum_returns2.iloc[0,i]=1+i_returns_port2.iloc[0,i]
        
    else :
        i_cum_returns2.iloc[0,i]=(1+i_returns_port2.iloc[0,i])*i_cum_returns2.iloc[0,i-1]
plt.plot(i_cum_returns2.iloc[0,:])         

risk_free = web.DataReader('DTB3', 'fred', start='2006-1-1', end='2021-12-01')
risk_free = risk_free.resample('MS').first()/1200
risk_free.columns = ['RF']
return_df2= i_returns_port2.T
return_df2.index=risk_free.index
excessReturn2 = return_df2.sub(risk_free['RF'], axis=0)
geometric_mean2 = ((1+return_df2).cumprod().iloc[-1]**(1/len(return_df1))-1)
meanReturn2 = return_df2.mean()*12
volReturn2 = return_df2.std()*(12**0.5)
SR2 = excessReturn2.mean()*12/volReturn2
skewness2=return_df2.skew()
kurtosis2=return_df2.kurtosis()
turnover2 = TO2_i.mean(axis=1)*12


# Optimization parameters
numberOfAssets=30
e = np.ones((numberOfAssets, 1))

# set the constraints
Aeq = e.T
beq = 1
eq_constraint = lambda x: np.dot(Aeq, x) - beq
cons = ({'type': 'eq', 'fun': eq_constraint})
lub = (0, 1)
bnds = ((lub, ) * numberOfAssets)

    



wMDPup=np.zeros((30,192))
wMDPdown = np.zeros((30,192))
wMDPup = pd.DataFrame(wMDPup)
wMDPdown  = pd.DataFrame(wMDPdown)
i_returns_asset=np.zeros((30,192))
i_returns_asset = pd.DataFrame(i_returns_asset)
i_returns_port3=np.zeros((1,192))
i_returns_port3 = pd.DataFrame(i_returns_port3)
i_cum_returns3=np.zeros((1,192))
i_cum_returns3 = pd.DataFrame(i_cum_returns3)
numberOfAssets=30
for i in range (192):
    iTop_Up_30names = integ_Top_Up_30[i+60].keys()
    iTop_Down_30names = integ_Top_Down_30[i+60].keys()
    SigmaUp = np.cov(returns.loc[iTop_Up_30names].iloc[:,i:60+i], bias=True)
    SigmaDown = np.cov(returns.loc[iTop_Down_30names].iloc[:,i:60+i], bias=True)
    sigmaVecup =np.diagonal(SigmaUp)
    sigmaVecdown =np.diagonal(SigmaDown)
    funMDP = lambda x: (-1*x.T @ sigmaVecup) / ((x.T @ SigmaUp @ x) ** 0.5)
    # run the optimization
    resMDP = minimize(funMDP, x0, method='SLSQP', tol=1e-8, bounds=bnds, constraints=cons)
    wMDPup.iloc[:,i] = resMDP.x.reshape((numberOfAssets, 1))
    funMDP = lambda x: (-1*x.T @ sigmaVecdown) / ((x.T @ SigmaDown @ x) ** 0.5)
    # run the optimization
    resMDP = minimize(funMDP, x0, method='SLSQP', tol=1e-8, bounds=bnds, constraints=cons)
    wMDPdown.iloc[:,i] = resMDP.x.reshape((numberOfAssets, 1)) 
    for j in range (30):
        
        
        i_returns_asset.iloc[j,i]= wMDPup.iloc[j,i]*returns.loc[iTop_Up_30names].iloc[j,i+61]-(wMDPdown.iloc[j,i]*returns.loc[iTop_Down_30names].iloc[j,i+61])
                                                                                
    i_returns_port3.iloc[0,i]=i_returns_asset.iloc[:,i].sum()
    if i==0 :
        i_cum_returns3.iloc[0,i]=1+i_returns_port3.iloc[0,i]
        
    else :
        i_cum_returns3.iloc[0,i]=(1+i_returns_port3.iloc[0,i])*i_cum_returns3.iloc[0,i-1]
        
plt.plot(i_cum_returns3.iloc[0,:])

#adding TC
TO3_i=np.zeros((1,191))
TO3_i = pd.DataFrame(TO3_i)
for i in range (191):
    my_list_up=integ_Top_Up_30[i+60].keys()    
    my_list_up=my_list_up.tolist()
    my_list_down=integ_Top_Down_30[i+60].keys()    
    my_list_down=my_list_down.tolist()        
    
    for j in range (30):
        if my_list_up.count(integ_Top_Up_30[i+61].keys()[j])>0 and integ_Top_Up_30[i+61].keys()[j]==integ_Top_Up_30[i+60].keys()[j]:
            TO3_i.iloc[0,i]=TO3_i.iloc[0,i]
        elif my_list_up.count(integ_Top_Up_30[i+61].keys()[j])>0 and integ_Top_Up_30[i+61].keys()[j]!=integ_Top_Up_30[i+60].keys()[j]:
            TO3_i.iloc[0,i]=TO3_i.iloc[0,i]+(wMDPup.iloc[my_list_up.index(integ_Top_Up_30[i+61].keys()[j]),i]-wMDPup.iloc[j,i])*2
        else:
            TO3_i.iloc[0,i]=TO3_i.iloc[0,i]+wMDPup.iloc[j,i]
            
            
        if my_list_down.count(integ_Top_Down_30[i+61].keys()[j])>0 and integ_Top_Down_30[i+61].keys()[j]==integ_Top_Down_30[i+60].keys()[j]:
            TO3_i.iloc[0,i]=TO3_i.iloc[0,i]
        elif my_list_down.count(integ_Top_Down_30[i+61].keys()[j])>0 and integ_Top_Down_30[i+61].keys()[j]!=integ_Top_Down_30[i+60].keys()[j]:
            TO3_i.iloc[0,i]=TO3_i.iloc[0,i]+(wMDPdown.iloc[my_list_down.index(integ_Top_Down_30[i+61].keys()[j]),i]-wMDPdown.iloc[j,i])*2
        else:
            TO3_i.iloc[0,i]=TO3_i.iloc[0,i]+wMDPdown.iloc[j,i]

TC3_i=0.004*TO3_i
TC3_i.insert(0,"",0)
TC3_i.columns=FCF_data.iloc[:,60:252].columns

i_returns_port3.columns = TC3_i.columns
i_returns_port3 = i_returns_port3-TC3_i
for i in range (192):    
    if i==0 :
        i_cum_returns3.iloc[0,i]=1+i_returns_port3.iloc[0,i]
        
    else :
        i_cum_returns3.iloc[0,i]=(1+i_returns_port3.iloc[0,i])*i_cum_returns3.iloc[0,i-1]
plt.plot(i_cum_returns3.iloc[0,:])


risk_free.columns = ['RF']
return_df3= i_returns_port3.T
return_df3.index=risk_free.index
excessReturn3 = return_df2.sub(risk_free['RF'], axis=0)
geometric_mean3 = ((1+return_df2).cumprod().iloc[-1]**(1/len(return_df1))-1)
meanReturn3 = return_df3.mean()*12
volReturn3 = return_df3.std()*(12**0.5)
SR3 = excessReturn3.mean()*12/volReturn3
skewness3=return_df3.skew()
kurtosis3=return_df3.kurtosis()



#Minimum Variance
wMVup=np.zeros((30,192))
wMVdown = np.zeros((30,192))
wMVup= pd.DataFrame(wMVup)
wMVdown  = pd.DataFrame(wMVdown)
i_returns_asset=np.zeros((30,192))
i_returns_asset = pd.DataFrame(i_returns_asset)
i_returns_port4=np.zeros((1,192))
i_returns_port4 = pd.DataFrame(i_returns_port4)
i_cum_returns4=np.zeros((1,192))
i_cum_returns4 = pd.DataFrame(i_cum_returns4)
numberOfAssets=30
for i in range (192):
    iTop_Up_30names = integ_Top_Up_30[i+60].keys()
    iTop_Down_30names = integ_Top_Down_30[i+60].keys()
    SigmaUp = np.cov(returns.loc[iTop_Up_30names].iloc[:,i:60+i], bias=True)
    SigmaDown = np.cov(returns.loc[iTop_Down_30names].iloc[:,i:60+i], bias=True)
    sigmaVecup =np.diagonal(SigmaUp)
    sigmaVecdown =np.diagonal(SigmaDown)
    funMinVar = lambda x: x.T @ SigmaUp @ x
    # run the optimization
    resMinVar = minimize(funMinVar, x0, method='SLSQP', tol=1e-8, bounds=bnds, constraints=cons)
    wMVup.iloc[:,i] = resMinVar.x.reshape((numberOfAssets, 1))
    funMinVar = lambda x: x.T @ SigmaDown @ x
    # run the optimization
    resMinVar = minimize(funMinVar, x0, method='SLSQP', tol=1e-8, bounds=bnds, constraints=cons)
    wMVdown.iloc[:,i] = resMinVar.x.reshape((numberOfAssets, 1))
    for j in range (30):
        
        
        i_returns_asset.iloc[j,i]= wMVup.iloc[j,i]*returns.loc[iTop_Up_30names].iloc[j,i+61]-(wMVdown.iloc[j,i]*returns.loc[iTop_Down_30names].iloc[j,i+61])
                                                                                
    i_returns_port4.iloc[0,i]=i_returns_asset.iloc[:,i].sum()
    if i==0 :
        i_cum_returns4.iloc[0,i]=1+i_returns_port4.iloc[0,i]
        
    else :
        i_cum_returns4.iloc[0,i]=(1+i_returns_port4.iloc[0,i])*i_cum_returns4.iloc[0,i-1]
        
plt.plot(i_cum_returns4.iloc[0,:])

#adding TC
TO4_i=np.zeros((1,191))
TO4_i = pd.DataFrame(TO4_i)
for i in range (191):
    my_list_up=integ_Top_Up_30[i+60].keys()    
    my_list_up=my_list_up.tolist()
    my_list_down=integ_Top_Down_30[i+60].keys()    
    my_list_down=my_list_down.tolist()        
    
    for j in range (30):
        if my_list_up.count(integ_Top_Up_30[i+61].keys()[j])>0 and integ_Top_Up_30[i+61].keys()[j]==integ_Top_Up_30[i+60].keys()[j]:
            TO4_i.iloc[0,i]=TO4_i.iloc[0,i]
        elif my_list_up.count(integ_Top_Up_30[i+61].keys()[j])>0 and integ_Top_Up_30[i+61].keys()[j]!=integ_Top_Up_30[i+60].keys()[j]:
            TO4_i.iloc[0,i]=TO4_i.iloc[0,i]+(wMVup.iloc[my_list_up.index(integ_Top_Up_30[i+61].keys()[j]),i]-wMVup.iloc[j,i])*2
        else:
            TO4_i.iloc[0,i]=TO4_i.iloc[0,i]+wMVup.iloc[j,i]
            
            
        if my_list_down.count(integ_Top_Down_30[i+61].keys()[j])>0 and integ_Top_Down_30[i+61].keys()[j]==integ_Top_Down_30[i+60].keys()[j]:
            TO4_i.iloc[0,i]=TO4_i.iloc[0,i]
        elif my_list_down.count(integ_Top_Down_30[i+61].keys()[j])>0 and integ_Top_Down_30[i+61].keys()[j]!=integ_Top_Down_30[i+60].keys()[j]:
            TO4_i.iloc[0,i]=TO4_i.iloc[0,i]+(wMVdown.iloc[my_list_down.index(integ_Top_Down_30[i+61].keys()[j]),i]-wMVdown.iloc[j,i])*2
        else:
            TO4_i.iloc[0,i]=TO4_i.iloc[0,i]+wMVdown.iloc[j,i]

TC4_i=0.004*TO4_i
TC4_i.insert(0,"",0)
TC4_i.columns=FCF_data.iloc[:,60:252].columns

i_returns_port4.columns = TC4_i.columns
i_returns_port4 = i_returns_port4-TC4_i
for i in range (192):    
    if i==0 :
        i_cum_returns4.iloc[0,i]=1+i_returns_port4.iloc[0,i]
        
    else :
        i_cum_returns4.iloc[0,i]=(1+i_returns_port4.iloc[0,i])*i_cum_returns4.iloc[0,i-1]
plt.plot(i_cum_returns4.iloc[0,:])

risk_free.columns = ['RF']
return_df4= i_returns_port4.T
return_df4.index=risk_free.index
excessReturn4 = return_df4.sub(risk_free['RF'], axis=0)
geometric_mean4 = ((1+return_df4).cumprod().iloc[-1]**(1/len(return_df4))-1)
meanReturn4 = return_df4.mean()*12
volReturn4 = return_df4.std()*(12**0.5)
SR4 = excessReturn4.mean()*12/volReturn4
skewness4=return_df4.skew()
kurtosis4=return_df4.kurtosis()
turnover4 = TO4_i.mean(axis=1)*12






i_returns_port_ = pd.DataFrame(i_returns_port.iloc[0,60:]).T
for i in range (192):    
    if i==0 :
        i_cum_returns.iloc[0,i]=1+i_returns_port_.iloc[0,i]
        
    else :
        i_cum_returns.iloc[0,i]=(1+i_returns_port_.iloc[0,i])*i_cum_returns.iloc[0,i-1]
        

i_returns_port_1 = pd.DataFrame(i_returns_port1.iloc[0,60:]).T
for i in range (192):    
    if i==0 :
        i_cum_returns1.iloc[0,i]=1+i_returns_port_1.iloc[0,i]
        
    else :
        i_cum_returns1.iloc[0,i]=(1+i_returns_port_1.iloc[0,i])*i_cum_returns1.iloc[0,i-1]
    
i_cum_returns_plot=pd.DataFrame(i_cum_returns.iloc[0,0:192]).T
i_cum_returns_plot.columns = risk_free.index
i_cum_returns_plot1=pd.DataFrame(i_cum_returns1.iloc[0,0:192]).T
i_cum_returns_plot1.columns = risk_free.index
i_cum_returns2.columns= risk_free.index
i_cum_returns4.columns= risk_free.index

# =============================================================================
# 
# 
# =============================================================================

plt.plot(i_cum_returns_plot.iloc[0,:], color='b', label='EW')
plt.plot( i_cum_returns_plot1.iloc[0,:], color='r', label='RW')
plt.plot( i_cum_returns2.iloc[0,:], color='g', label='ERC')
plt.plot( i_cum_returns4.iloc[0,:], color='y', label='MV')
plt.legend();

Sharpe_ratio =pd.DataFrame(pd.concat([SR, SR1, SR2, SR4])).T
Annualized_returns =pd.DataFrame(pd.concat([meanReturn, meanReturn1, meanReturn2, meanReturn4])).T
Volatility = pd.DataFrame(pd.concat([volReturn, volReturn1, volReturn2, volReturn4])).T
Skewnessf = pd.DataFrame(pd.concat([skewness, skewness1, skewness2, skewness4])).T
Kurtosisf = pd.DataFrame(pd.concat([kurtosis, kurtosis1, kurtosis2, kurtosis4])).T
turnoverf = pd.DataFrame(pd.concat([turnover, turnover1, turnover2, turnover4])).T

test= pd.concat([Annualized_returns, Volatility, Sharpe_ratio, Skewnessf, Kurtosisf, turnoverf ])
test.columns = ['EW', 'RW', 'ERC','MV']
test.index = ['Annualized returns', "Volatility", 'Sharpe ratio', 'Skewness', 'Kurtosis', 'Turn over']


# =============================================================================
# =============================================================================
# #fn command 4 makes a whole paragraph into comment
# # =============================================================================
#         
# =============================================================================
