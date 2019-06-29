import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing 
from sklearn.metrics import mean_squared_error
from math import sqrt
from math import sqrt
from multiprocessing import Process
import multiprocessing
from multiprocessing import Pool
from fbprophet import Prophet
import os
from fbprophet.diagnostics import cross_validation
#global data,Type,MRU,Programs,result,result_df,rmse_tracker,t,m,rms_h,rms_ma




def forecaster(x):
    #global data,Type,MRU,Programs
    
    data=pd.read_excel('C:/Users/saragada/Desktop/OPEX/Data/Opex_data.xlsx')

    data.head()

    data.Timestamp = pd.to_datetime(data.Date,format='%d-%m-%Y %H:%M')

    data.index = data.Timestamp 

    data['Other Program']=data['External Consulting'] + data['Operational Outsourced Services']+data['Training & Recruitment'] + data['Machinery & Equipment']+data['Property Costs']+data['Departmental Transfers']+data['Other Owned Expense']+data['Third Party Software Expense']+data['Telecommunications']


    Type=data['Type'].unique()
    dataMru=data['MRU'][(data.index > '2018-12-01')]

    MRU=dataMru.unique()

    

    data1=data[['Type','MRU','Compensation & Benefits','External Temporary Workforce','Travel','Other Program']]
    
    result={}

    result_df=pd.DataFrame()

    rmse_tracker=pd.DataFrame()
    t=[]
    m=[]
    rms_h=[]
    rms_ma=[]

    
    for i in Type[1:]:
        for j in MRU:
            df=data1[x][(data1['Type'].str.contains(i)) & (data1['MRU'].str.contains(j))]
            par=int(len(df)*0.9)
            train =df[:par]
            df_train=train.copy()
            df_train=df_train.reset_index()
            df_train.rename(columns={'Date': 'ds', x: 'y'}, inplace=True)
            test=df[par:]
            y_hat_avg = test.copy()
            y_hat_avg=y_hat_avg.reset_index()
            
            
            algo= Prophet()
            algo.fit(df_train)
            future = algo.make_future_dataframe(len(test), freq='M')
            forecast = algo.predict(future)
            forecast = algo.predict(future)
                        
                        
            y_hat_avg['Prophet']=pd.DataFrame(forecast[['yhat']])
            
            if len(df)>24:
                
                fit1 = ExponentialSmoothing(np.asarray(train) ,seasonal_periods=3 ,trend='add', seasonal='add',).fit()
                y_hat_avg['Holt_Winter'] = list(fit1.forecast(len(test)))
                
                
                
                
            forecast_test=train.reset_index()
            forecast_test=list(forecast_test[x])
            
            for future_period in range(0,6):
        
                forecast_test.append(sum(forecast_test[-6:])/ 6)
            y_hat_avg['Moving Average']=forecast_test[-len(test):]
            
            y_hat_avg['Mru']=j
            y_hat_avg['Type']=i
            
            
            result[str(i)+'  '+str(j)]=y_hat_avg
            t.append(i)
            m.append(j)
            
            result_df=result_df.append(y_hat_avg)
    
            
            try:
                
                rms_h.append(sqrt(mean_squared_error(y_hat_avg[x], y_hat_avg['Holt_Winter'])))
            except :
                
                rms_h.append('Na')
            rms_ma.append(sqrt(mean_squared_error(y_hat_avg[x], y_hat_avg['Moving Average'])))
            
            
            
        
    rmse_tracker['Type']=t
    rmse_tracker['Mru']=m
    rmse_tracker['Holt_Winter']=rms_h
    rmse_tracker['Moving Average']=rms_ma
    
    result_df.to_csv('C:/Users/saragada/Desktop/OPEX/Data/'+str(x)+'.csv')

for x in Programs:
    forecaster(x)


#if __name__ == '__main__':
#    Programs=['Compensation & Benefits','External Temporary Workforce','Travel','Other Program']
#    pool=multiprocessing.Pool()
#    pool.map(forecaster,Programs)
#    pool.close()
#    pool.join()




    
    
        
      
        



















            
        

##################### Test deposit ###########################
#
#df=data1['Other Program'][(data1['Type'].str.contains('RESEARCH & DEVELOPMENT')) & (data1['MRU'].str.contains('A450'))]  
#train =df[:-6]
#test=df[-6:]
#y_hat_avg = test.copy()
#y_hat_avg=y_hat_avg.reset_index()
#fit1 = ExponentialSmoothing(np.asarray(train) ,seasonal_periods=3 ,trend='add', seasonal='add',).fit()
#y_hat_avg['Holt_Winter'] = list(fit1.forecast(len(test)))
#
#forecast_test=train.reset_index()
#forecast_test=list(forecast_test['Other Program'])
#
#for future_period in range(0,3):
#    
#    forecast_test.append(sum(forecast_test[-3:])/ 3)









