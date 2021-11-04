#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import os

#os.chdir('C:\\Users\\Jim\\OneDrive - Texas A&M University\\JMF\\work\\hrr\\porta\\count\\')

#files = os.listdir()


# In[2]:


def JulianDate_to_MMDDYYY(y,jd):
    import calendar
    
    month = 1
    day = 0
    while jd - calendar.monthrange(y,month)[1] > 0 and month <= 12:
        jd = jd - calendar.monthrange(y,month)[1]
        month = month + 1
    return(month,jd,y)


# In[2]:


def builder(df):
    
    out = pd.DataFrame(columns = ['file','vol','major','minor','class','cells'])
    
    for i in df.file:
        test = pd.read_csv(i)
        test = test.iloc[:,[0,1,3,4,16,17]]

        test.columns = ['file','vol','major','minor','class','cells']

        test.file[:] = i
        
        test = test[(test['class'] == 'Dinophysis') | (test['class'] == 'Mesodinium_rubrum') | (test['class'] == 'Teleaulax') | (test['class'] == 'Asterionellopsis') | (test['class'] == 'Skeletonema') | (test['class'] == 'Pseudonitzschia') | (test['class'] == 'Asterionellopsis_single') | (test['class'] == 'Chaetoceros_single') | (test['class'] == 'Chaetoceros_peruvianus') | (test['class'] == 'Chaetoceros_socialis') | (test['class'] == 'Karenia_brevis') | (test['class'] == 'Prorocentrum_cordatum')]
        
        out = out.append(test)
        
    return out


# In[2]:


def convert_dates(df):
    
    for i in df.index:
        if 'IFCB3' in df.file[i] or 'IFCB6' in df.file[i]:
            m,d,y = JulianDate_to_MMDDYYY(int(df.file[i][6:10]), int(df.file[i][11:14]))
            df.year[i] = y
            df.month[i] = m
            df.day[i] = d
            df.hour[i] = int(df.file[i][15:17])
            df.minute[i] = int(df.file[i][17:19])
            df.sec[i] = '00'
        
        else:
            df.year[i] = df.file[i][1:5]
            df.month[i] = df.file[i][5:7]
            df.day[i] = df.file[i][7:9]
            df.hour[i] = df.file[i][10:12]
            df.minute[i] = df.file[i][12:14]
            df.sec[i] = '00'
    
    return df

