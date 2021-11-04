#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import pandas as pd
import os
from multiprocessing import Pool
from tqdm import tqdm
import multiprocessing

import time

from tqdm import tqdm

os.chdir("/home/darren/Desktop/jim_code")
#os.chdir("/media/darren/My Passport/Jim/jim_code/count")

#Read folder years, to loop through later when gathering data
yrs = os.listdir("/home/darren/synology//IFCB/class_files/CNN_class_files_2021/cell_chain_corrected/port_aransas")
yrs = [int(s) for s in yrs if s.isdigit()]

plt.rcParams["font.family"] = "arial"

pd.set_option('display.max_columns', None)

import calendar

from functions import builder #Gets chain length data and builds a dataframe
from functions import JulianDate_to_MMDDYYY

from functions import convert_dates

num_partitions = multiprocessing.cpu_count()-3 #number of partitions to split dataframe
num_cores = num_partitions #number of cores on your machine


# In[4]:


def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


# In[5]:


data = pd.DataFrame(columns = ['file','vol','major','minor','class','cells'])
for ii in yrs:
    path = "/home/darren/synology//IFCB/class_files/CNN_class_files_2021/cell_chain_corrected/port_aransas/"+str(ii)+'/'
    os.chdir(path)
    files = os.listdir(path)
    f = pd.DataFrame(columns = ['file'])
    f.file = files
    #Multithread time
    #start = time.time()
    x = parallelize_dataframe(f, builder)
    data = data.append(x, ignore_index = True)
    #end = time.time()
    #print(end-start)


