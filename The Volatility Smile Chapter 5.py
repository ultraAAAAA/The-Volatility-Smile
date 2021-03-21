#!/usr/bin/env python
# coding: utf-8

# ## Simulation of Geometric Brownian Motion 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import matplotlib.mlab as mlab 
from scipy.stats import norm


# In[2]:


s0 = 10
mu = 0.05
T = 1
n_time = 100
dt = T/n_time
n_sim = 1000
vol = 0.25
s = pd.DataFrame()


# In[91]:


for i in range(0,n_sim):
    step = np.exp((mu-vol**2/2)*dt)*np.exp(vol*np.random.normal(0,np.sqrt(dt),(1,n_time)))
    temp = pd.DataFrame(s0*step.cumprod())
    s = pd.concat([s,temp],axis=1)

plt.plot(s)
plt.xlabel('t')
plt.ylabel('S_t')
plt.title('Simulation of Geometric Brownian Motion')
plt.show()


# In[92]:


n, bins, patches = plt.hist(s.iloc[-1])
plt.xlabel('S_T')  
plt.ylabel('Probability') 
plt.title('The histogram of S_T with LogNomral fit')
plt.show()


# In[36]:


# n, bins, patches = plt.hist(s.iloc[-1],bins = 20,range=[0, 30], rwidth=0.8,density=0.9, facecolor='blue', alpha=0.5)
# s_mean = np.mean(s.iloc[-1])
# s_std = np.std(s.iloc[-1])
# y = norm.pdf(bins, s_mean, s_std)#拟合一条最佳正态分布曲线y
# plt.plot(bins, y, 'r--')  
# plt.xlabel('S_T')  
# plt.ylabel('Probability') 
# plt.title('The histogram of S_T with LogNomral fit')
# plt.show()


# ## P&L of Delta Hedging

# In[29]:


vol_R = 0.3
vol_I = 0.2
s0 = k = 50
T = 1
n_time = 100
n_sim = 10
dt = T/n_time
mu = 0.08
rf = 0.04


# In[30]:


def delta_function(s,vol,tenor):
    d1 = (math.log(s/k) + (rf + 0.5*vol**2)*tenor)/(vol*math.sqrt(tenor))
    delta = norm.cdf(d1,loc=0,scale=1)
    return delta


# In[31]:


def gamma_function(s,tenor):
    d1 = (math.log(s/k) + (rf + 0.5*vol_I**2)*tenor)/(vol_I*math.sqrt(tenor))
    gamma = norm.pdf(d1,loc=0,scale=1)/(s*vol_I*math.sqrt(tenor))
    return gamma


# In[32]:


pl = pd.DataFrame()
tenor_list = np.linspace(1,0,n_time+1).tolist()##1001
for j in range(0,n_sim):
    temp_pl = []
    s = pd.DataFrame()
    dz = np.random.normal(0,np.sqrt(dt),(1,n_time))#1000
    step = np.exp((mu-vol_R**2/2)*dt + vol_R*dz)
    temp = [50] + (s0*step.cumprod()).tolist()
    s = pd.concat([s,pd.DataFrame(temp)],axis=1)#1001

    for i in range(0,n_time):
        d_pl = 0.5*gamma_function(float(s.iloc[i]),tenor_list[i])*float(s.iloc[i])**2*(vol_R**2 - vol_I**2)*dt 
        + (delta_function(float(s.iloc[i]),vol_I,tenor_list[i]) - delta_function(float(s.iloc[i]),vol_R,tenor_list[i]))*((mu-rf)*float(s.iloc[i])*dt + vol_R*float(s.iloc[i])*dz[0][i])
        temp_pl.append(d_pl*math.exp(-rf*(1-tenor_list[i])))
        
    pl = pd.concat([pl,pd.DataFrame(np.array(temp_pl).cumsum())],axis=1)        


# In[33]:


plt.plot(pl)
plt.xlabel('t')
plt.ylabel('P&L')
plt.title('P&L of Delta Hedging')
plt.show()


# In[ ]:




