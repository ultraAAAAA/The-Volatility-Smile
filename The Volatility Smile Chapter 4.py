#!/usr/bin/env python
# coding: utf-8

# ## Replicate a Variance Swap with Options

# In[66]:


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


# ## Basic Paramater

# In[88]:


stock_price = list(np.linspace(20,200,2000))
tenor = 200/365
vol = 0.25
color_list = ['#F0F8FF','#00FFFF','#7FFFD4','#F0FFFF','#F5F5DC',
              '#A52A2A','#DEB887','#FF8C00','#8FBC8F','#483D8B',
              '#B22222','#FFFAF0','#228B22','#FF00FF','#DCDCDC',
              '#FFFFF0','#F0E68C','#E6E6FA','#FFF0F5','#7CFC00','#FFFACD',]


# ## Calculating Kappa, Average Kappa, Weighted Kappa

# In[89]:


def function_kappa(k):
    kappa = []
    for i in range(len(stock_price)):
        d1 = (1/(vol*math.sqrt(tenor))*math.log(stock_price[i]/k)) + vol*math.sqrt(tenor)/2
        temp = (stock_price[i]*math.sqrt(tenor)/(2*vol*math.sqrt(2*math.pi)))*math.exp(-0.5*d1**2)
        kappa.append(temp)
    return kappa


# In[92]:


average_kappa = weight_kappa = 0
strike_list = np.linspace(20,180,17)
weight = list(map(lambda num:1/(num*num), list(strike_list)))
plt.figure(figsize=(10,6))

for i in range(len(strike_list)):
    plt.scatter(stock_price, function_kappa(strike_list[i]), s = 5, c = color_list[i], marker='o',label = strike_list[i])
    average_kappa +=  np.array(function_kappa(strike_list[i]))/len(strike_list)
    weight_kappa += np.array((weight[i]/np.sum(weight))*pd.Series(function_kappa(strike_list[i])))

plt.xlabel('S')
plt.ylabel('Kappa')
plt.legend(loc = 0)
plt.show()


# In[93]:


plt.xlabel('S')
plt.ylabel('Kappa')
plt.scatter(stock_price, average_kappa, s=5, c="r", marker='o',label = 'Average')
plt.scatter(stock_price, weight_kappa, s=5, c="b", marker='o', label = '1/K^2')
plt.legend(loc = 0)
plt.show()


# ## The Value of the Replicated Portfolio at Expiration

# In[119]:


s_star = 100
stock_price2 = np.linspace(5,300,3000)
payoff = (stock_price2 - s_star)/s_star - np.log(stock_price2/s_star)
plt.xlabel('S')
plt.ylabel('Value(T)')
plt.scatter(stock_price2, payoff, s=5, c="r", marker='o',)
plt.show()


# In[ ]:




