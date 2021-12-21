#!/usr/bin/env python
# coding: utf-8

# # TESTE

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


plt.hist(np.random.normal(5, 10, 100), rwidth=0.95, density=True)
plt.grid(ls='--')
plt.show()

