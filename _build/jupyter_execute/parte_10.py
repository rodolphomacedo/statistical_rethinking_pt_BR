#!/usr/bin/env python
# coding: utf-8

# # 10 - Grande Entropia e Modelos Lineares Generalizados

# In[1]:


import numpy as np

from scipy import stats

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import pandas as pd

import networkx as nx
# from causalgraphicalmodels import CausalGraphicalModel

import arviz as az
# ArviZ ships with style sheets!
# https://python.arviz.org/en/stable/examples/styles.html#example-styles
az.style.use("arviz-darkgrid")

import xarray as xr

import stan
import nest_asyncio

plt.style.use('default')
plt.rcParams['axes.facecolor'] = 'lightgray'

# To DAG's
import daft
from causalgraphicalmodels import CausalGraphicalModel


# In[2]:


# Add fonts to matplotlib to run xkcd

from matplotlib import font_manager

font_dirs = ["fonts/"]  # The path to the custom font file.
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)


# In[3]:


# To make plots like drawing 
# plt.xkcd()


# In[4]:


# To running the stan in jupyter notebook
nest_asyncio.apply()


# ### R Code 10.1

# In[5]:


buckets = {
    "A": (0, 0, 1, 0, 0),
    "B": (0, 1, 8, 1, 0),
    "C": (0, 2, 6, 2, 0),
    "D": (1, 2, 4, 2, 1),
    "E": (2, 2, 2, 2, 2),
}

df = pd.DataFrame.from_dict(buckets)
df


# ### R Code 10.2

# In[6]:


# Normalize

p_norm = df / df.sum(axis=0)
p_norm


# ### R Code 10.3

# In[7]:


def entropy(bucket):
    uncertainty = []
    
    for q in bucket:
        if q == 0:
            uncertainty.append(q)
        else:
            uncertainty.append(q * np.log(q))
    
    return (-1) * np.sum(uncertainty)


# In[8]:


H = [entropy(p_norm[key]) for key in p_norm.keys()]

df_H = pd.DataFrame(H).T
df_H.columns = p_norm.keys()
df_H


# ### R Code 10.4

# In[9]:


ways = (1, 90, 1260, 37800, 113400)

logwayspp = np.log(ways) / 10

logwayspp


# In[10]:


plt.figure(figsize=(16, 6))

plt.plot(logwayspp, df_H.T.values, '--', c='black')
plt.plot(logwayspp, df_H.T.values, 'o', ms=10)

plt.title('Entropy in Buckets')
plt.xlabel('log(ways) per pebble')
plt.ylabel('entropy')

plt.show()


# ### R Code 10.5

# In[11]:


# Build list of canditate distributions
p = [
    [1/4, 1/4, 1/4, 1/4],
    [2/6, 1/6, 1/6, 2/6],
    [1/6, 2/6, 2/6, 1/6],
    [1/8, 4/8, 2/8, 1/8],
]

# Compute the expected values of each
result = [np.sum(np.multiply(p_i, [0, 1, 1, 2])) for p_i in p]
result


# ### R Code 10.6

# In[12]:


# Compute the entropy of each distribution

for p_i in p:
    print(-np.sum(p_i * np.log(p_i)))


# ### R Code 10.7

# In[13]:


p = 0.7

A = [
    (1-p)**2,
    p*(1-p),
    (1-p)*p,
    (p)**2,
]

np.round(A, 3)


# ### R Code 10.8

# In[14]:


- np.sum(A * np.log(A))


# ### R Code 10.9

# In[15]:


def sim_p(G=1.4):
    x = np.random.uniform(0, 1, size=4)
    x[3] = 0  # Removing the last random number x4
    
    x[3] = ( G * np.sum(x) - x[1] - x[2] ) / (2 - G)
    
    p = x / np.sum(x)
    
    return [-np.sum(p * np.log(p)), p]


# ### R Code 10.10

# In[16]:


H = pd.DataFrame([ sim_p(1.4) for _ in range(10000)], columns=('entropies', 'distributions'))

plt.figure(figsize=(17, 6))

plt.hist(H.entropies, density=True, rwidth=0.9)

plt.title('Entropy - Binomial')
plt.xlabel('Entropy')
plt.ylabel('Density')

plt.show()


# ### R Code 10.11

# In[17]:


# entropies = H.entropies
# distributions = H.distributions


# ### R Code 10.12

# In[18]:


H.entropies.max()


# ### R Code 10.13

# In[19]:


H.loc[H.entropies == H.entropies.max(), 'distributions'].values

