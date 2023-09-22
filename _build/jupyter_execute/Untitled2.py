#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


file_ = '/home/rodolpho/Projects/grabatus/grabatus-basketAnalysis/gbt_service/vendas_itens.xlsx'

file_


# In[3]:


df = pd.read_excel(file_)


# In[16]:


df.head()


# In[44]:


ids = df['id'].unique()

list_sales = [ df[df['id'] == id_i][' item'].to_list() for id_i in ids ]
             
list_sales              


# In[63]:


mercado = pd.read_excel('/home/rodolpho/Downloads/mercado.xlsx', header=None)
mercado = mercado.T


# In[78]:


mercado[3].dropna().to_list()


# In[91]:


id = []
product = []

for i in mercado.columns:
    for j in range(len(mercado[i].dropna())):
        id.append(i)
        product.append(mercado[i][j])


# In[95]:


df = pd.DataFrame(data={'id_client': id, 'product': product})
df.to_excel('/home/rodolpho/Projects/grabatus/grabatus-basketAnalysis/gbt_service/vendas_itens.xlsx')

