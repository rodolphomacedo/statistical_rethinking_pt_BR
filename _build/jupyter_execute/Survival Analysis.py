#!/usr/bin/env python
# coding: utf-8

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


# # Análise de Sobrevivência

# Análise de sobrevivência, também denominada análise de sobrevida[1], é um ramo da estatística que estuda o tempo de duração esperado até a ocorrência de um ou mais eventos, tais como morte em organismos biológicos e falha em sistemas mecânicos.
# 
# A análise de sobrevivência procura responder perguntas como: 
# 
# - Qual é a proporção de uma população que sobreviverá depois de um certo tempo? 
# 
# - Daqueles que sobrevivem, a que ritmo eles vão morrer ou falhar? 
# 
# - Podem várias causas de morte ou falha ser levado em conta? 
# 
# - Como circunstâncias ou características específicas aumentam ou diminuem a probabilidade de sobrevivência?
# 
# #### Definições de termos comuns na análise de sobrevivência
# Os seguintes termos são comumente usados em análises de sobrevivência:
# 
# - EVENTO - Morte, ocorrência de doença, recorrência da doença, recuperação ou outra experiência de interesse
# 
# 
# - TEMPO - O tempo desde o início de um período de observação (como cirurgia ou início de tratamento) até (i) ocorrer um evento, ou (ii) finalizar o estudo, ou (iii) ocorrer a perda de contato ou retirada do estudo.
# 
# 
# - CENSURA - Se um sujeito não experimenta um evento durante o tempo de observação ele será descrito como censurado. O sujeito é censurado no sentido em que nada é observado ou conhecido sobre ele após o tempo de censura. Um sujeito censurado pode ou não ter um evento após o final tempo de observação.
# 
# 
# - FUNÇÃO DE SOBREVIVÊNCIA - É uma função, S, que associa a cada tempo t o número S (t) que é a probabilidade de que um sujeito sobreviva além do tempo t.

# Ref: [Um modelo de sobrevivência em Stan - Eren M. Elçi](https://ermeel86.github.io/case_studies/surv_stan_example.html)

# In[5]:


df = pd.read_csv('./data/mastectomy.csv', sep=",")
df['event'] = [ 1 if event_i == True else 0 for event_i in df['event'] ]
df.head(10)


# Mais precisamente, cada linha no conjunto de dados representa observações de uma mulher com diagnóstico de câncer de mama que foi submetida a mastectomia.

# Legenda:
# 
# - A coluna `time` representa o tempo (em meses) pós-operatório em que a mulher foi observada.
# 
# 
# - A coluna `event` indica se a mulher morreu ou não durante o período de observação.
# 
# 
# - A coluna `metastized` representa se o câncer tinha metástase antes da cirurgia.

# ### Descritiva

# In[6]:


df.groupby(by=['metastized', 'event']).describe()


# In[7]:


df.groupby(by=['metastized', 'event']).count().plot.bar()
plt.show()


# ### Modelo

# Função de sobrevivência $S(t)$:
# 
# $$ S(t) = \mathbb{P} (T > t) = e^{-H(t)} $$
# 
# 
# $T :=$ é o tempo de sobrevivência de um indivíduo.
# 
# $(T > t):=$ Tempo ($T$) que o paciente sobreviveu além do tempo $t$.
# 
# $H(t):=$ definido como perigo acumulado.
# 
# 

# In[8]:


model = """
    data {
        int<lower=1> N_uncensored;  // Number of individuals not censured - event == 1
        int<lower=1> N_censored;  // Number of individuals censured - event == 0
        int<lower=0> NC;  // Number of covariates
        
        matrix[N_censored, NC] X_censored;
        matrix[N_uncensored, NC] X_uncensored;
        
        vector<lower=0>[N_censored] times_censored;  // time to censured event                         
        vector<lower=0>[N_uncensored] times_uncensored;  // // time to non-censured event
    }
    
    parameters {
        vector[NC] betas;
        real intercept;
    }
    
    model {
        // Prioris
        betas ~ normal(0,2);                                                            
        intercept ~ normal(-5,2);                                                     
        
        // Likelihood
        target += exponential_lpdf(times_uncensored | exp(intercept + X_uncensored * betas)); 
        target += exponential_lccdf(times_censored | exp(intercept + X_censored * betas));  
    }
"""


# In[9]:


# Build the data dict to stan
covariates_name = ['metastized']  # From the inputs user


dat_list = {
    'N_uncensored': len(df[df['event'] == 1]),
    'N_censored': len(df[df['event'] == 0]),
    'NC': len(covariates_name),
    'X_censored': np.matrix(df.loc[df['event'] == 0, covariates_name]),
    'X_uncensored': np.matrix(df.loc[df['event'] == 1, covariates_name]),
    'times_censored': df.loc[df['event'] == 0, 'time'].values,
    'times_uncensored': df.loc[df['event'] == 1, 'time'].values
}


# In[10]:


posteriori = stan.build(model, data=dat_list)
samples = posteriori.sample(num_chains=4, num_samples=1000)


# In[11]:


survival = az.from_pystan(
    posterior=samples,
    posterior_model=posteriori,
    observed_data=list(dat_list.keys())
)


# In[12]:


az.summary(survival)


# In[13]:


az.plot_forest(survival, var_names=['betas', 'intercept'], combined=True, figsize=(17, 5), hdi_prob=0.89)
plt.show()


# In[30]:


import json

a = {
    'status': "done"
}

aaa = json.dumps(a)
json.loads(aaa)


# In[17]:


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html
# scale = 1 / lambda

x = np.linspace(0, max(df.time), 100)
intercpt = survival.posterior.intercept.values.flatten()
betas = survival.posterior.betas.values.flatten()

lam_x0 = 1.0 / np.exp(intercpt + betas*0)
lam_x1 = 1.0 / np.exp(intercpt + betas*1)

y_x0 = np.array([stats.expon.sf(x_i, scale=lam_x0) for x_i in x])
y_x1 = np.array([stats.expon.sf(x_i, scale=lam_x1) for x_i in x])

y_x0_hdi = np.array([az.hdi(y_x0[i], hdi_prob=0.89) for i in range(len(y_x0))])
y_x1_hdi = np.array([az.hdi(y_x1[i], hdi_prob=0.89) for i in range(len(y_x1))])

# Graph
# =====
plt.figure(figsize=(17, 9))

plt.plot(x, np.mean(y_x0, axis=1), color='blue', label='metastized=0')
plt.fill_between(x, y_x0_hdi[:, 1], y_x0_hdi[:, 0], color='blue', alpha=0.3)

plt.plot(x, np.mean(y_x1, axis=1), color='red', label='metastized=1')
plt.fill_between(x, y_x1_hdi[:, 1], y_x1_hdi[:, 0], color='red', alpha=0.3)

plt.ylim(-0.1, 1.1)
plt.legend()
plt.show()


# In[15]:


# Graph
# =====
plt.figure(figsize=(17, 9))

plt.plot(x, np.mean(y_x0 - y_x1, axis=1), color='blue', label='metastized diff')
plt.legend()
plt.show()

