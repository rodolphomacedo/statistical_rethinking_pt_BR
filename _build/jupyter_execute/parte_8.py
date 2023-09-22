#!/usr/bin/env python
# coding: utf-8

# # 8 - Os peixes-bois condicionais

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


# plt.xkcd()


# In[4]:


# To running the stan in jupyter notebook
nest_asyncio.apply()


# ### R Code 8.1

# In[5]:


df = pd.read_csv('./data/rugged.csv', sep=';')
df.head()


# In[6]:


df['log_gdp'] = np.log(df['rgdppc_2000'])  # Log version of outcome
df[['rgdppc_2000', 'log_gdp']].head()


# In[7]:


ddf = df[~np.isnan(df['log_gdp'].values)].copy()

ddf['log_gdp_std'] = ddf['log_gdp'] / np.mean(ddf['log_gdp'])
ddf['rugged_std'] = ddf['rugged'] / np.max(ddf['rugged'])

ddf[['log_gdp_std', 'rugged_std']]


# In[8]:


plt.figure(figsize=(17, 8))

plt.hist(ddf['log_gdp_std'], density=True, rwidth=0.9)
plt.title('log_gdp \n 1: mean; \n 0.8: 80% of the average \n 1.1: 10% more than average')
plt.axvline(x=1, c='r', ls='--')
plt.show()

plt.figure(figsize=(17, 8))
plt.hist(ddf['rugged_std'], density=True, rwidth=0.9)
plt.title('Rugged \n min:0 and max:1')
plt.axvline(x=np.mean(ddf['rugged_std']), c='r', ls='--')
plt.show()


# ### R Code 8.2

# In[9]:


model = """
    data {
        int N;
        vector[N] log_gdp_std;
        vector[N] rugged_std;
        real rugged_std_average;
    }
    
    parameters {
        real alpha;
        real beta;
        real<lower=0> sigma;
    }
    
    transformed parameters {
        vector[N] mu;
        
        mu = alpha + beta * (rugged_std - rugged_std_average);
    }
    
    model {
        // Prioris
        
        alpha ~ normal(1, 1);
        beta ~ normal(0, 1);
        sigma ~ exponential(1);
        
        // Likelihood
        log_gdp_std ~ normal(mu, sigma);
    }
    
    generated quantities {
        vector[N] log_lik;  // By default, if a variable log_lik is present in the Stan model, it will be retrieved as pointwise log likelihood values.
        vector[N] log_gdp_std_hat;
        
        for(i in 1:N){
            log_lik[i] = normal_lpdf(log_gdp_std[i] | mu[i], sigma);
            log_gdp_std_hat[i] = normal_rng(mu[i], sigma);
        }
    }
"""

data = {
    'N': len(ddf),
    'log_gdp_std': ddf['log_gdp_std'].values,
    'rugged_std': ddf['rugged_std'].values,
    'rugged_std_average': ddf['rugged_std'].mean(),
}

posteriori = stan.build(model, data=data)
samples = posteriori.sample(num_chains=4, num_samples=1000)


# In[10]:


# Transform to dataframe pandas
df_samples = samples.to_frame()
df_samples.head()


# In[11]:


# stan_fit to arviz_stan

stan_data = az.from_pystan(
    posterior=samples,
    posterior_predictive="log_gdp_std_hat",
    observed_data=['log_gdp_std'],
    prior=samples,
    prior_model=posteriori,
    posterior_model=posteriori,
)


# In[12]:


stan_data


# In[13]:


az.summary(stan_data, var_names=['alpha', 'beta', 'sigma'])


# ### R Code 8.3

# In[14]:


plt.figure(figsize=(17, 8))

alpha = np.random.normal(1, 1, 1000)
beta = np.random.normal(0, 1, 1000)

rugged_seq = np.linspace(0, 1, 100)

for i in range(100):
    plt.plot(rugged_seq, alpha[i] + beta[i] * rugged_seq, c='blue')
    
plt.axhline(y=1.3, c='r', ls='--')    
plt.axhline(y=0.7, c='r', ls='--')    
    
plt.ylim((0.5, 1.5))
plt.title('Using vague priors')
plt.xlabel('Ruggedness')
plt.ylabel('Log GDP')

plt.show()


# ### R Code 8.4

# In[15]:


plt.figure(figsize=(17, 8))

plt.hist(beta, bins=30, rwidth=0.9)
plt.axvline(x=0.6, c='r', ls='--')
plt.axvline(x=-0.6, c='r', ls='--')
plt.show()


# In[16]:


np.sum(np.sum(np.abs(beta) > 0.6) / len(beta))


# ### R Code 8.5

# In[17]:


plt.figure(figsize=(17, 8))

alpha = np.random.normal(1, 0.1, 1000)
beta = np.random.normal(0, 0.3, 1000)

rugged_seq = np.linspace(0, 1, 100)

for i in range(100):
    plt.plot(rugged_seq, alpha[i] + beta[i] * rugged_seq, c='blue')
    
plt.axhline(y=1.3, c='r', ls='--')    
plt.axhline(y=0.7, c='r', ls='--')    

plt.ylim((0.5, 1.5))
plt.title('Using a informative prior')
plt.xlabel('Ruggedness')
plt.ylabel('Log GDP')

plt.show()


# In[18]:


model2 = """
    data {
        int N;
        vector[N] log_gdp_std;
        vector[N] rugged_std;
        real rugged_std_average;
    }
    
    parameters {
        real alpha;
        real beta;
        real<lower=0> sigma;
    }
    
    transformed parameters {
        vector[N] mu;
        
        mu = alpha + beta * (rugged_std - rugged_std_average);
    
    }
    
    model {
        // Prioris
        
        alpha ~ normal(1, 0.1);
        beta ~ normal(0, 0.3);
        sigma ~ exponential(1);
        
        // Likelihood
        log_gdp_std ~ normal(mu, sigma);
    }
    
    generated quantities {
        vector[N] log_lik;  // By default, if a variable log_lik is present in the Stan model, it will be retrieved as pointwise log likelihood values.
        vector[N] log_gdp_std_hat;
        
        for(i in 1:N){
            log_lik[i] = normal_lpdf(log_gdp_std[i] | mu[i], sigma);
            log_gdp_std_hat[i] = normal_rng(mu[i], sigma);
        }
    }
"""

data = {
    'N': len(ddf),
    'log_gdp_std': ddf['log_gdp_std'].values,
    'rugged_std': ddf['rugged_std'].values,
    'rugged_std_average': ddf['rugged_std'].mean(),
}

posteriori = stan.build(model2, data=data)
samples = posteriori.sample(num_chains=4, num_samples=1000)


# In[19]:


stan_data2 = az.from_pystan(
    posterior=samples,
    posterior_predictive="log_gdp_std_hat",
    observed_data=['log_gdp_std'],
    prior=samples,
    prior_model=posteriori,
    posterior_model=posteriori,
)


# In[20]:


az.summary(stan_data, var_names=['alpha', 'beta', 'sigma'])


# ### R Code 8.7

# In[21]:


ddf['cid'] = [1 if cont_africa == 1 else 2 for cont_africa in ddf['cont_africa']]
ddf[['cont_africa', 'cid']].head()


# ### R Code 8.8

# In[22]:


model3 = """
    data {
        int N;
        vector[N] log_gdp_std;
        vector[N] rugged_std;
        array[N] int cid;  // Must be integer because this is index to alpha.
        real rugged_std_average;
    }
    
    parameters {
        real alpha[2];  //Can be used to real alpha[2] or array[2] int alpha;
        real beta;
        real<lower=0> sigma;
    }
    
    transformed parameters {
        vector[N] mu;
        
        for (i in 1:N){
            mu[i] = alpha[ cid[i] ] + beta * (rugged_std[i] - rugged_std_average);
        }
    }
    
    model {
        // Prioris
        
        alpha ~ normal(1, 0.1);
        beta ~ normal(0, 0.3);
        sigma ~ exponential(1);
        
        // Likelihood
        log_gdp_std ~ normal(mu, sigma);
    }
    
    generated quantities {
        vector[N] log_lik;
        vector[N] log_gdp_std_hat;
        
        for(i in 1:N){
            log_lik[i] = normal_lpdf(log_gdp_std[i] | mu[i], sigma);
            log_gdp_std_hat[i] = normal_rng(mu[i], sigma);
        }
    }
"""

data = {
    'N': len(ddf),
    'log_gdp_std': ddf['log_gdp_std'].values,
    'rugged_std': ddf['rugged_std'].values,
    'rugged_std_average': ddf['rugged_std'].mean(),
    'cid': ddf['cid'].values,
}

posteriori = stan.build(model3, data=data)
samples = posteriori.sample(num_chains=4, num_samples=1000)


# In[23]:


stan_data3 = az.from_pystan(
    posterior=samples,
    posterior_predictive="log_gdp_std_hat",
    observed_data=['log_gdp_std'],
    prior=samples,
    prior_model=posteriori,
    posterior_model=posteriori,
    dims={
        "alpha": ["africa"],
    },
)


# ### R Code 8.9

# In[24]:


models_8 = { 'm8.1': stan_data2, 'm8.2': stan_data3 }

az.compare(models_8, ic='waic')


# ### R Code 8.10

# In[25]:


az.summary(stan_data3, var_names=['alpha', 'beta', 'sigma'])


# ### R Code 8.11

# In[26]:


stan_data3


# In[27]:


alpha_a1 = stan_data3.posterior.alpha.sel(africa=0)
alpha_a2 = stan_data3.posterior.alpha.sel(africa=1)

diff_alpha_a1_a2 = az.extract(alpha_a1 - alpha_a2).alpha.values

az.hdi(diff_alpha_a1_a2, hdi_prob=0.89)


# ### R Code 8.12

# In[28]:


# Extract 200 samples from arviz-fit to numpy
params_post = az.extract(stan_data3.posterior, num_samples=200)


# In[29]:


rugged_seq = np.linspace(0, 1, 30)

log_gdp_mean_africa = []
log_gdp_hdi_africa = []

log_gdp_mean_not_africa = []
log_gdp_hdi_not_africa = []

# Calculation posterior mean and interval HDI
for i in range(len(rugged_seq)):
        log_gdp_africa = params_post.alpha.sel(africa=0) + params_post.beta.values * rugged_seq[i]
        log_gdp_mean_africa.append(np.mean(log_gdp_africa.values))
        log_gdp_hdi_africa.append(az.hdi(log_gdp_africa.values, hdi_prob=0.89))
        
        log_gdp_not_africa = params_post.alpha.sel(africa=1) + params_post.beta.values * rugged_seq[i]
        log_gdp_mean_not_africa.append(np.mean(log_gdp_not_africa.values))
        log_gdp_hdi_not_africa.append(az.hdi(log_gdp_not_africa.values, hdi_prob=0.89))
        
log_gdp_hdi_africa = np.array(log_gdp_hdi_africa)
log_gdp_hdi_not_africa = np.array(log_gdp_hdi_not_africa) 


# In[30]:


plt.figure(figsize=(17, 8))

plt.plot(rugged_seq, log_gdp_mean_africa, c='blue')
plt.fill_between(rugged_seq, log_gdp_hdi_africa[:,0], log_gdp_hdi_africa[:,1], color='blue', alpha=0.3)

plt.plot(rugged_seq, log_gdp_mean_not_africa, c='black')
plt.fill_between(rugged_seq, log_gdp_hdi_not_africa[:,0], log_gdp_hdi_not_africa[:,1], color='darkgray', alpha=0.3)

plt.plot(ddf.loc[ddf.cid == 1,'rugged_std'], ddf.loc[ddf.cid==1, 'log_gdp_std'], 'o', markerfacecolor='blue', color='gray')
plt.plot(ddf.loc[ddf.cid == 2,'rugged_std'], ddf.loc[ddf.cid==2, 'log_gdp_std'], 'o', markerfacecolor='none', color='black')

plt.title('m8.4')
plt.xlabel('ruggedness (standardized)')
plt.ylabel('log GDP (as proportion of mean)')

plt.text(0.8, 1.04, 'Not Africa', fontsize='x-large', color='black')
plt.text(0.8, 0.86, 'Africa',  fontsize='x-large', color='darkblue')

plt.show()


# ### R Code 8.13

# In[31]:


model4 = """
    data {
        int N;
        vector[N] log_gdp_std;
        vector[N] rugged_std;
        array[N] int cid;  // Must be integer because this is index to alpha.
        real rugged_std_average;
    }
    
    parameters {
        real alpha[2];  //Can be used to real alpha[2] or array[2] int alpha;
        real beta[2];
        real<lower=0> sigma;
    }
    
    transformed parameters {
        vector[N] mu;
        
        for (i in 1:N){
            mu[i] = alpha[ cid[i] ] + beta[ cid[i] ] * (rugged_std[i] - rugged_std_average);
        }
    }
    
    model {
        // Prioris
        
        alpha ~ normal(1, 0.1);
        beta ~ normal(0, 0.3);
        sigma ~ exponential(1);
        
        // Likelihood
        log_gdp_std ~ normal(mu, sigma);
    }
    
    generated quantities {
        vector[N] log_lik;
        vector[N] log_gdp_std_hat;
        
        for(i in 1:N){
            log_lik[i] = normal_lpdf(log_gdp_std[i] | mu[i], sigma);
            log_gdp_std_hat[i] = normal_rng(mu[i], sigma);
        }
    }
"""

data = {
    'N': len(ddf),
    'log_gdp_std': ddf['log_gdp_std'].values,
    'rugged_std': ddf['rugged_std'].values,
    'rugged_std_average': ddf['rugged_std'].mean(),
    'cid': ddf['cid'].values,
}

posteriori = stan.build(model4, data=data)
samples = posteriori.sample(num_chains=4, num_samples=1000)


# In[32]:


stan_data4 = az.from_pystan(
    posterior=samples,
    posterior_predictive="log_gdp_std_hat",
    observed_data=['log_gdp_std'],
    prior=samples,
    prior_model=posteriori,
    posterior_model=posteriori,
    dims={
        "alpha": ["africa"],
        "beta": ["africa"],
    },
)


# ### R Code 8.14

# In[33]:


az.summary(stan_data4, var_names=['alpha', 'beta', 'sigma'])


# ### R Code 8.15

# In[34]:


models_8 = { 'm8.1': stan_data2, 'm8.2': stan_data3, 'm8.3': stan_data4 }

# https://python.arviz.org/en/stable/api/generated/arviz.compare.html
# https://rss.onlinelibrary.wiley.com/doi/10.1111/1467-9868.00353
az.compare(models_8, ic='loo')  # loo is same that PSIS


# ### R Code 8.16

# In[35]:


az.loo(stan_data4, pointwise=True)


# In[36]:


stan_data4


# ### R Code 8.16 - XXX

# In[37]:


# az.plot_loo_pit(stan_data4, y='log_gdp_std')


# ### R Code 8.17

# In[38]:


# Extract 200 samples from arviz-fit to numpy
params_post = az.extract(stan_data4.posterior, num_samples=200)


# In[39]:


rugged_seq = np.linspace(0, 1, 30)

log_gdp_mean_africa = []
log_gdp_hdi_africa = []

log_gdp_mean_not_africa = []
log_gdp_hdi_not_africa = []

# Calculation posterior mean and interval HDI
for i in range(len(rugged_seq)):
        log_gdp_africa = params_post.alpha.sel(africa=0) + params_post.beta.sel(africa=0).values * rugged_seq[i]
        log_gdp_mean_africa.append(np.mean(log_gdp_africa.values))
        log_gdp_hdi_africa.append(az.hdi(log_gdp_africa.values, hdi_prob=0.89))
        
        log_gdp_not_africa = params_post.alpha.sel(africa=1) + params_post.beta.sel(africa=1).values * rugged_seq[i]
        log_gdp_mean_not_africa.append(np.mean(log_gdp_not_africa.values))
        log_gdp_hdi_not_africa.append(az.hdi(log_gdp_not_africa.values, hdi_prob=0.89))
        
log_gdp_hdi_africa = np.array(log_gdp_hdi_africa)
log_gdp_hdi_not_africa = np.array(log_gdp_hdi_not_africa)


# In[40]:


fig = plt.figure(figsize=(17, 8))

gs = GridSpec(1, 2)

# Africa plots
ax_africa = fig.add_subplot(gs[0])

ax_africa.plot(rugged_seq, log_gdp_mean_africa, c='blue')
ax_africa.fill_between(rugged_seq, log_gdp_hdi_africa[:,0], log_gdp_hdi_africa[:,1], color='blue', alpha=0.3)
ax_africa.plot(ddf.loc[ddf.cid == 1,'rugged_std'], ddf.loc[ddf.cid==1, 'log_gdp_std'], 'o', markerfacecolor='blue', color='gray')

ax_africa.set_title('African Nations')
ax_africa.set_xlabel('ruggedness (standardized)')
ax_africa.set_ylabel('log GDP (as proportion of mean)')
ax_africa.text(0.7, 0.8, 'Africa',  fontsize='xx-large', color='darkblue')


# Non africa plots
ax_not_africa = fig.add_subplot(gs[1])

ax_not_africa.plot(rugged_seq, log_gdp_mean_not_africa, c='black')
ax_not_africa.fill_between(rugged_seq, log_gdp_hdi_not_africa[:,0], log_gdp_hdi_not_africa[:,1], color='darkgray', alpha=0.3)
ax_not_africa.plot(ddf.loc[ddf.cid == 2,'rugged_std'], ddf.loc[ddf.cid==2, 'log_gdp_std'], 'o', markerfacecolor='none', color='black')

ax_not_africa.set_title('Non-African Nations')
ax_not_africa.set_xlabel('ruggedness (standardized)')
ax_not_africa.set_ylabel('log GDP (as proportion of mean)')
ax_not_africa.text(0.7, 1.1, 'Not Africa', fontsize='xx-large', color='black')


plt.show()


# ### R Code 8.18

# In[41]:


log_gdp_delta_mean = []
log_gdp_delta_hdi = []

# Calculation posterior mean and interval HDI
for i in range(len(rugged_seq)):
        log_gdp_africa = params_post.alpha.sel(africa=0) + params_post.beta.sel(africa=0).values * rugged_seq[i]
        log_gdp_not_africa = params_post.alpha.sel(africa=1) + params_post.beta.sel(africa=1).values * rugged_seq[i]
        
        log_gdp_delta = log_gdp_africa - log_gdp_not_africa

        log_gdp_delta_mean.append(np.mean(log_gdp_delta.values))
        log_gdp_delta_hdi.append(az.hdi(log_gdp_delta.values, hdi_prob=0.89))
        
log_gdp_delta_hdi = np.array(log_gdp_delta_hdi)


# In[42]:


fig = plt.figure(figsize=(17, 8))

gs = GridSpec(1, 1)

# Africa delta log GDP plots
ax_delta = fig.add_subplot(gs[0])

ax_delta.plot(rugged_seq, log_gdp_delta_mean, c='black')
ax_delta.fill_between(rugged_seq, log_gdp_delta_hdi[:,0], log_gdp_delta_hdi[:,1], color='blue', alpha=0.3)

ax_delta.set_title('Delta log GDP')
ax_delta.set_xlabel('ruggedness (standardized)')
ax_delta.set_ylabel('expected difference log GDP')

ax_delta.text(0.0, 0.02, 'Africa higher GDP',  fontsize='xx-large', color='darkblue')
ax_delta.text(0.0, -0.03, 'Africa lower GDP',  fontsize='xx-large', color='darkblue')
ax_delta.axhline(y=0, ls='--', color='black')

plt.show()


# ### R Code 8.19

# In[43]:


df = pd.read_csv('./data/tulips.csv', sep=';',
                 dtype={
                     'bed': 'category',  # cluster of plants from same section of the greenhouse
                     'water': 'float',  # Predictor: Soil moisture - (1) low and (3) high 
                     'shade': 'float',  # Predictor: Light exposure - (1) high and (3) low
                     'blooms': 'float',  # What we wish to predict
                 })
df.tail()


# In[44]:


df.describe(include='category')


# In[45]:


df.describe().T


# ### R Code 8.20

# In[46]:


df['blooms_std'] = df['blooms'] / df['blooms'].max()
df['water_cent'] = df['water'] - df['water'].mean()
df['shade_cent'] = df['shade'] - df['shade'].mean()


# ### R Code 8.21

# $$ \alpha \sim Normal(0.5, 1) $$

# In[47]:


alpha = np.random.normal(0.5, 1, 1000)

alphas_true = np.any([alpha < 0, alpha > 1], axis=0)  # If (alpha < 0) or (alpha > 1) then True else False

np.sum(alphas_true) / len(alpha)


# ### R Code 8.22

# $$ \alpha \sim Normal(0.5, 0.25) $$

# In[48]:


alpha = np.random.normal(0.5, 0.25, 1000)

alphas_true = np.any([alpha < 0, alpha > 1], axis=0)  # If (alpha < 0) or (alpha > 1) then True else False

np.sum(alphas_true) / len(alpha)


# ### R Code 8.23

# In[49]:


model = """
    data {
        int N;
        vector[N] blooms_std;
        vector[N] water_cent;
        vector[N] shade_cent;
    }
    
    parameters {
        real alpha;
        real beta_w;
        real beta_s;
        real<lower=0> sigma;
    }
    
    transformed parameters {
        vector[N] mu;
        
        mu = alpha + beta_w * water_cent + beta_s * shade_cent;
    }
    
    model {
        // Priori
        
        alpha ~ normal(0.5, 0.25);
        beta_w ~ normal(0, 0.25);
        beta_s ~ normal(0, 0.25);
        sigma ~ exponential(1);
        
        blooms_std ~ normal(mu, sigma);
    }
    
    generated quantities {
        vector[N] log_lik;
        vector[N] blooms_std_hat;
        
        for(i in 1:N){
            log_lik[i] = normal_lpdf(blooms_std[i] | mu[i], sigma);
            blooms_std_hat[i] = normal_rng(mu[i], sigma);
        }
    }
"""

data = {
    'N': len(df),
    'blooms_std': df.blooms_std.values,   
    'water_cent': df.water_cent.values,
    'shade_cent': df.shade_cent.values,
}

posteriori = stan.build(model, data=data)
samples = posteriori.sample(num_chains=4, num_samples=1000)


# In[50]:


stan_blooms = az.from_pystan(
    prior_model=posteriori,
    prior=samples,
    posterior_model=posteriori,
    posterior=samples,
    posterior_predictive="blooms_std_hat",
    observed_data=['blooms_std', 'water_cent', 'shade_cent'],
)

stan_blooms


# In[51]:


az.summary(stan_blooms, var_names=['alpha', 'beta_w', 'beta_s'])


# ### R Code 8.24

# $$ B_i \sim Normal(\mu_i, \sigma) $$
# 
# $$ \mu_i = \alpha + \beta_W W_i + \beta_S S_i + \beta_{WS} S_i W_i $$

# In[52]:


model = """
    data {
        int N;
        vector[N] blooms_std;
        vector[N] water_cent;
        vector[N] shade_cent;
    }
    
    parameters {
        real alpha;
        real beta_w;
        real beta_s;
        real beta_ws;
        real<lower=0> sigma;
    }
    
    transformed parameters {
        vector[N] mu;
        
        mu = alpha + beta_w * water_cent + beta_s * shade_cent + beta_ws * water_cent .* shade_cent;
    }
    
    model {
        // Priori
        
        alpha ~ normal(0.5, 0.25);
        beta_w ~ normal(0, 0.25);
        beta_s ~ normal(0, 0.25);
        beta_ws ~ normal(0, 0.25);
        sigma ~ exponential(1);
        
        blooms_std ~ normal(mu, sigma);
    }
    
    generated quantities {
        vector[N] log_lik;
        vector[N] blooms_std_hat;
        
        for(i in 1:N){
            log_lik[i] = normal_lpdf(blooms_std[i] | mu[i], sigma);
            blooms_std_hat[i] = normal_rng(mu[i], sigma);
        }
    }
"""

data = {
    'N': len(df),
    'blooms_std': df.blooms_std.values,   
    'water_cent': df.water_cent.values,
    'shade_cent': df.shade_cent.values,
}

posteriori = stan.build(model, data=data)
samples = posteriori.sample(num_chains=4, num_samples=1000)


# In[53]:


stan_blooms_interaction = az.from_pystan(
    prior_model=posteriori,
    prior=samples,
    posterior_model=posteriori,
    posterior=samples,
    posterior_predictive="blooms_std_hat",
    observed_data=['blooms_std', 'water_cent', 'shade_cent'],
)

stan_blooms_interaction


# In[54]:


az.summary(stan_blooms_interaction, var_names=['alpha', 'beta_w', 'beta_s', 'beta_ws'])


# ### R Code 8.25

# In[55]:


blooms_post = az.extract(stan_blooms.posterior, num_samples=20)  # get 20 lines
blooms_post_int = az.extract(stan_blooms_interaction.posterior, num_samples=20)  # get 20 lines


# In[56]:


# ================
# Plot without interactions

water_seq = np.linspace(-1, 1, 30)

fig = plt.figure(figsize=(17, 6))

gs = GridSpec(1, 3)

shade_cent_values = [-1, 0, 1]

ax = [None] * len(shade_cent_values)

for j, shade_plot_aux in enumerate(shade_cent_values):
    ax[j] = fig.add_subplot(gs[j])

    for i in range(len(blooms_post.alpha)):
        mu = blooms_post.alpha[i].values +              blooms_post.beta_w[i].values * water_seq +              blooms_post.beta_s[i].values * (shade_plot_aux)

        ax[j].plot(water_seq, mu, c='gray')
        ax[j].set_ylim(0, 1)
        ax[j].set_title(f'8.4 post: Shade = ${shade_plot_aux}$')
        ax[j].set_xlabel('water')
        ax[j].set_ylabel('blooms')
        ax[j].set_xticks(shade_cent_values, shade_cent_values)
        ax[j].set_yticks([0, 0.5, 1.0], [0, 0.5, 1])

    ax[j].plot(
            df.loc[df['shade_cent'] == shade_plot_aux,'water_cent'].values, 
            df.loc[df['shade_cent'] == shade_plot_aux,'blooms_std'].values, 
            'o', c='blue')
    
plt.show()

# ================
# Plot with interactions

water_seq = np.linspace(-1, 1, 30)

fig = plt.figure(figsize=(17, 6))

gs = GridSpec(1, 3)

shade_cent_values = [-1, 0, 1]

ax = [None] * len(shade_cent_values)

for j, shade_plot_aux in enumerate(shade_cent_values):
    ax[j] = fig.add_subplot(gs[j])

    for i in range(len(blooms_post_int.alpha)):
        mu = blooms_post_int.alpha[i].values +              blooms_post_int.beta_w[i].values * water_seq +              blooms_post_int.beta_s[i].values * (shade_plot_aux) +              blooms_post_int.beta_ws[i].values * water_seq * (shade_plot_aux)

        ax[j].plot(water_seq, mu, c='gray')
        ax[j].set_ylim(0, 1)
        ax[j].set_title(f'8.5 post: Shade = ${shade_plot_aux}$')
        ax[j].set_xlabel('water')
        ax[j].set_ylabel('blooms')
        ax[j].set_xticks(shade_cent_values, shade_cent_values)
        ax[j].set_yticks([0, 0.5, 1.0], [0, 0.5, 1])

    ax[j].plot(
            df.loc[df['shade_cent'] == shade_plot_aux,'water_cent'].values, 
            df.loc[df['shade_cent'] == shade_plot_aux,'blooms_std'].values, 
            'o', c='blue')
    
plt.show()


# ### R Code 8.26

# In[57]:


# ================
# Plot without interactions

# Prior
alpha_prior =  np.random.normal(0.5, 0.25, 20)
beta_w_prior =  np.random.normal(0, 0.25, 20)
beta_s_prior =  np.random.normal(0, 0.25, 20)

water_seq = np.linspace(-1, 1, 30)

fig = plt.figure(figsize=(17, 6))

gs = GridSpec(1, 3)

shade_cent_values = [-1, 0, 1]

ax = [None] * len(shade_cent_values)

for j, shade_plot_aux in enumerate(shade_cent_values):
    ax[j] = fig.add_subplot(gs[j])

    for i in range(len(alpha_prior)):
        mu = alpha_prior[i] +              beta_w_prior[i] * water_seq +              beta_s_prior[i] * (shade_plot_aux)

        ax[j].plot(water_seq, mu, c='gray')
        ax[j].set_ylim(0, 1)
        ax[j].set_title(f'8.4 prior: Shade = ${shade_plot_aux}$')
        ax[j].set_xlabel('water')
        ax[j].set_ylabel('blooms')
        ax[j].set_xticks(shade_cent_values, shade_cent_values)
        ax[j].set_yticks([0, 0.5, 1.0], [0, 0.5, 1])
        ax[j].set_ylim(-1, 2)
        ax[j].axhline(y=1, ls='--', color='red')
        ax[j].axhline(y=0, ls='--', color='red')

    ax[j].plot(
            df.loc[df['shade_cent'] == shade_plot_aux,'water_cent'].values, 
            df.loc[df['shade_cent'] == shade_plot_aux,'blooms_std'].values, 
            'o', c='blue')
    
plt.show()

# ================
# Plot with interactions

# Prior
alpha_prior =  np.random.normal(0.5, 0.25, 20)
beta_w_prior =  np.random.normal(0, 0.25, 20)
beta_s_prior =  np.random.normal(0, 0.25, 20)
beta_ws_prior =  np.random.normal(0, 0.25, 20)

water_seq = np.linspace(-1, 1, 30)

fig = plt.figure(figsize=(17, 6))

gs = GridSpec(1, 3)

shade_cent_values = [-1, 0, 1]

ax = [None] * len(shade_cent_values)

for j, shade_plot_aux in enumerate(shade_cent_values):
    ax[j] = fig.add_subplot(gs[j])

    for i in range(len(alpha_prior)):
        mu = alpha_prior[i] +              beta_w_prior[i] * water_seq +              beta_s_prior[i] * (shade_plot_aux) +              beta_ws_prior[i] * water_seq * (shade_plot_aux)

        ax[j].plot(water_seq, mu, c='gray')
        ax[j].set_ylim(0, 1)
        ax[j].set_title(f'8.5 post: Shade = ${shade_plot_aux}$')
        ax[j].set_xlabel('water')
        ax[j].set_ylabel('blooms')
        ax[j].set_xticks(shade_cent_values, shade_cent_values)
        ax[j].set_yticks([0, 0.5, 1.0], [0, 0.5, 1])
        ax[j].set_ylim(-1, 2)
        ax[j].axhline(y=1, ls='--', color='red')
        ax[j].axhline(y=0, ls='--', color='red')

    ax[j].plot(
            df.loc[df['shade_cent'] == shade_plot_aux,'water_cent'].values, 
            df.loc[df['shade_cent'] == shade_plot_aux,'blooms_std'].values, 
            'o', c='blue')
    
plt.show()

