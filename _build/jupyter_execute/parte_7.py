#!/usr/bin/env python
# coding: utf-8

# # 7 - Compasso de Ulisses

# In[1]:


import numpy as np

from scipy import stats

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import pandas as pd
from videpy import Vide

import networkx as nx
# from causalgraphicalmodels import CausalGraphicalModel

import stan
import nest_asyncio

plt.style.use('default')

plt.rcParams['axes.facecolor'] = 'lightgray'

# To DAG's
import daft
from causalgraphicalmodels import CausalGraphicalModel


# In[2]:


# To running the stan in jupyter notebook
nest_asyncio.apply()


# ### R Code 7.1 - Pag 194

# In[3]:


sppnames = ('afarensis', 'africanus', 'habilis',  'boisei', 'rudolfensis', 'ergaster', 'sapiens')
brainvolcc = (438, 452, 612, 521,  752, 871, 1350)
masskg = (37.0, 35.5, 34.5, 41.5, 55.5, 61.0, 53.5)

d = pd.DataFrame({'species':sppnames, 'brain': brainvolcc, 'mass':masskg})
d


# In[4]:


fig, ax = plt.subplots(figsize=(15, 7))
ax.scatter(d.mass, d.brain, marker='o', s=50)

for i, text in enumerate(sppnames):
    ax.annotate(text, (d.mass[i]+0.3, d.brain[i]+10))
    
ax.set_title('Figure 7.2 - Relationshiip between Brain and Body hominin species')
ax.set_xlabel('Body mass')
ax.set_ylabel('Brain volume')

ax.grid(ls='--', color='white', alpha=0.4)
plt.show()


# ### R Code 7.2 - pag 196

# In[5]:


np.std(d.mass, ddof=1)


# In[6]:


d.mass.std()


# In[7]:


d['mass_std'] = (d.mass - d.mass.mean())/d.mass.std()
d['brain_std'] = d.brain / np.max(d.brain)
d


# In[8]:


fig, ax = plt.subplots(figsize=(15, 7))
ax.scatter(d.mass_std, d.brain_std, marker='o', s=50)

for i, text in enumerate(sppnames):
    ax.annotate(text, (d.mass_std[i]+0.03, d.brain_std[i]+0.03))
    
ax.set_title('Figure 7.2 - Relationshiip between Brain and Body hominin species')
ax.set_xlabel('Body mass - Rescaling')
ax.set_ylabel('Brain volume - Rescaling')

ax.grid(ls='--', color='white', alpha=0.4)
plt.show()


# ### R Code 7.3 - pag 196

# Modelo linear:
# 
# $\begin{align}
# b_i \sim Normal(\mu_i, \sigma) \\ 
# \mu_i = \alpha + \beta m_i \\
# \alpha \sim Normal(0.5, 1) \\
# \beta \sim Normal(0, 10) \\
# \sigma \sim LogNormal(0, 1) \\
# \end{align}$
# 
# 
# 
# 

# In[9]:


model = """
    data {
        int N;
        vector[N] brain;
        vector[N] body;
    }
    
    parameters {
        real alpha;
        real beta;
        real<lower=0> sigma;
        // real log_sigma;  // Like the book
    }
    
    model {
        vector[N] mu;
        
        mu = alpha + beta * body;
                
        // Prioris
        alpha ~ normal(0.5, 1);
        beta ~ normal(0, 10);
        sigma ~ lognormal(0, 1);
        // log_sigma ~ normal(0, 1);  // Like the book 
        
        // Likelihood
        brain ~ normal(mu, sigma);
        // brain ~ normal(mu, exp(sigma));  // Like the book
    }
"""

data = {
    'N': len(d.brain_std),
    'brain': d.brain_std.values,
    'body': d.mass_std.values,
}

posteriori_1 = stan.build(model, data=data)
samples_1 = posteriori_1.sample(num_chains=4, num_samples=1000)


# In[10]:


Vide.summary(samples_1)


# In[11]:


Vide.plot_forest(samples_1)


# ### R Code 7.4 - pag 196
# 
# Just example code in R
# 
# `m7.1_OLS <- lm(brain_std ~ brain_std, data=d)`

# ### R Code 7.5 - pag 197

# In[12]:


def var2(x):
    return np.sum(np.power(x - np.mean(x), 2))/len(x)


# In[13]:


mean_std = [np.mean((samples_1['alpha'].flatten() + samples_1['beta'].flatten() * mass)) for mass in d.mass_std.values]
r = mean_std - d.brain_std

resid_var = var2(r)
outcome_var = var2(d.brain_std)

1 - resid_var/outcome_var


# ### R Code 7.6 - Pag 197

# In[14]:


def R2_is_bad():
    pass


# ### R Code 7.7 - pag 198

# In[15]:


model = """
    data {
        int N;
        vector[N] brain;
        vector[N] body;
    }
    
    parameters {
        real alpha;
        real beta[2];
        real<lower=0> sigma;
    }
    
    model {
        vector[N] mu;
        
        mu = alpha + beta[1] * body + beta[2] * body^2;
        
        // Prioris
        alpha ~ normal(0.5, 1);
        for (j in 1:2){
            beta[j] ~ normal(0, 10);
        }
        sigma ~ lognormal(0, 1);    
        
        // Likelihood
        brain ~ normal(mu, sigma);
    }
"""

data = {
    'N': len(d.brain_std),
    'brain': d.brain_std.values,
    'body': d.mass_std.values,
}

posteriori_2 = stan.build(model, data=data)
samples_2 = posteriori_2.sample(num_chains=4, num_samples=1000)


# In[16]:


Vide.summary(samples_2)


# In[17]:


Vide.plot_forest(samples_2)


# ### R Code 7.8 - Pag 198

# In[18]:


model = """
    data {
        int N;
        vector[N] brain;
        vector[N] body;
    }
    
    parameters {
        real alpha;
        real beta[3];
        real<lower=0> sigma;
    }
    
    model {
        vector[N] mu;
        
        mu = alpha + beta[1] * body   + 
                     beta[2] * body^2 +
                     beta[3] * body^3;
                     
        // Likelihood
        brain ~ normal(mu, sigma);
        
        // Prioris
        alpha ~ normal(0.5, 1);
        for (j in 1:3){
            beta[j] ~ normal(0, 10);
        }
        sigma ~ lognormal(0, 1);    
    }
"""

data = {
    'N': len(d.brain_std),
    'brain': d.brain_std.values,
    'body': d.mass_std.values,
}

posteriori_3 = stan.build(model, data=data)
samples_3 = posteriori_3.sample(num_chains=4, num_samples=1000)


# In[19]:


model = """
    data {
        int N;
        vector[N] brain;
        vector[N] body;
    }
    
    parameters {
        real alpha;
        real beta[4];
        real<lower=0> sigma;
    }
    
    model {
        vector[N] mu;
        
        mu = alpha + beta[1] * body   + 
                     beta[2] * body^2 +
                     beta[3] * body^3 +
                     beta[4] * body^4;
                     
        // Likelihood
        brain ~ normal(mu, sigma);
        
        // Prioris
        alpha ~ normal(0.5, 1);
        for (j in 1:4){
            beta[j] ~ normal(0, 10);
        }
        sigma ~ lognormal(0, 1);    
    }
"""

data = {
    'N': len(d.brain_std),
    'brain': d.brain_std.values,
    'body': d.mass_std.values,
}

posteriori_4 = stan.build(model, data=data)
samples_4 = posteriori_4.sample(num_chains=4, num_samples=1000)


# In[20]:


model = """
    data {
        int N;
        vector[N] brain;
        vector[N] body;
    }
    
    parameters {
        real alpha;
        real beta[5];
        real<lower=0> sigma;
    }
    
    model {
        vector[N] mu;
        
        mu = alpha + beta[1] * body   + 
                     beta[2] * body^2 +
                     beta[3] * body^3 +
                     beta[4] * body^4 +
                     beta[5] * body^5;
                     
        // Likelihood
        brain ~ normal(mu, sigma);
        
        // Prioris
        alpha ~ normal(0.5, 1);
        for (j in 1:5){
            beta[j] ~ normal(0, 10);
        }
        sigma ~ lognormal(0, 1);    
    }
"""

data = {
    'N': len(d.brain_std),
    'brain': d.brain_std.values,
    'body': d.mass_std.values,
}

posteriori_5 = stan.build(model, data=data)
samples_5 = posteriori_5.sample(num_chains=4, num_samples=1000)


# ### R Code 7.9 - Pag 199

# In[21]:


model = """
    data {
        int N;
        vector[N] brain;
        vector[N] body;
    }
    
    parameters {
        real alpha;
        real beta[6];
    }
    
    model {
        vector[N] mu;
        
        mu = alpha + beta[1] * body   + 
                     beta[2] * body^2 +
                     beta[3] * body^3 +
                     beta[4] * body^4 +
                     beta[5] * body^5 +
                     beta[6] * body^6;
                     
        // Likelihood
        brain ~ normal(mu, 0.001);
        
        // Prioris
        alpha ~ normal(0.5, 1);
        for (j in 1:6){
            beta[j] ~ normal(0, 10);
        }    
    }
"""

data = {
    'N': len(d.brain_std),
    'brain': d.brain_std.values,
    'body': d.mass_std.values,
}

posteriori_6 = stan.build(model, data=data)
samples_6 = posteriori_6.sample(num_chains=4, num_samples=1000)


# ### R Code 7.10 - Pag 199

# In[22]:


mass_seq = np.linspace(start=d.mass_std.min(), stop=d.mass_std.max(), num=100)


# In[23]:


pp_1 = [samples_1['alpha'].flatten() + samples_1['beta'].flatten() * mass_i for mass_i in mass_seq]


# In[24]:


pp_2 = [samples_2['alpha'].flatten() + 
        samples_2['beta'][0].flatten() * mass_i + 
        samples_2['beta'][1].flatten() * np.power(mass_i, 2) for mass_i in mass_seq]


# In[25]:


pp_3 = [samples_3['alpha'].flatten() + 
        samples_3['beta'][0].flatten() * mass_i + 
        samples_3['beta'][1].flatten() * np.power(mass_i, 2) + 
        samples_3['beta'][2].flatten() * np.power(mass_i, 3)
        for mass_i in mass_seq]


# In[26]:


pp_4 = [samples_4['alpha'].flatten() + 
        samples_4['beta'][0].flatten() * mass_i + 
        samples_4['beta'][1].flatten() * np.power(mass_i, 2) + 
        samples_4['beta'][2].flatten() * np.power(mass_i, 3) +
        samples_4['beta'][3].flatten() * np.power(mass_i, 4) 
        for mass_i in mass_seq]


# In[27]:


pp_5 = [samples_5['alpha'].flatten() + 
        samples_5['beta'][0].flatten() * mass_i + 
        samples_5['beta'][1].flatten() * np.power(mass_i, 2) + 
        samples_5['beta'][2].flatten() * np.power(mass_i, 3) +
        samples_5['beta'][3].flatten() * np.power(mass_i, 4) + 
        samples_5['beta'][4].flatten() * np.power(mass_i, 5) 
        for mass_i in mass_seq]


# In[28]:


pp_6 = [samples_6['alpha'].flatten() + 
        samples_6['beta'][0].flatten() * mass_i + 
        samples_6['beta'][1].flatten() * np.power(mass_i, 2) + 
        samples_6['beta'][2].flatten() * np.power(mass_i, 3) +
        samples_6['beta'][3].flatten() * np.power(mass_i, 4) + 
        samples_6['beta'][4].flatten() * np.power(mass_i, 5) + 
        samples_6['beta'][5].flatten() * np.power(mass_i, 6) 
        for mass_i in mass_seq]


# In[29]:


fig = plt.figure(figsize=(18, 20))

gs = GridSpec(nrows=3, ncols=2)

ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(d.mass_std.values, d.brain_std.values, c='black')
ax1.plot(mass_seq, pp_1, c='blue', lw=0.01)
ax1.set_ylim(0, 1.1)
ax1.set_title('$R^2: 0.51$')
ax1.set_xlabel('Body mass (KG)')
ax1.set_ylabel('Brain Volume (cc)')

ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(d.mass_std.values, d.brain_std.values, c='black')
ax2.plot(mass_seq, pp_2, c='blue', lw=0.01)
ax2.set_ylim(0, 1.1)
ax2.set_title('$R^2: 0.54$')
ax2.set_xlabel('Body mass (KG)')
ax2.set_ylabel('Brain Volume (cc)')

ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(d.mass_std.values, d.brain_std.values, c='black')
ax3.plot(mass_seq, pp_3, c='blue', lw=0.01)
ax3.set_ylim(0, 1.1)
ax3.set_title('$R^2: 0.69$')
ax3.set_xlabel('Body mass (KG)')
ax3.set_ylabel('Brain Volume (cc)')

ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(d.mass_std.values, d.brain_std.values, c='black')
ax4.plot(mass_seq, pp_4, c='blue', lw=0.01)
ax4.set_ylim(0, 1.1)
ax4.set_title('$R^2: 0.82$')
ax4.set_xlabel('Body mass (KG)')
ax4.set_ylabel('Brain Volume (cc)')

ax5 = fig.add_subplot(gs[2, 0])
ax5.scatter(d.mass_std.values, d.brain_std.values, c='black')
ax5.plot(mass_seq, pp_5, c='blue', lw=0.01)
ax5.set_ylim(0, 1.1)
ax5.set_title('$R^2: 0.99$')
ax5.set_xlabel('Body mass (KG)')
ax5.set_ylabel('Brain Volume (cc)')

ax6 = fig.add_subplot(gs[2, 1])
ax6.scatter(d.mass_std.values, d.brain_std.values, c='black')
ax6.plot(mass_seq, pp_6, c='blue', lw=0.01)
ax6.set_ylim(0, 1.1)
ax6.set_title('$R^2: 1$')
ax6.set_xlabel('Body mass (KG)')
ax6.set_ylabel('Brain Volume (cc)')

plt.show()


# ### R Code 7.11 - Pag 201

# In[30]:


d.brain_std


# Deletando a linha $i=3$ do dataframe:

# In[31]:


d_deleted_line_3 = d.brain_std.drop(3)
d_deleted_line_3


# ### R Code 7.12 - Pag 206

# **Entropia da informação**:  *A incerteza contida na distribuição de probabilidade* é a média do $log$ da probabilidade do evento.

# In[32]:


def H(p):
    """Information Entropy
    H(p):= -sum(p_i * log(pi))
    """
    if not np.sum(p) == 1:
        print('ProbabilityError: This is not probability, its sum not equal to 1.')
    else:
        return - np.sum([p_i * np.log(p_i) if p_i > 0 else 0 for p_i in p])


# In[33]:


p = (0.3, 0.7)  # (p_rain, p_sum)

H(p)


# A incerteza do clima de *Abu Dhabi* é menor, pois é pouco provável que chova.

# In[34]:


p_AbuDhabi = (0.01, 0.99)  # (p_rain, p_sum)

H(p_AbuDhabi)


# In[35]:


p_3dim = (0.7, 0.15, 0.15)  # (p_1, p_2, p_3)

H(p_3dim)


# ### R Code 7.13 - pag 210

# References:  
# 
# - https://stackoverflow.com/questions/49147905/how-to-extract-posterior-samples-of-log-likelihood-from-pystan
# 
# - https://mc-stan.org/docs/2_20/functions-reference/normal-distribution.html

# **Obs**:
# 
# Here I learned to use the two blocks in stan: *transformed parameters* and *generated quantities*.
# 
# To calculate the **Deviance** it is necessary to make these changes. 
# 
# That's why I did all 6 estimates again with the calculations needed.

# In[36]:


model = """
    data {
        int N;
        vector[N] brain;
        vector[N] body;
    }
    
    parameters {
        real alpha;
        real beta;
        real<lower=0> sigma;
    }
    
    transformed parameters {
        vector[N] mu;
        mu = alpha + beta * body;
    }
    
    model {
        // Prioris
        alpha ~ normal(0.5, 1);
        beta ~ normal(0, 10);
        sigma ~ lognormal(0, 1);    
        
        // Likelihood
        brain ~ normal(mu, sigma);
    }
    
    generated quantities {
        vector[N] log_lik;
        vector[N] brain_hat;
        
        for (i in 1:N){
            log_lik[i] = normal_lpdf(brain[i] | mu[i], sigma);
            brain_hat[i] = normal_rng(mu[i], sigma);
        }
        
    }
"""

data = {
    'N': len(d.brain_std),
    'brain': d.brain_std.values,
    'body': d.mass_std.values,
}

posteriori_1 = stan.build(model, data=data)
samples_1 = posteriori_1.sample(num_chains=4, num_samples=1000)


# In[37]:


model = """
    data {
        int N;
        vector[N] brain;
        vector[N] body;
    }
    
    parameters {
        real alpha;
        real beta[2];
        real<lower=0> sigma;
    }
    
    transformed parameters{
        vector[N] mu;
        mu = alpha + beta[1] * body + 
                     beta[2] * body^2;
    }
    
    model {
        // Prioris
        alpha ~ normal(0.5, 1);
        for (j in 1:2){
            beta[j] ~ normal(0, 10);
        }
        sigma ~ lognormal(0, 10);    
        
        // Likelihood
        brain ~ normal(mu, sigma);
    }
"""

data = {
    'N': len(d.brain_std),
    'brain': d.brain_std.values,
    'body': d.mass_std.values,
}

posteriori_2 = stan.build(model, data=data)
samples_2 = posteriori_2.sample(num_chains=4, num_samples=1000)


# In[38]:


model = """
    data {
        int N;
        vector[N] brain;
        vector[N] body;
    }
    
    parameters {
        real alpha;
        real beta[3];
        real<lower=0> sigma;
    }
    
    transformed parameters {
        vector[N] mu;
        mu = alpha + beta[1] * body   + 
                     beta[2] * body^2 +
                     beta[3] * body^3;
    }
    
    model {                     
        // Prioris
        alpha ~ normal(0.5, 1);
        for (j in 1:3){
            beta[j] ~ normal(0, 10);
        }
        sigma ~ lognormal(0, 10);   
        
        // Likelihood
        brain ~ normal(mu, sigma);
    }
"""

data = {
    'N': len(d.brain_std),
    'brain': d.brain_std.values,
    'body': d.mass_std.values,
}

posteriori_3 = stan.build(model, data=data)
samples_3 = posteriori_3.sample(num_chains=4, num_samples=1000)


# In[39]:


model = """
    data {
        int N;
        vector[N] brain;
        vector[N] body;
    }
    
    parameters {
        real alpha;
        real beta[4];
        real<lower=0> sigma;
    }
    
    transformed parameters {
        vector[N] mu;
        
        mu = alpha + beta[1] * body   + 
                     beta[2] * body^2 +
                     beta[3] * body^3 +
                     beta[4] * body^4;
    }
    
    model {
        // Prioris
        alpha ~ normal(0.5, 1);
        for (j in 1:4){
            beta[j] ~ normal(0, 10);
        }
        sigma ~ lognormal(0, 10); 
        
        // Likelihood
        brain ~ normal(mu, sigma);   
    }
"""

data = {
    'N': len(d.brain_std),
    'brain': d.brain_std.values,
    'body': d.mass_std.values,
}

posteriori_4 = stan.build(model, data=data)
samples_4 = posteriori_4.sample(num_chains=4, num_samples=1000)


# In[40]:


model = """
    data {
        int N;
        vector[N] brain;
        vector[N] body;
    }
    
    parameters {
        real alpha;
        real beta[5];
        real<lower=0> sigma;
    }
    
    transformed parameters {
        vector[N] mu;
        
        mu = alpha + beta[1] * body   + 
                     beta[2] * body^2 +
                     beta[3] * body^3 +
                     beta[4] * body^4 +
                     beta[5] * body^5;
    }
    
    model {          
        // Prioris
        alpha ~ normal(0.5, 1);
        for (j in 1:5){
            beta[j] ~ normal(0, 10);
        }
        sigma ~ lognormal(0, 10);    
        
        // Likelihood
        brain ~ normal(mu, sigma);
    }
    
    generated quantities {
        vector[N] log_lik;
        vector[N] brain_hat;
        
        for (i in 1:N){
            log_lik[i] = normal_lpdf(brain[i] | mu[i], sigma);
            brain_hat[i] = normal_rng(mu[i], sigma);
        }
        
    }
"""

data = {
    'N': len(d.brain_std),
    'brain': d.brain_std.values,
    'body': d.mass_std.values,
}

posteriori_5 = stan.build(model, data=data)
samples_5 = posteriori_5.sample(num_chains=4, num_samples=1000)


# In[41]:


model = """
    data {
        int N;
        vector[N] brain;
        vector[N] body;
    }
    
    parameters {
        real alpha;
        real beta[6];
    }
    
    transformed parameters {
        vector[N] mu;
        
        mu = alpha + beta[1] * body   + 
                     beta[2] * body^2 +
                     beta[3] * body^3 +
                     beta[4] * body^4 +
                     beta[5] * body^5 +
                     beta[6] * body^6;
    }
    
    model {          
        // Prioris
        alpha ~ normal(0.5, 1);
        for (j in 1:6){
            beta[j] ~ normal(0, 10);
        }    
        
        // Likelihood
        brain ~ normal(mu, 0.001);
    }
    
    generated quantities {
        vector[N] log_ll_y;
        
        for (i in 1:N){
            log_ll_y[i] = normal_lpdf(brain[i] | mu, 0.001);
        }
    }
"""

data = {
    'N': len(d.brain_std),
    'brain': d.brain_std.values,
    'body': d.mass_std.values,
}

posteriori_6 = stan.build(model, data=data)
samples_6 = posteriori_6.sample(num_chains=4, num_samples=1000)


# In[42]:


def lppd(samples, outcome, std_residuals=True):
    """ Calculate the LOG-POINTWISE-PREDICTIVE-DENSITY
    
    samples : stan
        Sampler results of fit posteriori. 
        Need 'mu' already computed.
    
    outcome : list
        List with outcomes the original data.
        
    std_residuals : booleans
        Compute lppd using std of the residuals 
    """
    mu = samples['mu']
    N = len(outcome)
    K = np.shape(mu)[1]  # Qty samples (mu) sampled from posteriori

    outcome = outcome.reshape(-1, 1)

    if std_residuals:
        sigma = (np.sum((mu - outcome)**2, 0) / N)**0.5    
    else:
        sigma = samples['sigma'].flatten()
        
    lppd = np.empty(N, dtype=float)
       
    for i in range(N):
        log_posteriori_predictive = stats.norm.logpdf(outcome[i], mu[i], sigma)

        lppd[i] = np.log(np.sum(np.exp(log_posteriori_predictive))) - np.log(K)
    
    return lppd    


# ### R Code 7.15

# Like the book and the [this](https://github.com/pymc-devs/pymc-resources/blob/main/Rethinking_2/Chp_07.ipynb) one the values here have slight differeces. 
# 
# *Need review calculations - pag 211 - 2ed*

# In[43]:


lppd_lm = []

lppd_lm.append(np.sum(lppd(samples_1, d.brain_std.values, std_residuals=True)))
lppd_lm.append(np.sum(lppd(samples_2, d.brain_std.values, std_residuals=True)))
lppd_lm.append(np.sum(lppd(samples_3, d.brain_std.values, std_residuals=True)))
lppd_lm.append(np.sum(lppd(samples_4, d.brain_std.values, std_residuals=True)))
lppd_lm.append(np.sum(lppd(samples_5, d.brain_std.values, std_residuals=True)))
lppd_lm.append(np.sum(lppd(samples_6, d.brain_std.values, std_residuals=True)))

lppd_lm


# In[44]:


import arviz as az
# ArviZ ships with style sheets!
# https://python.arviz.org/en/stable/examples/styles.html#example-styles
az.style.use("arviz-darkgrid")


# In[45]:


data = az.from_pystan(
    posterior=samples_5,
    posterior_predictive="brain_hat",
    observed_data=data,
    log_likelihood={"brain": "log_lik"},
)

data


# In[46]:


data5 = az.from_pystan(
    posterior=samples_5,
    posterior_predictive="brain_hat",
    observed_data=data,
    log_likelihood={"brain": "log_lik"},
)

data5


# In[47]:


y_true = d.brain_std.values
y_pred = data5.posterior_predictive.stack(sample=("chain", "draw"))["brain_hat"].values.T
az.r2_score(y_true, y_pred)


# In[48]:


aa = az.loo(samples_5, pointwise=True)


# In[51]:


az.plot_elpd({'samples_1':samples_1, 'samples_5': samples_5}, xlabels=True)
plt.show()


# In[50]:


aaa = data.posterior_predictive.brain_hat[0].values
np.log(np.sum(np.exp(aaa.T), 1)) - np.log(1000)


# ### R Code 7.16
# 
# Just using Rethinking Packages - pag 213

# ### R Code 7.17
# 
# Just using Rethinking Packages - Pag 213

# ### R Code 7.18
# 
# Just using Rethinking Packages - pag 214

# -------
