#!/usr/bin/env python
# coding: utf-8

# # Arviz - Examples

# In[1]:


import arviz as az
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# ArviZ ships with style sheets!
# https://python.arviz.org/en/stable/examples/styles.html#example-styles
az.style.use("arviz-darkgrid")


# In[3]:


y = np.random.normal(0, 1, 1000)


# In[4]:


outcome = az.plot_posterior(y)
plt.show()


# In[5]:


# rcParams -runtime configuration Params - configuração de tempo de execução
az.rcParams['stats.hdi_prob'] = 0.89
az.plot_forest((y, y**2))
plt.show()


# ## Plotting with PyStan objects

# In[6]:


import nest_asyncio
nest_asyncio.apply()


# In[7]:


import stan  # pystan version 3.6.0


# In[8]:


schools_code = """
data {
  int<lower=0> J;
  array[J] real y;
  array[J] real<lower=0> sigma;
}

parameters {
  real mu;
  real<lower=0> tau;
  array[J] real theta;
}

model {
  // Prioris
  mu ~ normal(0, 5);
  tau ~ cauchy(0, 5);
  theta ~ normal(mu, tau);
  
  // Likelihood
  y ~ normal(theta, sigma);
}
generated quantities {
    vector[J] log_lik;
    vector[J] y_hat;
    
    for (j in 1:J) {
        log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
        y_hat[j] = normal_rng(theta[j], sigma[j]);
    }
}
"""

schools_dat = {
    "J": 8,
    "y": [28, 8, -3, 7, -1, 1, 18, 12],
    "sigma": [15, 10, 16, 11, 9, 11, 10, 18],
}

schools = np.array(
    [
        "Choate",
        "Deerfield",
        "Phillips Andover",
        "Phillips Exeter",
        "Hotchkiss",
        "Lawrenceville",
        "St. Paul's",
        "Mt. Hermon",
    ])
posterior = stan.build(schools_code, data=schools_dat, random_seed=1)
fit = posterior.sample(num_chains=4, num_samples=1000)


# In[14]:


az.plot_density(fit, var_names=["mu", "tau"])


# ### arviz.from_pystan
# 
# https://python.arviz.org/en/stable/api/generated/arviz.from_pystan.html 
# 
# ```{python3}
# arviz.from_pystan(
#     posterior=None,             # PyStan fit object for posterior - Samples from posteriori.
#     *, 
#     posterior_predictive=None,  # Posterior predictive samples for the posterior.
#     predictions=None,           # Out-of-sample predictions for the posterior.
#     prior=None,                 # PyStan fit object for prior.
#     prior_predictive=None,      # Posterior predictive samples for the prior.
#     observed_data=None, 
#     constant_data=None, 
#     predictions_constant_data=None, 
#     log_likelihood=None, 
#     coords=None, 
#     dims=None, 
#     posterior_model=None, 
#     prior_model=None, 
#     save_warmup=None, 
#     dtypes=None
# )
# ```

# In[20]:


data = az.from_pystan(
    posterior=fit,
    posterior_predictive="y_hat",
    observed_data=["y"],
    log_likelihood={"y": "log_lik"},
    coords={"school": schools},
    dims={
        "theta": ["school"],
        "y": ["school"],
        "log_lik": ["school"],
        "y_hat": ["school"],
        "theta_tilde": ["school"],
    },
)
data


# In[21]:


az.plot_pair(
    data,
    coords={"school": ["Choate", "Deerfield", "Phillips Andover"]},
    divergences=True,
);


# In[22]:


az.plot_trace(data, compact=False)
plt.show()


# ## Meu Exemplo Simples 1

# In[84]:


N = 100
alpha = 5
beta = 3

x = np.linspace(0, 10, N)
y = np.array([alpha + beta * x_i + np.random.normal(0, 3, 1) for x_i in x]).flatten()

plt.plot(x, y, lw=0.3, c='blue')

plt.show()


# In[85]:


model = """
    data{
        int N;
        vector[N] x;
        vector[N] y;
    }
    
    parameters {
        real alpha;
        real beta;
        real<lower=0> sigma;
    }
    
    transformed parameters {
        vector[N] mu;
        
          mu = alpha + beta * x;
    }
    
    model {
        //Prioris
        alpha ~ normal(0, 4);
        beta ~ normal(0, 2);
        sigma ~ lognormal(0, 1);

      
        // Likelihood
        y ~ normal(mu, sigma);
    }
    
    generated quantities {
        vector[N] log_lik;
        vector[N] y_hat;
        
        for (i in 1:N){
            log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
            y_hat[i] = normal_rng(mu[i], sigma);
        }
        
    }
"""

data = {
    'N': N,
    'x': x,
    'y': y,
}

posteriori = stan.build(model, data=data)
samples = posteriori.sample(num_chains=4, num_samples=1000)


# In[86]:


az.plot_density(samples, var_names=['alpha', 'beta', 'sigma'])
plt.show()


# In[87]:


az.plot_forest(samples, var_names=["alpha", "beta", "sigma"])
plt.show()


# In[90]:


data = az.from_pystan(
    posterior=samples,
    posterior_predictive="y_hat",
    observed_data=data,
    log_likelihood={"y": "log_lik"},
)

data


# In[91]:


# Posteriori
col=['r', 'b', 'g', 'c']

for j in range(4):
    for i in range(1000):
        plt.plot(data.posterior.mu.values[j][i] + 10*j, c=col[j], lw=0.1)
plt.show()


# In[92]:


# Posteriori predictive

col=['r', 'b', 'g', 'c']

for j in range(4):
    for i in range(1000):
        plt.plot(data.posterior_predictive.y_hat[j][i] + 30*j, c=col[j], lw=0.1)
plt.show()


# In[219]:


# Log-Likelihood

col=['r', 'b', 'g', 'c']

for j in range(4):
    for i in range(1000):
        plt.plot(data.log_likelihood.y[j][i] + 10*j, c=col[j], lw=0.1)
plt.show()


# In[237]:


var_stats = [
    'acceptance_rate',
    'step_size',
    'tree_depth',
    'n_steps',
    'diverging',
    'energy'
]

for i in range(len(var_stats)):
    plt.plot(data.sample_stats[var_stats[i]], c='b', lw=0.1)
    plt.title(var_stats[i])
    plt.show()


# #### Trabalhando com InferenceData

# In[143]:


posterior = data.posterior
posterior


# In[150]:


#observed_data = data.observed_data
#observed_data


# In[152]:


data.to_netcdf("example0.nc")


# ## Trabalhando com *InferenceData*

# In[93]:


import arviz as az
import numpy as np
import xarray as xr
xr.set_options(display_expand_data=False, display_expand_attrs=False);


# In[94]:


idata = az.load_arviz_data("centered_eight")
idata


# In[95]:


post = idata.posterior


# ### Adicionando uma nova variável

# In[96]:


post['log_tau'] = np.log(post['tau'])
idata.posterior


# ### Combine as cadeias (chains) e amostras (draws) 

# In[97]:


stacked = az.extract(idata)
stacked


# ### Obtendo um subconjunto aleatório de amostras

# In[98]:


az.extract(idata, num_samples=100)

# To set seed

# az.extract(idata, num_samples=100, rng=3) 
# az.extract(idata, group="log_likelihood", num_samples=100, rng=3)


# ### Obtendo um array Numpy de um parâmetro

# In[99]:


stacked


# In[100]:


stacked.mu


# In[101]:


stacked.mu.values  # Array numpy


# ### Obtendo o tamanho das dimensões

# In[102]:


len(idata.observed_data.school)


# ### Obtendo os valores das coordenadas

# In[176]:


idata.observed_data.school


# ### Obtendo um subconjunto das cadeias (chains)

# In[103]:


idata.sel(chain=[0, 2])


# ### Removendo as primeiras n amostras (burn-in)

# In[104]:


# Removendo as primeiras 100 amostras de todos os grupos
idata.sel(draw=slice(100, None))

# Aqui será removido as primeiras 100 amostras apenas do grupo selecionado
# idata.sel(draw=slice(100, None), groups="posterior")


# ### Calculando a média dos valores das cadeias e das amostras

# In[183]:


# Calculando a média de todas as dimensões da posteriori
idata.posterior.mean()


# In[184]:


# Calculandoa média apenas das dimensões 'chain' e 'draw'
idata.posterior.mean(dim=['chain', 'draw'])


# ### Calculando e armazenando a posteriori "pushforward" das quantidades

# In[188]:


# Média móvel do log_tau de 50 posições
post["mlogtau"] = post["log_tau"].rolling({'draw': 50}).mean()
post

