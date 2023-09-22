#!/usr/bin/env python
# coding: utf-8

# # 6 - Os DAGs assombrados & O Terror Causal

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
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


# ### RCode 6.1 - Pag 162

# In[3]:


np.random.seed(1914)

N = 200
p = 0.1

# Não correlacionado noticiabilidade(newsworthiness) e confiabilidade(trustworthiness)
nw = np.random.normal(0, 1, N)
tw = np.random.normal(0, 1, N)

# Selecionando os 10% melhores
s = nw + tw  # Score total
q = np.quantile(s, 1-p)  # Top 10% 
selected = [ True if s_i >= q else False for s_i in s ]
print('Noticiabilidade(newsworthiness): \n\n', nw[selected], '\n\n')
print('Confiabilidade(trustworthiness):\n\n', tw[selected], '\n\n')

print('Correlação: ', np.correlate(tw[selected], nw[selected]))


# In[4]:


plt.figure(figsize=(17, 6))

plt.scatter(tw, nw, s=6, color='gray')
plt.scatter(tw[selected], nw[selected], s=7, color='blue')

plt.title('Figure 6.1 - Pag162')
plt.xlabel('Noticiabilidade ($newsworthiness$)')
plt.ylabel('Confiabilidade ($trustworthiness$)')

plt.grid(ls='--', color='white', alpha=0.3)


# ### RCode 6.2 - pag163
# 

# In[5]:


N = 100

np.random.seed(909)  # Teste com outras sementes

height = np.random.normal(10, 2,  N)

leg_proportion = np.random.uniform(0.4, 0.5, N)

leg_left  = np.random.left = leg_proportion * height + np.random.normal(0, 0.02, N)
leg_right = np.random.left = leg_proportion * height + np.random.normal(0, 0.02, N)


df = pd.DataFrame({'height': height, 
                   'leg_left': leg_left, 
                   'leg_right': leg_right})
df.head()


# In[6]:


plt.figure(figsize=(17, 6))

plt.scatter(leg_right, leg_left, s=5, alpha=0.3)

plt.title('Dados dos legs')
plt.xlabel('Leg Right')
plt.ylabel('Leg Left')

plt.grid(ls='--', color='white', alpha=0.4)

plt.show()


# In[7]:


plt.figure(figsize=(17, 6))
plt.hist(df.height, rwidth=0.9, density=True)
plt.grid(ls='--', color='white', alpha=0.4)
plt.show()

plt.figure(figsize=(17, 6))
plt.hist(df.leg_left, rwidth=0.9, density=True, alpha=0.5)
plt.hist(df.leg_right, rwidth=0.9, density=True, alpha=0.5)

plt.grid(ls='--', color='white', alpha=0.4)
plt.show()


# In[8]:


model = """
    data {
        int<lower=0> N;
        vector[N] height;
        vector[N] leg_left;
        vector[N] leg_right;
    }
    
    parameters {
        real alpha;
        real beta_left;
        real beta_right;
        real<lower=0> sigma; 
    }
    
    model {
        alpha ~ normal(10, 100);
        beta_left ~ normal(2, 10);
        beta_right ~ normal(2, 10);
        sigma ~ exponential(1);
        
        height ~ normal(alpha + beta_left * leg_left + beta_right * leg_right, sigma);
    }
"""

data = {
    'N': N,
    'height': height,
    'leg_left': leg_left,
    'leg_right': leg_right
}

posteriori = stan.build(model, data=data)
samples = posteriori.sample(num_chains=4, num_samples=1000)

alpha = samples['alpha'].flatten()
beta_left = samples['beta_left'].flatten()
beta_right = samples['beta_right'].flatten()
sigma = samples['sigma'].flatten()


# ### RCode 6.4 - pag164

# In[9]:


Vide.summary(samples)


# In[10]:


Vide.plot_forest(samples, title='Leg right & Leg left')


# ### RCode 6.5 - pag164

# In[11]:


plt.figure(figsize=(17, 6))

plt.scatter(beta_right, beta_left, s=5, alpha=0.3)

plt.title('Posteriori Beta legs')
plt.xlabel('Beta Right')
plt.ylabel('Beta Left')

plt.grid(ls='--', color='white', alpha=0.4)

plt.show()


# ### RCode 6.6 - pag 165

# In[12]:


plt.figure(figsize=(17, 6))

plt.hist((beta_left + beta_right), density=True, alpha=0.8, bins=100, rwidth=0.9)

plt.title('Posteriori')
plt.xlabel('Soma de Beta_left e beta_right')
plt.ylabel('Densidade')

plt.grid(ls='--', color='white', alpha=0.4)

plt.show()


# In[13]:


# Comparação com os beta indivíduais
plt.figure(figsize=(17, 6))

plt.hist(beta_left, density=True, alpha=0.8, bins=100, rwidth=0.9)  # Beta left

plt.hist(beta_right, density=True, alpha=0.8, bins=100, rwidth=0.9)  # Beta Right

plt.title('Comparativo entre as Posterioris de Beta Left e Beta Right')
plt.xlabel('Betas')
plt.ylabel('Densidade')

plt.grid(ls='--', color='white', alpha=0.1)

plt.show()


# ### RCode 6.7 - Pag 166

# In[14]:


model = """
    data {
        int N;
        vector[N] leg_left;
        vector[N] height;
    }
    
    parameters {
        real alpha;
        real beta_left;
        real sigma;
    }
    
    model {
        alpha ~ normal(10, 100);
        beta_left ~ normal(2, 10);
        sigma ~ exponential(1);
        
        height ~ normal(alpha + beta_left * leg_left, sigma);
    }
"""

data = {
    'N': len(height),
    'leg_left': leg_left,
    'height': height,
}

posteriori = stan.build(model, data=data)
samples = posteriori.sample(num_chains=4, num_samples=1000)

alpha = samples['alpha'].flatten()
beta_left = samples['beta_left'].flatten()
sigma = samples['sigma'].flatten()


# In[15]:


# RCode 6.7 - Continuação
Vide.summary(samples)


# In[16]:


Vide.plot_forest(samples, title='Leg Left')


# ### R Code 6.8

# In[17]:


df = pd.read_csv('data/milk.csv', sep=';')

df_std = df[['kcal.per.g', 'perc.fat', 'perc.lactose']].copy()

df_std['kcal.per.g'] = (df_std['kcal.per.g'] - df_std['kcal.per.g'].mean()) / df_std['kcal.per.g'].std()
df_std['perc.fat'] = (df_std['perc.fat'] - df_std['perc.fat'].mean()) / df_std['perc.fat'].std()
df_std['perc.lactose'] = (df_std['perc.lactose'] - df_std['perc.lactose'].mean()) / df_std['perc.lactose'].std()

df_std.head()


# In[18]:


# Não tem nenhum 'missing values'

df_std.isna().sum()


# ### R Code 6.9 - Pag 167

# In[19]:


# kcal.per.g  regredido em perc.fat

model_kf = """
    data {
        int N;
        vector[N] outcome;
        vector[N] predictor;
    }
    
    parameters {
        real alpha;
        real beta;
        real<lower=0> sigma;
    }
    
    model {
        alpha ~ normal(0, 0.2);
        beta ~ normal(0, 0.5);
        sigma ~ exponential(1);
        
        outcome ~ normal(alpha + beta * predictor, sigma);
    }
"""

data_kf = {
    'N': len(df_std['kcal.per.g']),
    'outcome': list(df_std['kcal.per.g'].values),
    'predictor': list(df_std['perc.fat'].values),
}

posteriori_kf = stan.build(model_kf, data=data_kf)
samples_kf = posteriori_kf.sample(num_chains=4, num_samples=1000)


# In[20]:


# kcal.per.g  regredido em  perc.lactose

model_kl = """
    data {
        int N;
        vector[N] outcome;
        vector[N] predictor;
    }
    
    parameters {
        real alpha;
        real beta;
        real<lower=0> sigma;
    }
    
    model {
        alpha ~ normal(0, 0.2);
        beta ~ normal(0, 0.5);
        sigma ~ exponential(1);
        
        outcome ~ normal(alpha + beta * predictor, sigma);
    }
"""

data_kl = {
    'N': len(df_std['kcal.per.g']),
    'outcome': df_std['kcal.per.g'].values,
    'predictor': df_std['perc.lactose'].values,
}

posteriori_kl = stan.build(model_kl, data=data_kl)
samples_kl = posteriori_kl.sample(num_chains=4, num_samples=1000)


# In[21]:


Vide.summary(samples_kf)


# In[22]:


Vide.plot_forest(samples_kf, title='perc.fat')


# In[23]:


Vide.summary(samples_kl)


# In[24]:


Vide.plot_forest(samples_kl, title='perc.lactose')


# ### R Code 6.10  -  pag 167

# In[25]:


model = """
    data {
        int N;
        vector[N] F;  // Fat
        vector[N] L;  // Lactose
        vector[N] K;  // kcal/g
    }
    
    parameters {
        real alpha;
        real bF;
        real bL;
        real sigma;
    }
    
    model {
        alpha ~ normal(0, 0.2);
        bF ~ normal(0, 0.5);
        bL ~ normal(0, 0.5);
        sigma ~ exponential(1);
        
        K ~ normal(alpha + bF*F + bL*L, sigma);
    }
"""

data = {
    'N': len(df_std['kcal.per.g']),
    'F': df_std['perc.fat'].values,
    'L': df_std['perc.lactose'].values,
    'K': df_std['kcal.per.g'].values,
}

posteriori_FL = stan.build(model, data=data)
samples_FL = posteriori_FL.sample(num_chains=4, num_samples=1000)


# In[26]:


Vide.summary(samples_FL)


# In[27]:


Vide.plot_forest(samples_FL, title='Perc.Fat & Perc.Lactose')


# ### R Code 6.11 - Pag 168 - Figure 6.3

# In[28]:


pd.plotting.scatter_matrix(df_std, diagonal='hist', grid=True, figsize=(17, 6))
plt.show()


# ### R Code 6.12 - Overthinking - Rever

# In[29]:


model = """
    data {
        int N;
        vector[N] kcal_per_g;
        vector[N] perc_fat;
        vector[N] new_predictor_X;
    }
    
    parameters {
        real alpha;
        real bF;
        real bX;
        real<lower=0> sigma;
    }
    
    model {
        kcal_per_g ~ normal(alpha + bF * perc_fat + bX * new_predictor_X, sigma);
    }
"""


# In[30]:


def generate_predictor_x(r=0.9):
    N = len(df['perc.fat'].values)
    
    mean = r * df['perc.fat'].values
    sd = np.sqrt((1 - r**2) * np.var(df['perc.fat'].values))
    
    return np.random.normal(mean, sd, N)  # New Predictor X


# In[31]:


def generate_data_dict(r=0.9):
    data = {
        'N': len(df['kcal.per.g']),
        'kcal_per_g': df['kcal.per.g'].values,
        'perc_fat': df['perc.fat'].values,
        'new_predictor_X': generate_predictor_x(r=r),
    }
    return data


# In[32]:


def adjust_model(r=0.9):
    
    parameter_mean_samples  = []
    
    for _ in range(1):  # In book running 100x
        # Runnning the model
        posteriori = stan.build(model, data=generate_data_dict(r=r))
        samples = posteriori.sample(num_chains=4, num_samples=1000)
        
        # Get parameter slope mean
        parameter_mean_samples.append(samples['bF'].flatten().mean())
            
    return parameter_mean_samples


# In[33]:


stddev = []
r_sequence = np.arange(0, 0.99, 0.1)  # In book using 0.01

for r in r_sequence:
    parameter = adjust_model(r=r)
    stddev.append(np.mean(parameter))


# In[34]:


plt.figure(figsize=(17, 6))

plt.plot(r_sequence, stddev)
plt.xlabel("correlation", fontsize=14)
plt.ylabel("stddev", fontsize=14)
plt.show()


# ### R Code 6.13

# In[35]:


np.random.seed(3)

# Quantidade de plantas
N = 100

# Simulação inicial das alturas
h0 = np.random.normal(10, 2, N)

# Atribuindo tratamentos e simulando fungos e tratamentos
treatment = np.repeat([0,1], repeats=int(N/2))
fungus = np.random.binomial(n=1, p=(0.5 - treatment*0.4), size=N)
h1 = h0 + np.random.normal(5 - 3*fungus, 1, N)

# Dataframe
d = pd.DataFrame.from_dict({'h0': h0, 
                            'h1': h1, 
                            'treatment': treatment, 
                            'fungus': fungus})
d.describe().T


# ### R Code 6.14

# In[36]:


sim_p = np.random.lognormal(0, 0.25, int(1e4))
pd.DataFrame(sim_p, columns=['sim_p']).describe().T


# ### R Code 6.15

# Modelo:
# 
# $$ h_{1,i} \sim Normal(\mu_i, \sigma) $$
# 
# $$ \mu_i = h_{0, i} \times p $$
# 
# Prioris: 
# 
# $$ p \sim LogNormal(0, 0.25) $$
# 
# $$ sigma \sim Exponential(1) $$

# In[37]:


model = """
    data {
        int N; 
        vector[N] h1;
        vector[N] h0;
    }
    
    parameters {
        real<lower=0> p;
        real<lower=0> sigma;
        
    }
    
    model {
        vector[N] mu;
        mu = h0 * p;
        
        h1 ~ normal(mu, sigma);
        
        // Prioris
        p ~ lognormal(0, 0.25);
        sigma ~ exponential(1); 
    }
"""


data = {
    'N': N,
    'h1': h1,
    'h0': h0,
}

posteriori = stan.build(model, data=data)
samples = posteriori.sample(num_chains=4, num_samples=1000)


# In[38]:


Vide.summary(samples)


# ### RCode 6.16

# Modelo *post-treatment bias*:
# 
# $$ h_{1, i} \sim Normal(\mu_i, \sigma) $$
# 
# $$ \mu_i = h_{0, i} \times p $$
# 
# $$ p = \alpha + \beta_T T_i + \beta_F F_i $$
# 
# prioris:
# 
# $$ \alpha \sim LogNormal(0, 0.25) $$
# 
# $$ \beta_T \sim Normal(0, 0.5) $$
# 
# $$ \beta_F \sim Normal(0, 0.5) $$
# 
# $$ \sigma \sim Exponential(1) $$

# In[39]:


"""
To mu definition below
----------------------

vector[N] a;
vector[N] b;
vector[N] c; 

These operation:
c = a .* b;

Is the same operation:
for (n in 1:N) {
  c[n] = a[n] * b[n];
}

Reference:
https://mc-stan.org/docs/reference-manual/arithmetic-expressions.html
"""

model = """
    data {
        int N;
        vector[N] h0;
        vector[N] h1;
        vector[N] T;  // Treatment
        vector[N] F;  // Fungus
    }

    parameters {
        real alpha;
        real bT;
        real bF;
        real<lower=0> sigma;
    }

    model {
        vector[N] mu;
        vector[N] p;
        
        p = alpha + bT * T + bF * F;
        mu = h0 .* p;  
    
        // likelihood
        h1 ~ normal(mu, sigma);
    
        // prioris
        alpha ~ lognormal(0, 0.25);
        bT ~ normal(0, 0.5);
        bF ~ normal(0, 0.5);
        sigma ~ exponential(1);
    }
"""

data = {
    'N': N,
    'h0': h0,
    'h1': h1,
    'T': treatment,
    'F': fungus,
}

posteriori = stan.build(model, data=data)
samples = posteriori.sample(num_chains=4, num_samples=1000)


# In[40]:


Vide.summary(samples)


# In[41]:


Vide.plot_forest(samples, title='Treatment and Fungus')


# ### R Code 6.17

# In[42]:


model = """
    data {
        int N;
        vector[N] h0;
        vector[N] h1;
        vector[N] T;  // Treatment
    }

    parameters {
        real<lower=0> alpha;
        real bT;
        real<lower=0> sigma;
    }
    
    model {
        vector[N] mu;
        vector[N] p;

        p = alpha + bT * T;
        mu = h0 .* p;

        h1 ~ normal(mu, sigma);

        alpha ~ lognormal(0, 0.2);
        bT ~ normal(0, 0.5);
        sigma ~ exponential(1);
    }
"""

data = {
    'N': N,
    'h0': h0,
    'h1': h1,
    'T': treatment,
}

posteriori = stan.build(model, data=data)
samples = posteriori.sample(num_chains=4, num_samples=1000)


# In[43]:


Vide.summary(samples)


# In[44]:


Vide.plot_forest(samples, title='Treatment without fungus')


# ### R Code 6.18

# In[45]:


G = nx.DiGraph()

nodes = {0: '$H_0$', 
         1: '$H_1$', 
         2: 'F', 
         3: 'T'}

for i in nodes:
    G.add_node(nodes[i])

edges = [(nodes[0], nodes[1]),
         (nodes[2], nodes[1]),
         (nodes[3], nodes[2])]

G.add_edges_from(edges)

# explicitly set positions
pos = {nodes[0]: (0, 0), 
       nodes[1]: (1, 0), 
       nodes[2]: (1.5, 0), 
       nodes[3]: (2, 0)}

options = {
    "font_size": 15,
    "node_size": 400,
    "node_color": "white",
    "edgecolors": "white",
    "linewidths": 1,
    "width": 1,
}

nx.draw(G, pos, with_labels=True, **options)

# Set margins for the axes so that nodes aren't clipped
ax = plt.gca()
# ax.margins(0.01)
plt.axis("off")
plt.show()


# ### R Code 6.19

# $$F \_||\_ H_0$$
# 
# $$H_0 \_||\_ T$$
# 
# $$ H_1 \_||\_ T | F $$
# 

# ### R Code 6.20

# In[46]:


# np.random.seed(3)

# Quantidade de plantas
N = 100

# Simulação inicial das alturas
h0 = np.random.normal(10, 2, N)

# Atribuindo tratamentos e simulando fungos e tratamentos
treatment = np.repeat([0, 1], repeats=int(N/2))

M = np.random.binomial(n=1, p=0.5, size=N)  # Moisture -> Bernoulli(p=0.5)

fungus = np.random.binomial(n=1, p=(0.5 - treatment * 0.4 + 0.4 * M), size=N)
h1 = h0 + np.random.normal((5 + 3 * M), 1, N)

# Dataframe
d2 = pd.DataFrame.from_dict({'h0': h0, 
                            'h1': h1, 
                            'treatment': treatment, 
                            'fungus': fungus})
d2.describe().T 


# In[47]:


# RCode 6.17 with new database

model = """
    data {
        int N;
        vector[N] h0;
        vector[N] h1;
        vector[N] T;  // Treatment
    }

    parameters {
        real<lower=0> alpha;
        real bT;
        real<lower=0> sigma;
    }
    
    model {
        vector[N] mu;
        vector[N] p;

        p = alpha + bT * T;
        mu = h0 .* p;

        h1 ~ normal(mu, sigma);

        alpha ~ lognormal(0, 0.2);
        bT ~ normal(0, 0.5);
        sigma ~ exponential(1);
    }
"""

data = {
    'N': N,
    'h0': d2.h0.values,
    'h1': d2.h1.values,
    'T': d2.treatment.values,
}

posteriori = stan.build(model, data=data)
samples = posteriori.sample(num_chains=4, num_samples=1000)


# In[48]:


Vide.summary(samples)


# In[49]:


Vide.plot_forest(samples, title="Only Treatment using d2")


# In[50]:


# RCode 6.16 with new database

model = """
    data {
        int N;
        vector[N] h0;
        vector[N] h1;
        vector[N] T;  // Treatment
        vector[N] F;  // Fungus
    }

    parameters {
        real alpha;
        real bT;
        real bF;
        real<lower=0> sigma;
    }

    model {
        vector[N] mu;
        vector[N] p;
        
        p = alpha + bT * T + bF * F;
        mu = h0 .* p;  
    
        // likelihood
        h1 ~ normal(mu, sigma);
    
        // prioris
        alpha ~ lognormal(0, 0.25);
        bT ~ normal(0, 0.5);
        bF ~ normal(0, 0.5);
        sigma ~ exponential(1);
    }
"""

data = {
    'N': N,
    'h0': d2.h0.values,
    'h1': d2.h1.values,
    'T': d2.treatment.values,
    'F': d2.fungus.values,
}

posteriori = stan.build(model, data=data)
samples = posteriori.sample(num_chains=4, num_samples=1000)


# In[51]:


Vide.summary(samples)


# In[52]:


Vide.plot_forest(samples, title="Treatments and Fungus")


# ## Collider Bias

# ### 6.21

# **Simulação**
# 
# 1. Cada ano, $20$ pessoas nascem com valores de felicidade uniformente distribuídos
# 
# 
# 2. Cada ano, cada uma das pessoas envelhece $1$ ano. A sua felicidade não muda.
# 
# 
# 3. Aos $18$ anos, um indivíduo de casa com a probabilidade de porporcional a sua felicidade.
# 
# 
# 4. Uma vez casado, o indivíduo se mantém casado.
# 
# 
# 5. Aos 65 anos, o indivíduo deixa a amostra (Vai morar na Espanha)

# In[53]:


# Function based in https://github.com/rmcelreath/rethinking/blob/master/R/sim_happiness.R
# Inv_logit R-function in https://stat.ethz.ch/R-manual/R-devel/library/boot/html/inv.logit.html

def inv_logit(x):
    return np.exp(x) / (1 + np.exp(x))

def sim_happiness(seed=1977 , N_years=1000 , max_age=65 , N_births=20 , aom=18):
    np.random.seed(seed)
    
    df = pd.DataFrame(columns=['age', 'married', 'happiness'])

    for i in range(N_years):
        # Update age
        df['age'] += 1

        # Move to Spain when age == max_age
        df.drop(df[df['age'] == max_age].index, inplace=True)
        
        # Will marry?
        index_unmarried_aom = df.query((f'age>={aom} and married==0')).index.tolist()
        weddings = np.random.binomial(1, inv_logit(df.loc[index_unmarried_aom, 'happiness'] - 4))
        df.loc[index_unmarried_aom, 'married'] = weddings
        
        # New borns
        df_aux = pd.DataFrame(columns=['age', 'married', 'happiness'])

        df_aux.loc[:, 'age'] = np.zeros(N_births).astype(int)
        df_aux.loc[:, 'married'] = np.zeros(N_births).astype(int)
        df_aux.loc[:, 'happiness'] = np.linspace(-2, 2, N_births)  # np.random.uniform(0, 1, N_births)
        
        df = df.append(df_aux, ignore_index=True)
        
    return df


# In[54]:


df = sim_happiness(seed=1997, N_years=1000)
df.describe(percentiles=[0.055, 0.945], include='all').T


# In[55]:


# Figure 6.21
plt.figure(figsize=(17, 6))

colors = ['white' if is_married == 0 else 'blue' for is_married in df.married ]

plt.scatter(df.age, df.happiness, color=colors)

plt.title('White - Unmarried   |    Blue - Married')
plt.xlabel('Age')
plt.ylabel('Happiness')
plt.show()


# ### R Code 6.22

# In[56]:


df2 = df[df.age > 17].copy()  # Only adults
df2.loc[:, 'age'] = (df2.age - 18) / (65 - 18)


# ### R Code 6.23

# Modelo:
# 
# $$ happiness \sim Normal(\mu_i, \sigma) $$
# 
# $$ \mu_i = \alpha_{_{MID}[i]} + \beta_A \times A_i $$
# 
# prioris:
# 
# $$ \alpha_{_{MID}[i]} \sim Normal(0, 1)$$
# 
# $$ \beta_A \sim Normal(0, 2) $$
# 
# $$ \sigma \sim Exponential(1); $$

# In[57]:


model = """
    data {
        int N;
        vector[N] age;
        vector[N] happiness;
        array[N] int married;  // Must be integer because this is index to alpha. 
    }
    
    parameters {
        vector[2] alpha;  // can also be written like this: real alpha[2] or array[2] int alpha;
        real beta_age;
        real<lower=0> sigma;
    }
    
    model {
        vector[N] mu;
        
        for (i in 1:N){
            mu[i] = alpha[ married[i] ] + beta_age * age[i];
        }
        
        happiness ~ normal(mu, sigma);
        
        // Prioris
        alpha ~ normal(0, 1);
        beta_age ~normal(0, 2);
        sigma ~ exponential(1);
    }
"""

data = {
    'N': len(df2.happiness.values),
    'age': df2.age.values,
    'happiness': df2.happiness.values,
    'married': df2.married.values + 1  # Because the index in stan starting with 1 
}

posteriori = stan.build(model, data=data)
samples = posteriori.sample(num_chains=4, num_samples=1000)


# In[58]:


Vide.summary(samples)


# In[59]:


Vide.plot_forest(samples, title='Married and Unmarried')


# ### R Code 6.24

# In[60]:


model = """
    data {
        int N;
        vector[N] age;
        vector[N] happiness;
    }
    
    parameters {
        real alpha;
        real beta_age;
        real<lower=0> sigma;
    }
    
    model {
        vector[N] mu;
        
        for (i in 1:N){
            mu[i] = alpha + beta_age * age[i];
        }
        
        happiness ~ normal(mu, sigma);
        
        alpha ~ normal(0, 1);
        beta_age ~ normal(0, 2);
        sigma ~ exponential(1);
    }
"""

data = {
    'N': len(df2.happiness.values),
    'happiness': df2.happiness.values,
    'age': df2.age.values,
}

posteriori = stan.build(model, data=data)
samples = posteriori.sample(num_chains=4, num_samples=1000)


# In[61]:


Vide.summary(samples)


# In[62]:


Vide.plot_forest(samples, title='Without married variable')


# ### RCode 6.25

# In[63]:


N = 200  # Qty of triads  (G, P, C)

b_GP = 1  # Direct effect of G on P 
b_GC = 0  # Direct effect of G on C
b_PC = 1  # Direct effect of P on C
b_U  = 2  # Direct effect of U on  P and C


# ### R Code 6.26

# In[64]:


np.random.seed(3)

U = 2 * np.random.binomial(n=1, p=0.5, size=N) - 1  # {-1, 1}
# U = np.random.normal(0, 1, N)  # Simulation more realistic example 

G = np.random.normal(0, 1, size=N)  # Has not influence

P = np.random.normal(b_GP*G + b_U*U, 1, size=N)

C = np.random.normal(b_PC*P + b_GC*G + b_U*U, 1, size=N)

d = pd.DataFrame.from_dict({'C':C, 'P':P, 'G':G, 'U':U})

d.head()


# ### R Code 6.27

# In[65]:


# C ~ P + G

model = """
    data {
        int N;
        vector[N] C;
        vector[N] P;
        vector[N] G;
    }
    
    parameters {
        real alpha;
        real b_PC;
        real b_GC;
        real<lower=0> sigma;
    }
    
    model {
        vector[N] mu;
        
        for (i in 1:N){
            mu[i] = alpha + b_PC*P[i] + b_GC*G[i];
        }
        
        C ~ normal(mu, sigma);
        
        alpha ~ normal(0, 1);
        b_PC ~ normal(0, 1);
        b_GC ~ normal(0, 1);
        sigma ~ exponential(1);
    }
"""

data = {
    'N': len(d.C.values),
    'C': d.C.values,
    'P': d.P.values,
    'G': d.G.values,
}

posteriori = stan.build(model, data=data)
samples = posteriori.sample(num_chains=4, num_samples=1000)


# In[66]:


Vide.summary(samples)


# In[67]:


Vide.plot_forest(samples)


# In[68]:


# Figure 6.5
plt.figure(figsize=(17, 6))

colors = ['black' if u <= 0 else 'blue' for u in U]  # Unobserved

x = np.linspace(-3, 4)
y = np.mean(samples['alpha']) + np.mean(samples['b_GC']) * x

plt.plot(x, y, c='k')

plt.scatter(G, C, c=colors)
plt.xlabel('Granparent Education (G)')
plt.ylabel('GranChild Education (C)')
plt.title('Educations - Bias Collider')
plt.show()


# ### R Code 6.28

# In[69]:


model = """
    data {
        int N;
        vector[N] C;
        vector[N] P;
        vector[N] G;
        vector[N] U;
    }
    
    parameters {
        real alpha;
        real b_PC;
        real b_GC;
        real b_U;
        real<lower=0> sigma;
    }
    
    model {
        vector[N] mu;
        
        for (i in 1:N){
            mu[i] = alpha + b_PC*P[i] + b_GC*G[i] + b_U*U[i];
        }
        
        C ~ normal(mu, sigma);
        
        alpha ~ normal(0, 1);
        b_GC ~ normal(0, 1);
        b_PC ~ normal(0, 1);
        b_U  ~ normal(0, 1);
        sigma ~ exponential(1);
    }
"""

data = {
    'N': len(d.C.values),
    'C': d.C.values,
    'P': d.P.values,
    'G': d.G.values,
    'U': d.U.values,
}

posteriori = stan.build(model, data=data)
samples = posteriori.sample(num_chains=4, num_samples=1000)


# In[70]:


Vide.summary(samples)


# In[71]:


Vide.plot_forest(samples)


# ### R Code 6.29
# 
# Reference: [*ksachdeva*](https://colab.research.google.com/github/ksachdeva/rethinking-tensorflow-probability/blob/master/notebooks/06_the_haunted_dag_and_the_causal_terror.ipynb#scrollTo=wyUj47f7-kLy)

# In[72]:


dag_6_1 = CausalGraphicalModel(
    nodes=["C", "U", "B", "A", "X", "Y"],
    edges=[
        ("U", "X"),
        ("A", "U"),
        ("A", "C"),
        ("C", "Y"),
        ("U", "B"),
        ("C", "B"),
        ("X", "Y"),
    ],
)

pgm = daft.PGM()
coordinates = {
    "U": (0, 2),
    "C": (4, 2),
    "A": (2, 3),
    "B": (2, 1),
    "X": (0, 0),
    "Y": (4, 0),
}
for node in dag_6_1.dag.nodes:
    pgm.add_node(node, node, *coordinates[node])
for edge in dag_6_1.dag.edges:
    pgm.add_edge(*edge)
pgm.render()

plt.show()


# In[73]:


all_adjustment_sets = dag_6_1.get_all_backdoor_adjustment_sets("X", "Y")

for s in all_adjustment_sets:
    if all(not t.issubset(s) for t in all_adjustment_sets if t != s):
        if s != {"U"}:
            print(s)


# ### R Code 6.30

# In[74]:


dag_6_2 = CausalGraphicalModel(
    nodes=['S', 'A', 'M', 'W', 'D'],
    edges=[
        ('S','W'),
        ('S','M'),
        ('S','A'),
        ('A','M'),
        ('A','D'),
        ('M','D'),
        ('W','D'),
    ],
)

# Drawing the DAG
pgm = daft.PGM()
coordinates = {
    "S": (0, 2),
    "A": (0, 0),
    "M": (1, 1),
    "W": (2, 2),
    "D": (2, 0),
}
for node in dag_6_2.dag.nodes:
    pgm.add_node(node, node, *coordinates[node])
for edge in dag_6_2.dag.edges:
    pgm.add_edge(*edge)
pgm.render()

plt.show()


# In[75]:


# R Code 6.30

all_adjustment_sets = dag_6_2.get_all_backdoor_adjustment_sets("W", "D")


for s in all_adjustment_sets:
    if all(not t.issubset(s) for t in all_adjustment_sets if t != s):
        print(s)


# ### R Code 6.31
# 
# Reference: [*Fehiepsi - Numpyro*](https://github.com/fehiepsi/rethinking-numpyro/blob/master/notebooks/06_the_haunted_dag_and_the_causal_terror.ipynb)

# In[76]:


all_independencies = dag_6_2.get_all_independence_relationships()

for s in all_independencies:
    if all(
        t[0] != s[0] or t[1] != s[1] or not t[2].issubset(s[2])
        for t in all_independencies
        if t != s
    ):
        print(s)

