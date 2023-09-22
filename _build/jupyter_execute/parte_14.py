#!/usr/bin/env python
# coding: utf-8

# # 14 - Aventuras na Covariâncias

# ## Imports, loadings and functions

# In[1]:


import numpy as np

from scipy import stats

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

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
from causalgraphicalmodels import CausalGraphicalModel  # Just work in < python3.9 


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


# In[5]:


# https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
def get_correlated_dataset(n, dependency, mu, scale):
    latent = np.random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]


# In[6]:


# https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    #scale_x = n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    #scale_y = n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D()         .rotate_deg(45)         .scale(scale_x, scale_y)         .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# In[7]:


def draw_ellipse(mu, sigma, level, plot_points=None):
    
    fig, ax_nstd = plt.subplots(figsize=(17, 8))

    dependency_nstd = sigma.tolist()

    scale = 5, 5

    x, y = get_correlated_dataset(500, dependency_nstd, mu, scale)
    
    if plot_points is not None:
        ax_nstd.scatter(plot_points[:, 0], plot_points[:, 1], s=25)

    for l in level:
        confidence_ellipse(x, y, ax_nstd, n_std=l, edgecolor='black', alpha=0.3)

    # ax_nstd.scatter(mu[0], mu[1], c='red', s=3)
    ax_nstd.set_title('Cafés sampled - Figure 14.2')
    ax_nstd.set_xlabel('intercepts (a_cafe)')
    ax_nstd.set_ylabel('slopes (b_cafe)')
    
    ax_nstd.set_xlim(1.8, 6.2)
    ax_nstd.set_ylim(-1.9, -0.51)
    
    plt.show()


# ## 14.1 Varing Slopes by Contructions

# ### R Code 14.1

# In[8]:


a = 3.5        # Average morning wait time
b = (-1)       # Average difference afternoon wait time
sigma_a = 1    # std dev in intercepts
sigma_b = 0.5  # std in slopes
rho = (-0.7)   # correlation between intercepts and slopes


# ### R Code 14.2

# In[9]:


Mu = np.array([a, b])  # Vector
Mu


#  ### R Code 14.3

# The matrix variance and covariance:
# 
# $$ 
# \begin{pmatrix}
# \mbox{variance of intercepts} & \mbox{covariance of intercepts & Slopes} \\
# \mbox{covariance of intercepts & Slopes} & \mbox{variance of slopes}
# \end{pmatrix}
# $$
# 
# And in math form:
# 
# 
# $$
# \begin{pmatrix}
# \sigma^2_\alpha & \sigma_\alpha\sigma_\beta\rho \\
# \sigma_\alpha\sigma_\beta\rho & \sigma^2_\beta
# \end{pmatrix}
# $$

# ### R Code 14.3

# In[10]:


cov_ab = sigma_a * sigma_b * rho
Sigma = np.matrix([
    [sigma_a**2, cov_ab],
    [cov_ab, sigma_b**2]
])
Sigma


# ### R Code 14.4

# In[11]:


np.matrix([[1,2], [3, 4]])


# ### R Code 14.5

# In[12]:


sigmas = np.array([sigma_a, sigma_b])  # Standart Deviations

sigmas


# In[13]:


Rho = np.matrix([
    [1, rho],
    [rho, 1]
])  # Correlation Matrix

Rho


# In[14]:


# Multiply to get covariance matrix
Sigma = np.diag(sigmas) * Rho * np.diag(sigmas)

Sigma


# ### R Code 14.6

# In[15]:


N_cafe = 20


# ### R Code 14.7

# In[16]:


np.random.seed(5)   # used to replicate example

vary_effects = stats.multivariate_normal(mean=Mu, cov=Sigma).rvs(N_cafe)
vary_effects


# ### R Code 14.8

# In[17]:


a_cafe = vary_effects[:, 0]  # Intercepts samples
b_cafe = vary_effects[:, 1]  # Slopes samples


# ### R Code 14.9

# In[18]:


levels = [0.1, 0.3, 0.5, 0.8, 0.99]
draw_ellipse(mu=Mu, sigma=Sigma, level=levels, plot_points=vary_effects)


# **Overthinking**: We define the *covariance* with 3 parameters: (pag 440)
# 
#  1. Firt variable standart deviations  $\sigma_\alpha$
#  2. Second variable standart deviations  $ \sigma_\beta $
#  3. Correlation between two variables  $\rho_{\alpha\beta}$
#     
# Whu the covariance is equal to $\sigma_\alpha \sigma_\beta \rho_{\alpha\beta}$?
# 
# The usual definition of the covariance between two variables $x$ and $y$ is:
# 
# $$ cov(x, y) = \mathbb{E}(xy) - \mathbb{E}(x)\mathbb{E}(y) $$
# 
# 
# To simple variance this concept is the same:
# 
# $$ var(x) = cov(x, x) = \mathbb{E}(x^2) - \mathbb{E}(x)^2$$
# 
# 
# If the variable with only mean equal zero ($0$), then the variance to be:
# 
# $$ var(x) = cov(x) = \mathbb{E}(x^2), \mbox{ if the } \mathbb{E}(x) = 0 $$
# 
# 
# The correlation live in $[-1, 1]$, this is only the covariance rescaled. To be rescaled covariance just:
# 
# $$ cov(xy) = \sqrt{var(x)var(y)} \rho_{xy} $$
# 
# or better:
# 
# $$ cov(xy) = \sigma_x \sigma_y \rho_{xy} $$
# 

# ### R Code 14.10

# In[19]:


N_visit = 10

afternoon = np.tile([0, 1], reps=int((N_visit*N_cafe)/2))
cafe_id = np.repeat(np.arange(0, N_cafe), N_visit)

mu = a_cafe[cafe_id] + b_cafe[cafe_id]*afternoon
sigma = 0.5  # std dev within cafes

wait = np.random.normal(mu, sigma, size=N_visit*N_cafe)
cafe_id += 1  # index from 1 to 20 cafe_id, need to Stan index

d = pd.DataFrame({'cafe': cafe_id, 'afternoon':afternoon, 'wait': wait})
d


# ## 14.1.3 The varing slope model

# The likelihood:
# 
# $$ W_i \sim Normal(\mu_i, \sigma) $$
# 
# and the linear model:
# 
# $$ \mu_i = \alpha_{CAFÉ[i]} + \beta_{CAFÉ[i]}A_i $$
# 
# 
# Then comes the matrix of varing intercepts and slopes, with it's covariance matrix:
# 
# Population of varying effects:
# $$
# \begin{bmatrix}
# \alpha_{CAFÉ} \\
# \beta_{CAFÉ}
# \end{bmatrix} 
# \sim Normal
# \begin{pmatrix}
# \begin{bmatrix}
# \alpha \\
# \beta
# \end{bmatrix}, S
# \end{pmatrix}
# $$
# 
# construct covariance matrix:
# 
# $$S = 
# \begin{pmatrix} 
# \sigma_\alpha &  0 \\
# 0  & \sigma_\beta
# \end{pmatrix}
# R
# \begin{pmatrix} 
# \sigma_\alpha &  0 \\
# 0  & \sigma_\beta
# \end{pmatrix}
# $$
# 
# The $R$ is the correlation matrix. In this simple case $R$ is defined like:
# 
# $$
# R = 
# \begin{pmatrix}
# 1 & \rho  \\
# \rho & 1
# \end{pmatrix}
# $$
# 
# 
# And the priors and hyper-priors:
# 
# - prior for intercept:
# 
# $$ \alpha \sim Normal(5, 2) $$
# 
# 
# - prior for average slope:
# 
# $$ \beta \sim Normal(-1, 0.5) $$
# 
# 
# 
# - prior for stddev within cafés
# 
# $$\sigma \sim Exponential(1) $$
# 
# 
# - prior for stddev among cafés
# 
# $$ \sigma_\alpha \sim Exponential(1) $$
# 
# 
# - prior for stddev among slopes
# 
# $$ \sigma_\beta \sim Exponential(1) $$
# 
# - prior for correlation  matrix
# 
# $$ R \sim LKJcorr(2) $$

# ### R Code 14.11

# In[20]:


# Dont have the function LKJcorr implemented in numpy or scipy
# Stan implementation - https://mc-stan.org/docs/functions-reference/lkj-correlation.html


# ### R Code 14.12

# In[21]:


# https://mc-stan.org/docs/stan-users-guide/multivariate-hierarchical-priors.html

model = """
    data {
        int N;
        int N_cafe;
        int N_periods;
        
        array[N] real wait;
        array[N] int cafe;
        array[N] int afternoon; 
    }
    
    parameters {
        real<lower=0> sigma;
        vector<lower=0>[2] sigma_cafe;
        
        vector[N_cafe] alpha_cafe;
        vector[N_cafe] beta_cafe;
        
        real alpha;  // Like bar_alpha (hyper-prior)
        real beta;   // Like bar_beta (hyper-prior)
        
        corr_matrix[N_periods] Rho;
    }
    
    model {
        vector[N] mu;
        vector[2] Y_cafe[N_cafe];
        vector[2] MU_cafe;
        
        //(sigma) - Prior to likelihood
        sigma ~ exponential(1);  
        
        
        // (sigma_cafe) - Hyper Prior
        sigma_cafe ~ exponential(1);
        
        // (Rho) - Hyper Prior
        Rho ~ lkj_corr(2);
        
        // (MU_cafe) - Hyper Prior
        alpha ~ normal(5, 2);
        beta ~ normal(-1, 0.5);
  
        MU_cafe = [alpha, beta]';
        
        // (alpha_cafe, beta_cafe) - Prior 
        for (j in 1:N_cafe){
            Y_cafe[j] = [alpha_cafe[j], beta_cafe[j]]' ; 
        }
        
        Y_cafe ~ multi_normal(MU_cafe, quad_form_diag(Rho, sigma_cafe));  // Prior to alpha_cafe and beta_cafe
    
    
        // (mu) - Linear model
        for (i in 1:N){
            mu[i] = alpha_cafe[ cafe[i] ] + beta_cafe[ cafe[i] ] * afternoon[i];
        }
        
        // Likelihood 
        wait ~ normal(mu, sigma);
    }
"""

dat_list = {
    'N': len(d),
    'N_cafe': N_cafe,
    'N_periods': len(d.afternoon.unique()),
    'wait':  d.wait.values,
    'afternoon': d.afternoon.values,
    'cafe': d.cafe.values,
}

posteriori = stan.build(model, data=dat_list)
samples = posteriori.sample(num_chains=4, num_samples=1000)


# In[22]:


model_another_way = """
    data {
        int N;
        int N_cafe;
        int N_periods;
        
        array[N] real wait;
        array[N] int cafe;
        array[N] int afternoon; 
    }
    
    parameters {
        real<lower=0> sigma;
        vector<lower=0>[2] sigma_cafe;
        
        vector[N_cafe] alpha_cafe;
        vector[N_cafe] beta_cafe;
        
        real alpha;  // Like bar_alpha (hyper-prior)
        real beta;   // Like bar_beta (hyper-prior)
        
        corr_matrix[N_periods] Rho;
    }
    
    model {
        vector[N] mu;
        
        //(sigma) - Prior to likelihood
        sigma ~ exponential(1);  
        
        
        // (sigma_cafe) - Hyper Prior
        sigma_cafe ~ exponential(1);
        
        // (Rho) - Hyper Prior
        Rho ~ lkj_corr(2);
        
        // (MU_cafe) - Hyper Prior
        alpha ~ normal(5, 2);
        beta ~ normal(-1, 0.5);

        for (i in 1:N_cafe){
            [alpha_cafe[i], beta_cafe[i]]' ~ multi_normal([alpha, beta]', quad_form_diag(Rho, sigma_cafe));
        }
        
    
        // (mu) - Linear model
        for (i in 1:N){
            mu[i] = alpha_cafe[ cafe[i] ] + beta_cafe[ cafe[i] ] * afternoon[i];
        }
        
        // Likelihood 
        wait ~ normal(mu, sigma);
    }
"""


# In[23]:


m14_1 = az.from_pystan(
    posterior=samples,
    posterior_model=posteriori,
    observed_data=dat_list.keys()
)


# In[24]:


az.summary(m14_1, hdi_prob=0.89)


# In[25]:


az.plot_forest(m14_1, hdi_prob=0.89, combined=True, figsize=(17, 25))
plt.grid(axis='y', color='white', alpha=0.3)
plt.axvline(x=0, ls='--', color='white', alpha=0.4)
plt.show()


# ### R Code 14.13

# In[26]:


az.plot_density(m14_1.posterior.Rho.sel(Rho_dim_0=0, Rho_dim_1=1), figsize=(17, 6))
plt.xlabel = 'Correlation'
plt.ylabel = 'Density'
plt.show()


# ### R Code 14.14 & R Code 14.15

# In[27]:


post_alpha = m14_1.posterior.alpha.mean().values
post_beta = m14_1.posterior.beta.mean().values

post_sigma_alpha = m14_1.posterior.sigma_cafe.sel(sigma_cafe_dim_0=0).mean().values
post_sigma_beta = m14_1.posterior.sigma_cafe.sel(sigma_cafe_dim_0=1).mean().values
post_rho = m14_1.posterior.Rho.sel(Rho_dim_0=0, Rho_dim_1=1).mean().values

print('Posterior alpha:', post_alpha)
print('Posterior beta:', post_beta)
print('Posterior sigma alpha:', post_sigma_alpha)
print('Posterior sigma beta:', post_sigma_beta)
print('Posterior Rho:', post_rho)


# In[28]:


post_matrix_sigma_alpha_beta = np.matrix([[post_sigma_alpha, 0],[0, post_sigma_beta]])
post_matrix_sigma_alpha_beta


# In[29]:


post_matrix_rho = np.matrix([[1, post_rho],[post_rho, 1]])
post_matrix_rho


# In[30]:


post_Sigma = post_matrix_sigma_alpha_beta * post_matrix_rho * post_matrix_sigma_alpha_beta
post_Sigma


# In[31]:


a1 = [d.loc[(d['afternoon'] == 0) & (d['cafe'] == i)]['wait'].mean() for i in range(1, N_cafe+1)]
b1 = [d.loc[(d['afternoon'] == 1) & (d['cafe'] == i)]['wait'].mean() for i in range(1, N_cafe+1)]

# Remember that in R Code 14.1 above, the data is build as
# a = 3.5        # Average morning wait time
# b = (-1)       # Average difference afternoon wait time
# because this, b value is b - a, like this

b1 = [b1[i] - a1[i] for i in range(len(a1))]


# In[32]:


# Extract posterior means of partially polled estimates
a2 = [m14_1.posterior.alpha_cafe.sel(alpha_cafe_dim_0=i).mean().values for i in range(N_cafe)]
b2 = [m14_1.posterior.beta_cafe.sel(beta_cafe_dim_0=i).mean().values for i in range(N_cafe)]


# In[33]:


fig, ax_nstd = plt.subplots(figsize=(17, 8))

mu = np.array([post_alpha, post_beta])
dependency_nstd = post_Sigma.tolist()

scale = 5, 5

level = [0.1, 0.3, 0.5, 0.8, 0.99]

x, y = get_correlated_dataset(500, dependency_nstd, mu, scale)


ax_nstd.scatter(a1, b1, s=25, color='b')
ax_nstd.scatter(a2, b2, s=25, facecolor='gray')

ax_nstd.plot([a1, a2], [b1, b2], 'k-', linewidth=0.5)
    

for l in level:
    confidence_ellipse(x, y, ax_nstd, n_std=l, edgecolor='black', alpha=0.3)

# ax_nstd.scatter(mu[0], mu[1], c='red', s=3)
ax_nstd.set_title('Cafés sampled - Figure 14.5 | Left')
ax_nstd.set_xlabel('intercepts (a_cafe)')
ax_nstd.set_ylabel('slopes (b_cafe)')


ax_nstd.set_xlim(1.8, 6.2)
ax_nstd.set_ylim(-1.9, -0.51)

# ax_nstd.legend()
ax_nstd.legend(['Raw data', 'Partial polling'])

plt.plot(1,2, c='r')

plt.show()


# ### R Code 14.16 & R Code 14.17

# In[34]:


wait_morning_1 = (a1)
wait_afternoon_1 = [sum(x) for x in zip(a1, b1)]
wait_morning_2 = (a2)
wait_afternoon_2 = [sum(x) for x in zip(a2, b2)]


# In[35]:


# now shrinkage distribution by simulation  

v = np.random.multivariate_normal(mu ,post_Sigma, 1000)
v[:, 1] = [sum(x) for x in v]  # calculation afternoon wait
Sigma_est2 = np.cov(v[:, 0], v[:, 1])

mu_est2 = [0, 0]  # Empty vector mu
mu_est2[0] = mu[0] 
mu_est2[1] = mu[0] + mu[1]


# In[36]:


fig, ax_nstd = plt.subplots(figsize=(17, 8))

scale = 5, 5
#level = [0.1, 0.3, 0.5, 0.8, 0.99]
level = [0.1, 0.69, 1.5, 2.0, 3.0]

dependency_nstd = Sigma_est2.tolist()
x, y = get_correlated_dataset(500, dependency_nstd, mu_est2, scale)

ax_nstd.scatter(wait_morning_1, wait_afternoon_1, s=25, color='b')
ax_nstd.scatter(wait_morning_2, wait_afternoon_2, s=25, facecolor='gray')

ax_nstd.plot([wait_morning_1, wait_morning_2],
             [wait_afternoon_1, wait_afternoon_2], 
             'k-', linewidth=0.5)

for l in level:
    confidence_ellipse(x, y, ax_nstd, n_std=l, edgecolor='black', alpha=0.3)

ax_nstd.set_title('Cafés Waiting - Figure 14.5 | Right')
ax_nstd.set_xlabel('morning wait')
ax_nstd.set_ylabel('afternoon wait')

ax_nstd.plot([0, 10], [0, 10], ls='--', color='white', alpha=0.5, linewidth=4)
ax_nstd.text(3.75, 4.1, "x==y",color='white', fontsize=16)

ax_nstd.set_xlim(1.7, 6)
ax_nstd.set_ylim(1, 5)

# ax_nstd.legend()
ax_nstd.legend(['Raw data', 'Partial polling'])

plt.plot(1,2, c='r')

plt.show()


# ## 14.2 Advanced varing Slopes

# ### R Code 14.18

# In[37]:


# Previous chimpanzees models is in chapter 11

df = pd.read_csv('./data/chimpanzees.csv', sep=';')


# In[38]:


# Build the treatment data
df['treatment'] = 1 + df['prosoc_left'] + 2 * df['condition']  
df.head(20)


# In[39]:


df.tail(20)


# $$ L_i \sim  Binomial(1, p_i) $$
# 
# $$ logit(p_i) = \gamma_{_{TID}[i]} + \alpha_{_{ACTOR}[I], _{TID}[I]} + \beta_{_{BLOCK}[I], _{TID}[I]} $$
# 
# $$\gamma_{_{TID}} \sim Normal(0, 1)$$
# 
# $$\begin{bmatrix}
# \alpha_{j, 1} \\
# \alpha_{j, 2} \\
# \alpha_{j, 3} \\
# \alpha_{j, 4} \end{bmatrix} \sim MVNormal
# \begin{pmatrix}
# \begin{bmatrix}
# 0 \\
# 0 \\
# 0 \\
# 0 
# \end{bmatrix} , S_{_{ACTOR}}
# \end{pmatrix}$$
# 
# $$\begin{bmatrix}
# \beta_{j, 1} \\
# \beta_{j, 2} \\
# \beta_{j, 3} \\
# \beta_{j, 4} \end{bmatrix} \sim MVNormal
# \begin{pmatrix}
# \begin{bmatrix}
# 0 \\
# 0 \\
# 0 \\
# 0 
# \end{bmatrix} , S_{_{BLOCK}}
# \end{pmatrix}$$
# 
# Hyper-prior to $S_{_{ACTOR}}$:
# 
# $$\sigma_{_{ACTOR}} \sim Exponential(1)$$
# 
# $$ \rho_{_{ACTOR}} \sim dlkjcorr(4)  $$
# 
# Hyper-prior to $S_{_{BLOCK}}$:
# $$ \sigma_{_{BLOCK}} \sim Exponential(1) $$
# 
# $$ \rho_{_{BLOCK}} \sim dlkjcorr(4)  $$
# 
# 
# This is essentially an interaction model that allow the effect of each **treatment** to vary by **actor** and each **block**. 

# ## Important features to understanding this model
# 
# ### 5.1 Overview of data types
# 
# #### Array types ([stan docs link](https://mc-stan.org/docs/reference-manual/overview-of-data-types.html#array-types))
# 
# ```
# array[10] real x;
# array[6, 7] matrix[3, 3] m;
# array[12, 8, 15] complex z;
# ```
# 
# - declares $x$ to be a **one-dimensional** array of size $10$ containing real values.
# 
# - declares $m$ to be a **two-dimensional** array of size $[6 \times 7]$ containing values that are $[3 \times 3]$ matrices .
# 
# - declares $z$ to be a $[12 \times 8 \times 15]$ array of **complex** numbers.
# 
# 
# 
# And read this two parts below
# 
# 
# ### 5.5 [Vector and matrix data types](https://mc-stan.org/docs/reference-manual/vector-and-matrix-data-types.html)
# 
# Stan provides three types of container objects: arrays, vectors, and matrices.
# 
# 
# ### 5.6 [Array data type](https://mc-stan.org/docs/reference-manual/array-data-types.html)
# Stan supports arrays of arbitrary dimension.
# 
# 

# ### Depreciated Features in Stan to build this example:
# 
# - [13.10 Brackets array syntax](https://mc-stan.org/docs/reference-manual/brackets-array-syntax.html)
# 
# After Stan 2.26 the declarations below:
# 
# ```
# int n[5];
# real a[3, 4];
# real<lower=0> z[5, 4, 2];
# vector[7] mu[3];
# matrix[7, 2] mu[15, 12];
# cholesky_factor_cov[5, 6] mu[2, 3, 4];
# ```
# 
# was replaced to:
# 
# ```
# array[5] int n;
# array[3, 4] real a;
# array[5, 4, 2] real<lower=0> z;
# array[3] vector[7] mu;
# array[15, 12] matrix[7, 2] mu;
# array[2, 3, 4] cholesky_factor_cov[5, 6] mu;
# ```
# 
# - [13.12 Nested multiple indexing in assignments](https://mc-stan.org/docs/reference-manual/nested-multiple-indexing-in-assignments.html)
# 
# This statements
# 
# ```
# a[:][1] = b;
# ```
# 
# was replaced by
# 
# ```
# a[:, 1] = b;
# ```

# #### Explanation:
# 
# Why this `alpha ~ multi_normal(rep_vector(0, 4), quad_form_diag(Rho_actor, sigma_actor));` works in code below?
# 
# #### Response:
# 
# Let be the code:
# ```
# parameters{
#     ...
#    vector[qty_treatments] gamma;
#    ...
# }
# ```
# There are $4$ positions in vector `gamma`, then when we atrribute the `normal(0, 1)`, we realize make the broadcast:
# 
# ```
# gamma ~ normal(0, 1);
# ```
# 
# is equivalent to:
# 
# ```
# for (i in 1:qty_treatments){
#     gamma[i] ~ normal(0, 1);
# }
# ```
# 
# 
# This is the same reason that code with multi_normal works. Let see this in more details:
# 
# ```
# parameters{
#     ...
#    array[qty_chimpanzees] vector[qty_treatments] alpha;
#    ...
# }
# ```
# 
# Now, the `alpha` have two dimensional structure. 
# 
# *Note: index in Stan start in 1, not in zero like python.*
# 
# - The `alpha[1, 1]` is `alpha` to `chimpanzees=1` and to `treatment=1`  
# - The `alpha[1, 2]` is `alpha` to `chimpanzees=1` and to `treatment=2`  
# 
# - The `alpha[2, 3]` is `alpha` to `chimpanzees=2` and to `treatment=3`  
# - The `alpha[7, 4]` is `alpha` to `chimpanzees=7` and to `treatment=4`  
# 
# And
# 
# - The `alpha[1, :]` is `alpha` to `chimpanzees=1` and to all `treatments`  
# 
# 
# The type of data to `alpha[1, :]` is a `vector[qty_treatments]`. 
# 
# This is the same type that `multi_normal` expected to attribuite the results.
# 
# Then, *the code works*!

# In[40]:


model = """
    data {
        int N;
        int qty_chimpanzees;
        int qty_blocks;
        int qty_treatments;
        
        array[N] int pulled_left;
        array[N] int actor;
        array[N] int block;
        array[N] int treatment;
    }
    
    parameters{
        vector[qty_treatments] gamma;
        array[qty_chimpanzees] vector[qty_treatments] alpha;
        array[qty_blocks] vector[qty_treatments] beta;
        
        corr_matrix[qty_treatments] Rho_actor;
        vector<lower=0>[qty_treatments] sigma_actor;
        
        corr_matrix[qty_treatments] Rho_block;
        vector[qty_treatments] sigma_block;
    }
    
    model {
        vector[N] p;
        
        // Fixed Priors
        Rho_actor ~ lkj_corr(4);
        sigma_actor ~ exponential(1);
        
        Rho_block ~ lkj_corr(4);
        sigma_block ~ exponential(1);
        
    
        // Adaptatives Priors
        gamma ~ normal(0, 1);
        
        alpha ~ multi_normal(rep_vector(0, 4), quad_form_diag(Rho_actor, sigma_actor));
        beta ~ multi_normal(rep_vector(0, 4), quad_form_diag(Rho_block, sigma_block));
        
        
        // Linear model
        for (i in 1:N){
            p[i] = gamma[ treatment[i] ] + alpha[ actor[i], treatment[i] ] + beta[ block[i], treatment[i] ];
            p[i] = inv_logit(p[i]);  // Link function
        }
    
    
        // Likelihood
        pulled_left ~ binomial(1, p);
    }
 
"""


dat_list = df[['pulled_left', 'actor', 'block', 'treatment']].to_dict('list')
dat_list['N'] = len(df)
dat_list['qty_chimpanzees'] = len(df['actor'].unique())
dat_list['qty_blocks'] = len(df['block'].unique())
dat_list['qty_treatments'] = len(df['treatment'].unique())

posteriori = stan.build(model, data=dat_list)
samples = posteriori.sample(num_chains=4, num_samples=1000)


# In[41]:


m_14_2 = az.from_pystan(
    posterior=samples,
    posterior_model=posteriori,
    observed_data=dat_list.keys()
)


# In[42]:


print(az.summary(m_14_2).to_string())


# In[43]:


az.plot_forest(m_14_2, hdi_prob=0.89, combined=True, figsize=(17, 30))
plt.grid(axis='y', ls='--', color='white', alpha=0.3)
plt.axvline(x=0, color='red', ls='--', alpha=0.6)
plt.show()


# In[44]:


# Like trankplot in this book

az.rcParams['plot.max_subplots'] = 100

az.plot_ess(
    m_14_2, kind="local", drawstyle="steps-mid", color="k",
    linestyle="-", marker=None
)
plt.show()


# ### R Code 14.19

# In[45]:


# Undertanding Cholesky
rho_test = 0.3
R_test = np.matrix([[1, rho_test], [rho_test, 1]])
R_test


# In[46]:


# Cholesky decomposition
cholesky_matrix_L = np.linalg.cholesky(R_test)
cholesky_matrix_L


# In[47]:


# Testing that R=LL^t
cholesky_matrix_L * np.transpose(cholesky_matrix_L)


# In[48]:


sample_uncorrelated_z_score = np.random.normal(0, 1, [2,200]) # ?
np.corrcoef(sample_uncorrelated_z_score)


# #### How this model is works?
# 
# To understanding the chunck of code below.
# 
# ```
#     transformed parameters{
#         matrix[qty_treatments, qty_chimpanzees] alpha;
#         matrix[qty_treatments, qty_blocks] beta;
#         
#         alpha = (diag_pre_multiply(sigma_actor, L_Rho_actor) * z_actor)';
#         beta = (diag_pre_multiply(sigma_block, L_Rho_block) * z_block)';   
#     }
# ```
# 
# *  PS: $qty\_actor$ *is equal to* $qty\_chimpanzees$ *for this example!* 
# 
# **Firts, thinking about matrices dimensions:**
# 
# To z's ($z_{anything}$)
# - `z_actor`: $[qty\_treatment, qty\_actor]$
# - `z_block`: $[qty\_treatment, qty\_block]$
# 
# 
# To sigmas ($\sigma$):
# 
# - `sigma_actor`: $\sigma_{_{ACTOR}}[qty\_treatments]$
# - `sigma_block`: $\sigma_{_{BLOCK}}[qty\_treatments]$
# 
# 
# To Rho's ($\rho$):
# 
# - `L_Rho_actor`: $\rho_{_{ACTOR}}[qty\_treatments \times qty\_treatments]$
# - `L_Rho_block`: $\rho_{_{BLOCK}}[qty\_treatments \times qty\_treatments]$
# 
# **Now, understanding the stan functions:**
# 
# The Stan function `diag_pre_multiply` [works](https://mc-stan.org/docs/functions-reference/dot-products-and-specialized-products.html):
# 
# - `matrix diag_pre_multiply(vector v, matrix m)`
# 
#     Return the product of the diagonal matrix formed from the vector `v` and the matrix `m`, i.e., `diag_matrix(v) * m`.
#     
# 
# The Stan function `diag_matrix(v)` [works](https://mc-stan.org/docs/functions-reference/diagonal-matrix-functions.html):
# 
# - `matrix diag_matrix(vector x)`
# 
#     The diagonal matrix with diagonal x.
#  
# For example: $x = \begin{bmatrix}
# 0.25 \\
# 0.30 \\
# 0.25 \\
# 0.20
# \end{bmatrix}$ then `diag_matrix(x)` is equal $\begin{bmatrix}
# 0.25 & 0 & 0 & 0 \\
# 0 & 0.30 & 0 & 0 \\
# 0 & 0 & 0.25 & 0 \\
# 0 & 0 & 0 & 0.20
# \end{bmatrix}$, this is a $[4 \times 4]$ matrix.
#  
# **Finally, understanding dimensions for alpha and beta:**
# 
# Then, this `diag_pre_multiply(sigma_actor, L_Rho_actor)` is:
# 
# - `diag_matrix(sigma_actor)` $\times$ `L_Rho_actor`
#     $[qty\_treatments \times qty\_treatments] \times [qty\_treatments \times qty\_treatments] = [qty\_treatments \times qty\_treatments]$
#     
# The complete calculations, have this dimension:
# 
# - `(diag_pre_multiply(sigma_actor, L_Rho_actor) * z_actor)`
# 
#     $[qty\_treatments \times qty\_treatments] \times [qty\_treatments \times qty\_chimpanzees] = 
# [qty\_treatments \times qty\_chimpanzees]$ 
#     
# And to finish, transpose the matrix ( ' ):
# - `(diag_pre_multiply(sigma_actor, L_Rho_actor) * z_actor)'` 
# 
#     $[qty\_chimpanzees \times qty\_treatments]$
#     
#     
# Each of the index in matrix above, is a prior to alpha (or beta) in the model. Make the model with this structure is not need using the `multi_normal()`, like the previous model `m_14_2`.
#     

# In[49]:


model = """
    data {
        int N;
        int qty_chimpanzees;
        int qty_blocks;
        int qty_treatments;
        
        array[N] int pulled_left;
        array[N] int actor;
        array[N] int block;
        array[N] int treatment;
    }
    
    parameters{
        vector[qty_treatments] gamma;
        
        matrix[qty_treatments, qty_chimpanzees] z_actor;
        cholesky_factor_corr[qty_treatments] L_Rho_actor;
        vector<lower=0>[qty_treatments] sigma_actor;
        
        matrix[qty_treatments, qty_blocks] z_block;
        cholesky_factor_corr[qty_treatments] L_Rho_block;
        vector<lower=0>[qty_treatments] sigma_block;
    }
    
    transformed parameters{
        matrix[qty_chimpanzees, qty_treatments] alpha;
        matrix[qty_blocks, qty_treatments] beta;
        
        alpha = (diag_pre_multiply(sigma_actor, L_Rho_actor) * z_actor)';  // matrix[qty_actors, qty_treatments]
        beta = (diag_pre_multiply(sigma_block, L_Rho_block) * z_block)';   // Above explanations about this part
    }
    
    model {
        vector[N] p;
    
        // Adaptative Priors - Non-centered
        gamma ~ normal(0, 1);
    
        to_vector(z_actor) ~ normal(0, 1);
        L_Rho_actor ~ lkj_corr_cholesky(2);
        sigma_actor ~ exponential(1);
        
        to_vector(z_block) ~ normal(0, 1);
        L_Rho_block ~ lkj_corr_cholesky(2);
        sigma_block ~ exponential(1);
        
        // Linear model
        for (i in 1:N){
            p[i] = gamma[ treatment[i] ] + alpha[ actor[i], treatment[i] ] + beta[ block[i], treatment[i] ];
            p[i] = inv_logit(p[i]);  // Link function
        }
    
    
        // Likelihood
        pulled_left ~ binomial(1, p);
    }
 
"""


dat_list = df[['pulled_left', 'actor', 'block', 'treatment']].to_dict('list')
dat_list['N'] = len(df)
dat_list['qty_chimpanzees'] = len(df['actor'].unique())
dat_list['qty_blocks'] = len(df['block'].unique())
dat_list['qty_treatments'] = len(df['treatment'].unique())

posteriori = stan.build(model, data=dat_list)
samples = posteriori.sample(num_chains=4, num_samples=1000)


# In[50]:


m_14_3 = az.from_pystan(
    posterior=samples,
    posterior_model=posteriori,
    observed_data=dat_list.keys()
)


# In[51]:


print(az.summary(m_14_3, hdi_prob=0.89).to_string())


# In[52]:


az.plot_forest(m_14_3, combined=True, hdi_prob=0.89, figsize=(17, 35))

plt.axvline(x=0, color='red', ls='--', alpha=0.7)
plt.grid(axis='y', color='white', ls='--', alpha=0.3)

plt.show()


# ### R Code 14.20

# In[53]:


import matplotlib.pyplot as plt
from importlib import reload
plt=reload(plt)


# In[54]:


alpha_ess_14_3 = az.ess(m_14_3, var_names=["alpha", "beta"]).alpha.values.flatten()
beta_ess_14_3 = az.ess(m_14_3, var_names=["alpha", "beta"]).alpha.values.flatten()

alpha_ess_14_2 = az.ess(m_14_2, var_names=["alpha", "beta"]).alpha.values.flatten()
beta_ess_14_2 = az.ess(m_14_2, var_names=["alpha", "beta"]).beta.values.flatten()

plt.figure(figsize=(17, 8))

plt.scatter(alpha_ess_14_2, alpha_ess_14_3, c='b')
plt.scatter(beta_ess_14_2, beta_ess_14_3[:24], c='b')

plt.plot([0, 5000], [0, 5000], ls='--', c='red')
plt.text(1150, 1250, 'x = y', color='red')

plt.ylim(900, 5000)
plt.xlim(0, 1500)

plt.title('ESS - Effective Simple Size Comparison')
plt.xlabel('Model centered (Default)')
plt.ylabel('Model non-centered (Cholesky)')

plt.show()


# ### R Code 14.21

# In[55]:


az.summary(m_14_3, var_names=['sigma_actor', 'sigma_block'], hdi_prob=0.89)


# ### R Code 14.22

# In[56]:


# Like R Code 11.17
# ==================

TODO = """
plt.figure(figsize=(18, 6))

plt.ylim(0, 1.3)
plt.xlim(0, 7)
plt.axhline(y=1, ls='-', c='black')
plt.axhline(y=0.5, ls='--', c='black')


for i in range(7):
    plt.axvline(x=i+1, c='black')
    plt.text(x=i + 0.25, y=1.1, s=f'Actor {i + 1}', size=20)
    
    alpha_chimp = az.extract(samples_chimpanzees.posterior.alpha.sel(alpha_dim_0=i)).alpha.values
    
    RN = inv_logit(alpha_chimp + az.extract(samples_chimpanzees.posterior.beta.sel(beta_dim_0=0)).beta.values)
    LN = inv_logit(alpha_chimp + az.extract(samples_chimpanzees.posterior.beta.sel(beta_dim_0=1)).beta.values)
    RP = inv_logit(alpha_chimp + az.extract(samples_chimpanzees.posterior.beta.sel(beta_dim_0=2)).beta.values)
    LP = inv_logit(alpha_chimp + az.extract(samples_chimpanzees.posterior.beta.sel(beta_dim_0=3)).beta.values)
    
    # To R/N and R/P
    # ===============
    if not i == 1:
        plt.plot([0.2 + i, 0.8 + i], [RN.mean(), RP.mean()], color='black')
    
    # Plot hdi compatibility interval
    RN_hdi_min, RN_hdi_max = az.hdi(RN, hdi_prob=0.89)
    RP_hdi_min, RP_hdi_max = az.hdi(RP, hdi_prob=0.89)
    plt.plot([0.2 + i, 0.2 + i], [RN_hdi_min, RN_hdi_max], c='black')
    plt.plot([0.8 + i, 0.8 + i], [RP_hdi_min, RP_hdi_max], c='black')
    
    # Plot points
    plt.plot(0.2 + i, RN.mean(), 'o', markerfacecolor='white', color='black')
    plt.plot(0.8 + i, RP.mean(), 'o', markerfacecolor='black', color='white')
    
    # To L/N and L/P
    # ===============
    if not i == 1:
        plt.plot([0.3 + i, 0.9 + i], [LN.mean(), LP.mean()], color='black', linewidth=3)

    # Plot hdi compatibility interval
    LN_hdi_min, LN_hdi_max = az.hdi(LN, hdi_prob=0.89)
    LP_hdi_min, LP_hdi_max = az.hdi(LP, hdi_prob=0.89)
    plt.plot([0.3 + i, 0.3 + i], [LN_hdi_min, LN_hdi_max], c='black')
    plt.plot([0.9 + i, 0.9 + i], [LP_hdi_min, LP_hdi_max], c='black')

    plt.plot(0.3 + i, LN.mean(), 'o', markerfacecolor='white', color='black')
    plt.plot(0.9 + i, LP.mean(), 'o', markerfacecolor='black', color='white')
    
    # Labels for only first points
    if i == 0:
        plt.text(x=0.08, y=RN.mean() - 0.08, s='R/N')
        plt.text(x=0.68, y=RP.mean() - 0.08, s='R/P')
        plt.text(x=0.17, y=LN.mean() + 0.08, s='L/N')
        plt.text(x=0.77, y=LP.mean() + 0.08, s='L/P')
    
plt.title('Posterior Predictions')
plt.ylabel('Proportion left lever')
plt.yticks([0, 0.5, 1], [0, 0.5, 1])
plt.xticks([])

plt.show()
"""

# TODO: Rebuild to this example. 


# ## 14.3 Instrumental and Causal Designs

# The variable $^*U$ is an unobserved. 

# In[57]:


dag_14_3_1 = CausalGraphicalModel(
    nodes=['e', 'U*', 'w'],
    edges=[
        ('U*', 'e'),
        ('U*', 'w'),
        ('e', 'w')
    ],
)

# Drawing the DAG
pgm = daft.PGM()
coordinates = {
    "U*": (2, 1),
    "e": (0, 0),
    "w": (4, 0),
}
for node in dag_14_3_1.dag.nodes:
    pgm.add_node(node, node, *coordinates[node])
for edge in dag_14_3_1.dag.edges:
    pgm.add_edge(*edge)
pgm.render()

plt.show()


# Now, we add *Instrumental Variable* **Q**:

# In[58]:


dag_14_3_1 = CausalGraphicalModel(
    nodes=['e', 'U*', 'w', 'Q'],
    edges=[
        ('U*', 'e'),
        ('U*', 'w'),
        ('e', 'w'),
        ('Q', 'e')
    ],
)

# Drawing the DAG
pgm = daft.PGM()
coordinates = {
    "U*": (3, 1),
    "e": (1, 0),
    "w": (5, 0),
    "Q": (0, 1),
}
for node in dag_14_3_1.dag.nodes:
    pgm.add_node(node, node, *coordinates[node])
for edge in dag_14_3_1.dag.edges:
    pgm.add_edge(*edge)
pgm.render()

plt.show()


# ### R Code 14.23

# In[59]:


def standartize(data, get_parameters=False):
    mean = np.mean(data)
    std = np.std(data)
    
    if not get_parameters:
        return list((data - mean)/std)
    else:
        return list((data - mean)/std), mean, std


# In[60]:


N = 500

U_sim = np.random.normal(0, 1, N)
Q_sim = np.random.choice([1,2,3,4], size=N, replace=True)

E_sim = np.random.normal(U_sim + Q_sim, 1, N)  # E is influencied by U and Q
W_sim = np.random.normal(U_sim + 0*E_sim, 1, N)  # W is influencied only by U, not by E

dat_sim = pd.DataFrame({
    'W': standartize(W_sim),
    'E': standartize(E_sim),
    'Q': standartize(Q_sim)
})

dat_sim


# In[61]:


plt.figure(figsize=(17,5))

plt.plot(dat_sim.E, dat_sim.W, 'o')

plt.title('Wages explained by Education')
plt.xlabel('Education (std)')
plt.ylabel('Wages (std)')

plt.show()


# In[62]:


model = """
    data {
        int N;
        array[N] real W;
        array[N] real E;
    }
    
    parameters {
        real aW;
        real bEW;
        real<lower=0> sigma;
    }
    
    model {
        array[N] real mu;
        
        aW ~ normal(0, 0.2);
        bEW ~ normal(0, 0.5);
        
        sigma ~ exponential(1);
        
        for (i in 1:N){
            mu[i] = aW + bEW*E[i];
        }
        
        W ~ normal(mu, sigma);
    }

"""

dat_list = {
    'N': len(dat_sim),
    'W': dat_sim.W.values,
    'E': dat_sim.E.values,
}

posteriori = stan.build(model, data=dat_list)
samples = posteriori.sample(num_chains=4, num_samples=1000)


# In[63]:


model_14_4 = az.from_pystan(
    posterior=samples,
    posterior_model=posteriori,
    observed_data=dat_list.keys()
)


# In[64]:


az.summary(model_14_4, hdi_prob=0.89)


# ### R Code 14.25

# In[65]:


model = """
    data {
        int N;
        array[N] real W;
        array[N] real E;
        array[N] real Q;
    }
    
    parameters {
        real aW;
        real bEW;
        real bQW;
        real<lower=0> sigma;
    }
    
    model {
        array[N] real mu;
        
        aW ~ normal(0, 0.2);
        bEW ~ normal(0, 0.5);
        bQW ~ normal(0, 0.5);
        
        sigma ~ exponential(1);
        
        for (i in 1:N){
            mu[i] = aW + bEW*E[i] + bQW*Q[i];
        }
        
        W ~ normal(mu, sigma);
    }

"""

dat_list = {
    'N': len(dat_sim),
    'W': dat_sim.W.values,
    'E': dat_sim.E.values,
    'Q': dat_sim.Q.values,
}

posteriori = stan.build(model, data=dat_list)
samples = posteriori.sample(num_chains=4, num_samples=1000)


# In[66]:


model_14_5 = az.from_pystan(
    posterior=samples,
    posterior_model=posteriori,
    observed_data=dat_list.keys()
)


# In[67]:


az.summary(model_14_5, hdi_prob=0.89)


# Include instrument variable $Q$, the bias increase. The `bEW` now is $0.6$ instead of $0.4$ from previous model. The results from the new model is *terrible*! 
# 
# Now, we goes to use *instrument variable* correctly.

# Let's wirter  four models, simples generative versions of the DAG's. 
# 
# 1. The firt model for how wages $W$ are caused by education $E$ and the unobserverd confound $U$.
# 
# $$ W_i \sim Normal(\mu_{w,i}, \sigma_{w})$$
# 
# $$ \mu_{w, i} = \alpha_{w} + \beta_{EW}E_i + U_i $$
# 
# 
# 2. The second model, for education levels $E$ are caused by quarter of birth $Q$ and the same unobserved confound $U$.
# 
# $$ E_i \sim Normal(\mu_{E, i}, \sigma_E) $$
# 
# $$ \mu_{E, i} = \alpha_{E} + \beta_{QE}Q_i + U_i $$
# 
# 3. The thirt model is for $Q$, that says that one-quarter of all people born in each quarter of the year.
# 
# $$ Q_i \sim Categorical([0.25, 0.25, 0.25, 0.25]) $$
# 
# 4. Finally, the model says that the unobserved confound $U$ is normally distributed with mean standart gaussian.
# 
# $$ U_i \sim Normal(0, 1) $$
# 
# Now, we writer this generative model to statistical model, like multivariate normal distribution. Like this:
# 
# $$\begin{pmatrix}
# W_i \\
# E_i
# \end{pmatrix} \sim
#  MVNormal\begin{pmatrix}
#  \begin{bmatrix}
#  \mu_{w, i} \\
#  \mu_{E, i}
#  \end{bmatrix}, S
#  \end{pmatrix}
# $$

# In[ ]:




