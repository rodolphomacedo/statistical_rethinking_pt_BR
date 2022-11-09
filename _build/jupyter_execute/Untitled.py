#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from sklearn.preprocessing import SplineTransformer


# In[2]:


k = 2  # Grau da spline
t = [0, 1, 2, 3, 4, 5, 6]
c = [-1, 2, 0, -1]

spl = BSpline(t, c, k)


# In[3]:


print(dir(spl))


# In[4]:


plt.plot(spl(np.arange(0, 6, 0.1)))
plt.show()


# In[5]:


spl.design_matrix


# In[49]:


X = np.arange(0, 60).reshape(60, 1)
spline = SplineTransformer(degree=4, n_knots=5)
B = spline.fit_transform(X)
plt.plot(B)
plt.show()


# In[47]:


from scipy.interpolate import make_interp_spline, BSpline
import numpy as np
x = np.linspace(0, np.pi * 2, 4)
y = np.sin(x)
k = 3

bspl = make_interp_spline(xb
                          , y, k=k)
design_matrix = bspl.design_matrix(x, bspl.t, k)


# In[48]:


k = 2
t = [-1, 0, 1, 2, 3, 4, 5, 6]
x = [1, 2, 3, 4]
design_matrix = BSpline.design_matrix(x, t, k).toarray()
design_matrix


# In[53]:


a = [1, 2, 3, 4, 5]

np.pad(a, (1, 1), '')

