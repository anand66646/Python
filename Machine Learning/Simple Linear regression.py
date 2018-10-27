
# coding: utf-8

# Simple linear Regression

# Import all relavent libraries for the model

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Generate random input data to train on
# 

# In[ ]:


Observations=1000
xs = np.random.uniform(-10,10,(Observations,1))
xz = np.random.uniform(-10,10,(Observations,1))
inputs = np.column_stack((xs,xz))


# In[ ]:


print(inputs.shape)


# Create the targets that we aim at

# In[21]:


noise = np.random.uniform(-1,1,(Observations,1))
targets = 2*xs - 3*xz +5 +noise
print(targets.shape)


# Plot the training data

# In[32]:


targets = targets.reshape(Observations,)


# In[33]:


fig = plt.figure()
ax= fig.add_subplot(111,projection='3d')
ax.plot(xs,xz,targets)
ax.set_xlabel('xs')
ax.set_ylabel('ys')
ax.set_zlabel('Targets')
ax.view_init(azim=100)
plt.show()


# In[38]:


init_range=0.1
weights = np.random.uniform(-init_range, init_range,(2,1))
biases = np.random.uniform(-init_range, init_range,1)
print(weights)
print(biases)


# In[39]:


targets = targets.reshape(Observations,1)


# In[41]:


### set a learning rate

learning_rate =0.2

### Train the model

for i in range(100):
    outputs=np.dot(inputs,weights)+biases
    deltas = outputs-targets
    loss=np.sum(deltas**2)/2/Observations
    print(loss)
    deltas_scaled = deltas/Observations
    weights = weights - learning_rate*np.dot(inputs.T, deltas_scaled)
    biases = biases - learning_rate*np.sum(deltas_scaled)
    
print(weights, biases)


# In[42]:


#### plot the final output
plt.plot(outputs, targets)
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()

