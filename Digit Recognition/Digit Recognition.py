#!/usr/bin/env python
# coding: utf-8

# In[8]:


import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


# In[10]:


len(X_train)


# In[11]:


len(y_train)


# In[12]:


len(X_test)


# In[14]:


X_train[0].shape


# In[15]:


X_train[0]


# In[16]:


plt.matshow(X_train[0])


# In[18]:


y_train[:5]


# In[26]:


X_train = X_train / 255
X_test = X_test /255


# In[27]:


#flatten 

X_train_flatten = X_train.reshape(len(X_train),28*28)
X_test_flatten = X_test.reshape(len(X_test ),28*28)


# In[28]:


X_train_flatten.shape


# In[30]:


model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,),activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    
)

model.fit(X_train_flatten, y_train, epochs=10)


# In[33]:


model.evaluate(X_test_flatten, y_test)


# In[34]:


plt.matshow(X_test[0])


# In[35]:


y_predicted = model.predict(X_test_flatten)
y_predicted[0]


# In[36]:


np.argmax(y_predicted[0])


# In[37]:


y_predicted_labels = [np.argmax(i) for i in y_predicted]
y_predicted_labels[:10]


# In[42]:


confusion_matrix = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
confusion_matrix


# In[39]:


import seaborn as sn
plt.figure(figsize=(10,8))
sn.heatmap(confusion_matrix, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[40]:


#now we will do the same thing using the hidden layer

model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,),activation='relu'),
    keras.layers.Dense(10 ,activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    
)

model.fit(X_train_flatten, y_train, epochs=10)


# In[41]:


model.evaluate(X_test_flatten, y_test)


# In[46]:


y_predicted = model.predict(X_test_flatten)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
confusion_matrix = tf.math.confusion_matrix(labels = y_test, predictions = y_predicted_labels)

plt.figure(figsize=(12,8))
sn.heatmap(confusion_matrix, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[50]:


#if I dont want to flatten all the tiem then the procedure is given below

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    
)

model.fit(X_train, y_train, epochs=5)


# In[ ]:




