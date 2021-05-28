#!/usr/bin/env python
# coding: utf-8

# # handwritten digit recognition problem 

# In[2]:


from sklearn.datasets import load_digits 
digits = load_digits()


# In[3]:


dir(digits)


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[5]:


plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])


# In[7]:


import pandas as pd
df = pd.DataFrame(digits.data)
df.head()


# In[8]:


df['target']=digits.target
df[0:12]


# In[9]:


x = df.drop('target',axis=1)
y = df.target


# In[13]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[14]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train,y_train)


# In[16]:


model.score(X_test,y_test)


# In[17]:


y_predicted = model.predict(X_test)


# In[19]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predicted)
cm


# In[20]:


import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[28]:


model.predict(x.iloc[0,:].values.reshape(1,-1))


# In[30]:


model.predict(x.iloc[2,:].values.reshape(1,-1))


# In[ ]:




