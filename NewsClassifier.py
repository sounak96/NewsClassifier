#!/usr/bin/env python
# coding: utf-8

# In[24]:


#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from sklearn.datasets import fetch_20newsgroups
get_ipython().run_line_magic('matplotlib', 'inline')
data = fetch_20newsgroups()
data.target_names


# In[25]:


categories = data.target_names
#Training the data on these categories
train = fetch_20newsgroups(subset='train',categories=categories)
#Testing the data on these categories
test = fetch_20newsgroups(subset='test',categories=categories)


# In[26]:


#importing necessary packages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
#Creating a model on multinomial naive bayes
model = make_pipeline(TfidfVectorizer(),MultinomialNB())
#Training the model with train data
model.fit(train.data,train.target)
#Creating labels for the test data
labels = model.predict(test.data)


# In[27]:


#creating confusion matrix and heat map
from sklearn.metrics import confusion_matrix
mat=confusion_matrix(test.target,labels)
sns.heatmap(mat.T,square=True,annot=True,fmt='d',cbar=True,xticklabels=train.target_names,yticklabels=train.target_names)
#plotting heatmap of confusion matrix
plt.xlabel('true label')
plt.ylabel('predicted label')


# In[28]:


#predicting labels on new data
def predict_category(s,train=train,model=model):
    pred=model.predict([s])
    return train.target_names[pred[0]]


# In[29]:


predict_category('School Shooting in Columbine')


# In[ ]:




