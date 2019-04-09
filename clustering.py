
# coding: utf-8

# In[19]:


#Clustering using sklearn
#Source http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

from sklearn.cluster import KMeans
import numpy as np


# In[20]:


X = np.array([[1, 2], [1, 4], [1, 0],[4, 2], [4, 4], [4, 0]])
print(X)


# In[21]:


kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print(kmeans.labels_)

#predicting new data
kp = kmeans.predict([[0, 0], [4, 3]])
print(kp)
print(kmeans.cluster_centers_)


# In[ ]:


#K-Means Clustering with Scikit-Learn
#Source http://stackabuse.com/k-means-clustering-with-scikit-learn/

import matplotlib.pyplot as plt  
get_ipython().magic('matplotlib inline')
import numpy as np  
from sklearn.cluster import KMeans


# In[ ]:


#preprare data

X = np.array([[5,3],  
     [10,15],
     [15,12],
     [24,10],
     [30,45],
     [85,70],
     [71,80],
     [60,78],
     [55,52],
     [80,91],])


# In[ ]:


#visualize the data

plt.scatter(X[:,0],X[:,1], label='True Position') 


# In[ ]:


kmeans = KMeans(n_clusters=2)  
kmeans.fit(X)  


# In[ ]:


print(kmeans.cluster_centers_)  


# In[ ]:


print(kmeans.labels_)  


# In[ ]:


plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow') 


# In[ ]:


plt.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap='rainbow')  
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,0], color='black') 


# In[6]:


#Source https://analytics4all.org/2016/05/21/python-k-means-cluster/
#get data from excel 
import pandas as pd

df = pd.read_excel("kmeans1.xlsx")
df
#df.head


# In[7]:


#Now, we are going to drop a few columns: ID Tag – is a random number, has no value in clustering
#Dropping the columns
#This section is a part of feature selection

df1 = df.drop(["ID Tag", "Model", "Department"], axis = 1)
df1.head()


# In[8]:


#We then initialize KMeans

from sklearn.cluster import KMeans
km = KMeans(n_clusters=4, init='k-means++', n_init=10)


# In[9]:


#fit the model

km.fit(df1)


# In[17]:


#the cluster centers

print(km.cluster_centers_)


# In[18]:


#kmeans labels

print(km.labels_)  


# In[10]:


#export the cluster identifiers to a list

x = km.fit_predict(df1)
x


# In[11]:


#Create a new column on the original dataframe called Cluster and place your results (x) in that column

df["Cluster"]= x
df.head()


# In[13]:


#Sort your dataframe by cluster

df1 = df.sort_values(by=['Cluster'])
df1


# In[14]:


import matplotlib.pyplot as plt  
get_ipython().magic('matplotlib inline')

