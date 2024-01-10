#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
    import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.impute import SimpleImputer


# In[21]:


# Read the stock data from CSV file
stock_data = pd.read_csv("D:\\CIT\\SEM-IV\\Predictive Analytics Project\\Microsoft Stock data.csv")
stock_data.head()


# # DATA PRE-PROCESSING

# In[23]:


# Print summary statistics of the stock data
print(stock_data.describe())


# In[24]:


#Checking for repeated values
stock_data=stock_data.drop_duplicates()
        stock_data[stock_data.duplicated()]


# In[25]:


#Null Values Check
stock_data.isnull().sum()


# In[26]:


# Splitting the data
np.random.seed(1234)
split = np.random.rand(len(stock_data)) < 0.90
training_set = stock_data[split]
test_set = stock_data[~split]
training_set.head()


# # PRINCIPAL COMPONENT ANALYSIS

# In[27]:


pca = PCA(n_components=4)
pca.fit(training_set.iloc[:, 2:7])
pca_data = pca.transform(training_set.iloc[:, 2:7])

# Print the results of PCA
print(pca_data)


# Plotting of PCA components

# In[28]:


# Plot the principal components
plt.scatter(pca_data[:, 0], pca_data[:, 1],color='red')
plt.show()


# # ARIMA FORECASTING

# In[29]:


#Fitting the model with Close value of stock
arima_model = ARIMA(stock_data['Close'], order=(1, 1, 1))
arima_model_fit = arima_model.fit()


# In[30]:


#Predicting the close value for the next 10 days
prediction = arima_model_fit.forecast(steps=10)
print(prediction)


# In[31]:


plt.plot(["01-01-2022","02-01-2022","03-01-2022","04-01-2022","05-01-2022","06-01-2022","07-01-2022","08-01-2022","09-01-2022","10-01-2022"],prediction, color='red')
plt.ylabel("Close Value")
plt.title("Close Value with date")
plt.show()


# # CLUSTERING

#     Find Optimum K-Value using Elbow Method

# In[32]:


from sklearn.preprocessing import StandardScaler
# Select the features that you want to cluster on
features = ['Open', 'High', 'Low', 'Close']
X_scaled =StandardScaler().fit_transform(training_set[features])
#Elbow Method
w = []
for k in range(1,7):
    km = KMeans(n_clusters=k, random_state=0, n_init = 10)
    km.fit(X_scaled)
    w.append(km.inertia_)
plt.figure(figsize = (10 ,3))
plt.plot(range(2, 8),w,color='red')
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.title("Elbow method")
plt.show()


# INFERENCE:
#     From the above plot, we can see that the value of wcss is showing a tragic downfall from to 2 to 3. In other cases, there is a straight line curve. So the optimum number of clusters is 3 i.e., k=3.

# # K-Means Algorithm For Clustering

# In[33]:


# Perform k-means clustering on the stock data
k=3
kmodel = KMeans(n_clusters=k, random_state=0,n_init=10)
kmodel.fit(training_set[features])
print(kmodel.labels_)

Plotting clusters 
# In[34]:


plt.scatter(training_set['Open'], training_set['Close'], c=kmodel.labels_)
plt.xlabel('Open')
plt.ylabel('Close')
plt.show()


# In[35]:


#Classifying the test set into clusters
c_test=kmodel.predict(test_set[features])
print(test_set)


# In[36]:


plt.scatter(test_set['Open'], test_set['Close'], c=c_test)
plt.xlabel('Open')
plt.ylabel('Close')
plt.show()


# INFERENCE:
#     From the plot, we find that most of the data in the test set fall in the 0th cluster.

# In[39]:


#Predicting the cluster for some random stock data
op=float(input("Enter open value:"))
cl=float(input("Enter close value:"))
hi=float(input("Enter high value:"))
lo=float(input("Enter low value:"))

data=pd.DataFrame({'Open':[op],'Close':[cl],'High':[hi],
                                     'Low':[lo]})
cluster=kmodel.predict(data[features])

print("The given data will belong to cluster:",cluster[0])


# # Discrimination and Classification

# In[ ]:





# In[ ]:




