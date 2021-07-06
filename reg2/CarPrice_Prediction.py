#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('car_data.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())


# In[6]:


##checking none or missing values
df.isnull().sum()


# In[7]:


df.describe()


# In[8]:


df['Years'] = int(datetime.datetime.now().year) - df['Year']
df=df.drop(['Car_Name', 'Year'], axis =1)


# In[9]:


df.head()


# In[10]:


df = pd.get_dummies(df, drop_first=True)
df.head()


# In[11]:


corr_matrix = df.corr()
top_corr_features = corr_matrix.index
plt.figure(figsize=(20,20))
sns.heatmap(df[top_corr_features].corr(),annot=True, cmap='RdYlGn')


# In[12]:


X = df.iloc[:,1:]
Y = df.iloc[:,0]


# In[13]:


from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,Y)


# In[14]:


feat_importance = pd.Series(model.feature_importances_, index= X.columns)
feat_importance.nlargest(5).plot(kind='barh')
plt.show()


# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)


# In[16]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
rf=RandomForestRegressor()

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# In[17]:


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 25, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[18]:


rf_random.fit(X_train,Y_train)


# In[19]:


predictions=rf_random.predict(X_test)
sns.distplot(Y_test-predictions)


# In[20]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(Y_test, predictions))
print('MSE:', metrics.mean_squared_error(Y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, predictions)))
rmse=np.sqrt(metrics.mean_squared_error(Y_test, predictions))


# In[21]:


print("Accuracy= {}".format(100*max(0,rmse)))


# In[22]:


import pickle
file = open('random_forest_regression_model.pkl', 'wb')
pickle.dump(rf_random, file)


# In[ ]:




