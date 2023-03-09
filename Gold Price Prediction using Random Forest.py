#!/usr/bin/env python
# coding: utf-8

# In[117]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('Downloads/gld_price_data.csv')


# In[3]:


df.head()


# In[4]:


X = df.drop(['Date','GLD'],axis=1)


# In[15]:


y = df['GLD']


# In[16]:


from sklearn.model_selection import train_test_split


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[26]:


from sklearn.ensemble import RandomForestRegressor


# In[62]:


rfc = RandomForestRegressor(n_estimators=30,max_features=7,random_state=101)


# In[78]:


rfc.fit(X_train,y_train)


# In[65]:


from sklearn.metrics import mean_absolute_error, mean_absolute_error


# In[74]:


n_estimator = [10,20,30,40,50]
max_features = [2,3,4]
bootstrap = [True,False]
oob_score = [True,False]


# In[75]:


from sklearn.model_selection import GridSearchCV


# In[77]:


param_grid = {'n_estimators':n_estimator,'max_features':max_features,'bootstrap':bootstrap,'oob_score':oob_score}


# In[79]:


rfc = RandomForestRegressor()


# In[80]:


grid = GridSearchCV(rfc,param_grid)


# In[81]:


grid.fit(X_train,y_train)


# In[82]:


grid.best_params_


# In[83]:


rfc = RandomForestRegressor(max_features=2,n_estimators=50,oob_score=False,bootstrap=False)


# In[84]:


rfc.fit(X_train,y_train)


# In[85]:


predictions = rfc.predict(X_test)


# In[87]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[92]:


mse = mean_absolute_error(y_test,predictions)
mse


# In[93]:


rmse = np.sqrt(mean_absolute_error(y_test,predictions))
rmse


# In[128]:


y_test = list(y_test)
plt.subplots(figsize=(15,6))
plt.plot(y_test, color='blue', label = 'Actual Value')
plt.plot(predictions, color='green', label='Predicted Value')
plt.show()


# In[ ]:




