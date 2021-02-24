#!/usr/bin/env python
# coding: utf-8

# # INSAID Hiring Exercise

# ## Important: Kindly go through the instructions mentioned below.
# 
# - The Sheet is structured in **4 steps**:
#     1. Understanding data and manipulation
#     2. Data visualization
#     3. Implementing Machine Learning models(Note: It should be more than 1 algorithm)
#     4. Model Evaluation and concluding with the best of the model.
#     
#     
#     
# 
# - Try to break the codes in the **simplest form** and use number of code block with **proper comments** to them
# - We are providing **h** different dataset to choose from(Note: You need to select any one of the dataset from this sample sheet only)
# - The **interview calls** will be made solely based on how good you apply the **concepts**.
# - Good Luck! Happy Coding!

# ### Importing required libraries

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# ### Importing the data

# In[95]:


churn_data = pd.read_csv("Churn.csv")


# ### Understanding the data

# In[32]:


# Overview of data

churn_data.head()


# In[33]:


# Checking dimension,missing values and data type of data

churn_data.info()


# In[34]:


churn_data.isnull().sum()


# ### Data Manipulation

# In[96]:


# Converting TotalCharges to numeric

np.unique(churn_data.TotalCharges.values)


# In[97]:


# There are missing values in TotalCharges represented as '', that is why it is considered as Object data type
# Now will replace '' with nan

churn_data = churn_data.replace(to_replace=" ",value=np.nan)


# In[98]:


churn_data.TotalCharges = pd.to_numeric(churn_data.TotalCharges)


# In[51]:


churn_data.dtypes


# In[99]:


churn_data.isnull().sum()


# In[100]:


# Impute missing values with Median

churn_data.TotalCharges = churn_data.TotalCharges.fillna((churn_data.TotalCharges.median()))


# In[101]:


churn_data.TotalCharges.isnull().sum()


# In[67]:


# Drop CustomerID Column
churn_data.customerID = churn_data.drop(['customerID'],inplace=True,axis=1)


# In[102]:


churn_data.shape


# In[103]:


# Seperating numeric and categorical variables

numeric_variables = churn_data.select_dtypes(include=[np.number])
numeric_variables.columns


# In[104]:


categorical_variables = churn_data.select_dtypes(include=[np.object])
categorical_variables.columns


# ### Data Visualization

# In[12]:


sns.countplot(x= 'gender',data = churn_data)
plt.show() # Count of Male and Female is almost same


# In[13]:


sns.countplot(x="SeniorCitizen",data=churn_data)
plt.show() # from this graph we can say that ,in dataset there are 6 times less senior citizen


# In[14]:


sns.countplot(x='Dependents',data=churn_data)
plt.show() # there are two times more people with no dependents


# In[15]:


sns.countplot(x='Contract',data= churn_data)
plt.show() # most of the customer using monthly contract


# In[21]:


y=sns.countplot(x="PaymentMethod",data=churn_data)
plt.setp(y.get_xticklabels(),rotation=30)[1] # customers are using electronic check for payment


# In[28]:


# Plot all categorical varaibles with Churn

plt.figure(figsize=(15,15))

for k in range (1,len(categorical_variables.columns)) :
    plt.subplot(4,4,k)
    sns.countplot(categorical_variables.columns[k],data=churn_data,hue='Churn')


# From this graph it is observed that :
# Internet Service with fiber optic,customer with no online backup,no device protection,no technical support and those who have monthly contract are more likely to churn.

# In[105]:


# Converting categorical variables to numeric 

numeric_cat = categorical_variables.apply(lambda x : pd.factorize(x)[0])
numeric_cat.head()


# Here we can see that magnitude is different for tenure,MonthlyCharges,TotalCharges.So we will do scaling for these varaibles.

# In[72]:


# Scaling Data
from sklearn import preprocessing  


# In[73]:


scaler= preprocessing.StandardScaler()


# In[74]:


scale_variables = scaler.fit_transform(numeric_variables.iloc[:,1:])


# In[75]:


scale_varaibles =pd.DataFrame(scale_variables,columns=['tenure','MonthlyCharges','TotalCharges'])
scale_varaibles.head()


# In[76]:


# Merging scaled variables and SeniorCitizen
scale_varaibles = pd.concat([numeric_variables['SeniorCitizen'],scale_varaibles],axis=1)
scale_varaibles.head()


# In[79]:


# Merging all varaibles to one dataframe

final_df = pd.concat([scale_varaibles,numeric_cat],axis=1)
final_df.shape


# In[80]:


final_df.head()


# In[109]:


# Feature Selection Using Pearson Correlation

plt.figure(figsize=(15,15))
cor = final_df.corr()
sns.heatmap(cor,xticklabels=cor.columns, yticklabels=cor.columns,linewidths=.2,cmap='Greens')
plt.show()


# Churn is correlated with below varaibles :
# 
# SeniorCitizen,MonthlyCharges,Partner,MultipleLines,OnlineBackup,PhoneService

# ### Implement Machine Learning Models

# In[112]:


X = final_df.iloc[:,0:19]
X.head()


# In[117]:


Y = final_df['Churn']


# ## Random Forest

# In[191]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,roc_auc_score


# In[132]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,shuffle=False)


# In[133]:


X_train.shape


# In[129]:


X_test.shape


# In[134]:


rf_classifier = RandomForestClassifier(n_estimators=100,
                                       n_jobs=4,
                                       oob_score=True,
                                       warm_start=True,
                                       criterion='entropy')


# In[135]:


rf_model= rf_classifier.fit(X_train,Y_train)


# In[136]:


predict_rf = rf_model.predict(X_test)


# In[171]:


feature_imp = pd.Series(rf_model.feature_importances_,index=X_train.columns).sort_values(ascending=False)
feature_imp


# In[173]:


feature_varaible= rf_model.feature_importances_


# In[176]:


plt.barh(X_train.columns, feature_varaible)
plt.show()


# ## Model after feature Selection

# In[177]:


X_train_feature = X_train[['TotalCharges','MonthlyCharges','tenure','Contract','PaymentMethod']]


# In[178]:


X_test_feature = X_test[['TotalCharges','MonthlyCharges','tenure','Contract','PaymentMethod']]


# In[180]:


rf_feature = RandomForestClassifier(n_estimators=100,
                                       n_jobs=4,
                                       oob_score=True,
                                       warm_start=True,
                                       criterion='entropy')


# In[182]:


rf_feature_model= rf_feature.fit(X_train_feature,Y_train)


# In[183]:


predict_rf_feature = rf_feature_model.predict(X_test_feature)


# ## Logistic Regression

# In[142]:


from sklearn.linear_model import LogisticRegression


# In[143]:


logistic_classifier = LogisticRegression()


# In[145]:


logistic_classifier.fit(X_train,Y_train)


# In[149]:


logistic_predict = logistic_classifier.predict_proba(X_test)


# ## Model after feature selection

# In[186]:


logistic_classifier.fit(X_train_feature,Y_train)


# In[187]:


logistic_feature_predict = logistic_classifier.predict_proba(X_test_feature)


# ### Random forest Model Evaluation with all feature

# In[139]:


rf_confusionMatrix = confusion_matrix(Y_test,predict_rf)
rf_confusionMatrix


# In[141]:


rf_accuracy = metrics.accuracy_score(Y_test,predict_rf)
rf_accuracy


# ## Random forest Model Evaluation with  feature selection

# In[184]:


rf_feature_confusionMatrix = confusion_matrix(Y_test,predict_rf_feature)


# In[185]:


rf_feature_accuracy = metrics.accuracy_score(Y_test,predict_rf_feature)
rf_feature_accuracy


# ## Logistic Regression Model Evaluation with all feature

# In[154]:


logistic_accuracy = roc_auc_score(y_true=Y_test,y_score=logistic_predict[:,1])
logistic_accuracy


# ## Logistic Regression Model Evaluation with feature selection

# In[188]:


logistic_feature_accuracy = roc_auc_score(y_true=Y_test,y_score=logistic_feature_predict[:,1])
logistic_feature_accuracy


# ### Final Conclusions

# After implementing Random Forest and Logistic regression for this business problem, it is observed that Logfistic Regression is giving good accuracy i.e. 83.91
# Models are implemented with feature selection as well as with all features but models are performing well with all features. So we can go ahead with Logistic Regression model with all feature.
# 
