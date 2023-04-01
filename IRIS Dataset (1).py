#!/usr/bin/env python
# coding: utf-8

# # Importing Modules

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.simplefilter("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# # Load the Dataset

# In[2]:


df = pd.read_csv('Iris.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# # Checking for Null Values

# In[5]:


df.isnull().sum()
#isnull() finds if there any NULL value is present or not and it gives the output in the form of TRUE or FALSE i.e., we used sum() function so that we can get the output in numeric form that is the sum of all the NULL values present in the dataset.


# # Some Basic Information about the Dataset

# In[6]:


df.columns


# In[7]:


df.shape


# In[8]:


df.describe()


# # Drop the Unwanted Columns

# In[9]:


df = df.drop(columns='Id') 
#Drop will delete the particular column given inside the paranthesis here the column is 'Id'


# In[10]:


df.head()


# In[11]:


df.shape
#after dropping one column ['Id'] now we have only 5 columns left


# # Label Encoding

# In[12]:


df["Species"] = LabelEncoder().fit_transform(df["Species"])
#This LabelEncoder() will change the categorical values of the 'Species' column into a numerical values


# In[13]:


df.head()


# # Data Visualization

# In[14]:


df["Species"].value_counts()
#This will provide the count value of each type of species


# In[15]:


sns.countplot(x='Species',data=df)


# In[16]:


sns.pairplot(df, hue='Species', height=3.0)


# # Splitting the Data

# In[17]:


x = df.iloc[:,:4]
y = df.iloc[:,4]
#x will store all the data from column 1 to 4 (0 to 3 in programming) i.e., SepalLengthCm, SepalWidthCm, PetalLengthCm and PetalWidthCm
#y will store the data of only column 5 (4 in programming) i.e., Species


# In[18]:


x.head()


# In[19]:


y.head()


# # Training and Testing the Data

# In[20]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
#It will split the data into training data and testing data (testing data will be 20% of the whole data and remaining 80% will be the training data)


# In[21]:


x_train.shape


# In[22]:


x_test.shape


# In[23]:


y_train.shape


# In[24]:


y_test.shape


# # Create the Model (classification)

# In[25]:


model = LogisticRegression().fit(x_train,y_train)
model


# In[26]:


y_pred = model.predict(x_test)


# In[27]:


y_pred


# In[28]:


score = accuracy_score(y_pred,y_test)
score


# # Testing the Model

# In[29]:


model.predict([[5,3.2,1.1,0.3]])

