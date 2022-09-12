#!/usr/bin/env python
# coding: utf-8

# # Basic Python

# ## 1. Split this string

# In[1]:


s = "Hi there Sam!"


# In[2]:


a = s.split()
print(a)


# ## 2. Use .format() to print the following string. 
# 
# ### Output should be: The diameter of Earth is 12742 kilometers.

# In[3]:


planet = "Earth"
diameter = 12742


# In[4]:


print("The diameter of {} is {} kilometers." .format("Earth",12742))


# ## 3. In this nest dictionary grab the word "hello"

# In[5]:


d = {'k1':[1,2,3,{'tricky':['oh','man','inception',{'target':[1,2,3,'hello']}]}]}


# In[6]:


d ['k1'][3]['tricky'][3]['target'][3]


# # Numpy

# In[7]:


import numpy as np


# ## 4.1 Create an array of 10 zeros? 
# ## 4.2 Create an array of 10 fives?

# In[8]:


array = np.zeros(10)
print(array)


# In[9]:


array = np.ones(10)*5
print(array)


# ## 5. Create an array of all the even integers from 20 to 35

# In[10]:


array = np.arange(20,36,2)
print(array)


# ## 6. Create a 3x3 matrix with values ranging from 0 to 8

# In[11]:


x = np.arange(0,9).reshape(3,3)
print(x)


# ## 7. Concatinate a and b 
# ## a = np.array([1, 2, 3]), b = np.array([4, 5, 6])

# In[12]:


a= np.array([1,2,3])
b= np.array([4,5,6])
x = np.concatenate((a,b),axis=0)
print(x)


# # Pandas

# ## 8. Create a dataframe with 3 rows and 2 columns

# In[13]:


import pandas as pd


# In[14]:


data =['Teddy','Kamal','Harry potter']
df = pd.DataFrame(data, columns=['Name'])
df


# ## 9. Generate the series of dates from 1st Jan, 2023 to 10th Feb, 2023

# In[15]:


d= pd.date_range(start='1-1-2023', end='10-02-2023')
print(d)


# ## 10. Create 2D list to DataFrame
# 
# lists = [[1, 'aaa', 22],
#          [2, 'bbb', 25],
#          [3, 'ccc', 24]]

# In[16]:


lists = [[1, 'aaa', 22], [2, 'bbb', 25], [3, 'ccc', 24]]


# In[17]:


df = pd.DataFrame(lists,columns=['S.No','XXX','No'])
print(df)

