#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import duckdb as db


# In[3]:


df = pd.read_excel("C:/Users/elifs/OneDrive/Masaüstü/watson_healthcare_modified.xlsx")


# In[4]:


print(df.columns)


# In[5]:


db.query("""
    SELECT Round(AVG(age),2) AS avg_age
    FROM df
""").to_df()


# In[6]:


db.query("""
    SELECT  EducationField, Round(AVG(age),2) AS avg_age
    FROM df
    GROUP BY EducationField
""").to_df()


# In[7]:


db.query("""
    SELECT JobRole, Round(AVG(age),2) AS avg_age
    FROM df
    GROUP BY JobRole
""").to_df()


# In[1]:


db.query("""
SELECT COUNT(DISTINCT EmployeeID) AS n_age
FROM df
WHERE age > (SELECT AVG(age) FROM df);
""").to_df()


# In[ ]:





# In[ ]:





# In[ ]:




