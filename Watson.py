#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import duckdb as db


# In[2]:


df = pd.read_excel("C:/Users/elifs/OneDrive/Masaüstü/new_watson_healthcare_modified.xlsx")


# In[33]:


db.query("""
    SELECT Round(AVG(age),2) AS AvgAe
    FROM df
""").to_df()


# In[15]:


db.query("""
    SELECT  EducationField, Round(AVG(age),2) AS EFAvgAge
    FROM df
    GROUP BY EducationField
""").to_df()


# In[18]:


db.query("""
    SELECT JobRole, Round(AVG(age),2) AS JRAvgAge
    FROM df
    GROUP BY JobRole
""").to_df()


# In[16]:


db.query("""
SELECT Department, COUNT(*) AS PersonNumber
FROM df
GROUP BY Department
ORDER BY personnumber;
""").to_df()


# In[22]:


db.query("""
SELECT Attrition, ROUND(AVG(DistanceFromHome), 2) AS AvgDistance
FROM df
GROUP BY Attrition;
""").to_df()


# In[25]:


db.query("""
SELECT Attrition, ROUND(AVG(TotalWorkingYears), 2) AS AvgTWY
FROM df
GROUP BY Attrition;
""").to_df()


# In[17]:


db.query("""
SELECT BusinessTravel,
       COUNT(*) AS PersonNumber,
       SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS Leaver,
       ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS AttritionRate
FROM df
GROUP BY BusinessTravel;
""").to_df()


# In[57]:


db.query("""
SELECT 
    CASE 
        WHEN Age < 30 THEN '18-29'
        WHEN Age BETWEEN 30 AND 39 THEN '30-39'
        WHEN Age BETWEEN 40 AND 49 THEN '40-49'
        ELSE '50+' 
    END AS AgeGroup,
    COUNT(*) AS PersonNumber,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS Leaver,
    ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS AttritionRate
FROM df
GROUP BY AgeGroup
ORDER BY AgeGroup ASC;
""").to_df()


# In[62]:


db.query("""
SELECT WorkLifeBalance, ROUND(AVG(MonthlyIncome), 2) AS AvgMI,
       COUNT(*) AS PersonNumber,
       SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS LeaverPN,
       ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS AttritionRate
FROM df
GROUP BY WorkLifeBalance
ORDER BY WorkLifeBalance ASC;
""").to_df()


# In[64]:


db.query("""
SELECT OverTime,
       COUNT(*) AS PersonNumber,
       SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS LeaverPN,
       ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS AttritionRate
FROM df
GROUP BY OverTime;
""").to_df()


# In[67]:


db.query("""
SELECT MaritalStatus,
       COUNT(*) AS PersonNumber,
       SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS LeaverPN,
       ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS AttritionRate
FROM df
GROUP BY MaritalStatus
ORDER BY AttritionRate ASC;
""").to_df()


# In[70]:


db.query("""
SELECT 
       CASE 
        WHEN YearsSinceLastPromotion < 2 THEN '0-1'
        WHEN YearsSinceLastPromotion BETWEEN 2 AND 4 THEN '2-4'
        ELSE '5+'
       END AS YSLPGroup,
       COUNT(*) AS PersonNumber,
       SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS LeaverPN,
       ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS AttritionRate
FROM df
GROUP BY YSLPGroup
ORDER BY YSLPGroup ASC;
""").to_df()


# In[73]:


db.query("""
SELECT 
       CASE 
        WHEN YearsWithCurrManager < 2 THEN '0-1'
        WHEN YearsWithCurrManager BETWEEN 2 AND 4 THEN '2-4'
        ELSE '5+'
       END AS YWCMGroup,
       COUNT(*) AS PersonNumber,
       SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS LeaverPN,
       ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS AttritionRate
FROM df
GROUP BY YWCMGroup
ORDER BY YWCMGroup ASC;
""").to_df()


# In[74]:


db.query("""
SELECT 
       CASE 
        WHEN YearsSinceLastPromotion < 2 THEN '0-1'
        WHEN YearsSinceLastPromotion BETWEEN 2 AND 4 THEN '2-4'
        ELSE '5+'
       END AS YSLPGroup,
       COUNT(*) AS PersonNumber,
       SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS LeaverPN,
       ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS AttritionRate
FROM df
GROUP BY YSLPGroup
ORDER BY YSLPGroup ASC;
""").to_df()


# In[76]:


db.query("""
SELECT 
       CASE 
        WHEN TotalWorkingYears < 2 THEN '0-1'
        WHEN TotalWorkingYears BETWEEN 2 AND 4 THEN '2-4'
        ELSE '5+'
       END AS TWYGroup,
       COUNT(*) AS PersonNumber,
       SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS LeaverPN,
       ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS AttritionRate
FROM df
GROUP BY TWYGroup
ORDER BY TWYGroup ASC;
""").to_df()


# In[ ]:




