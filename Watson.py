#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import duckdb as db
import matplotlib.pyplot as plt
import os


# In[2]:


df = pd.read_excel("C:/Users/elifs/OneDrive/Masaüstü/new_watson_healthcare_modified.xlsx")


# In[7]:


# Average age of employees #
db.query("""
    SELECT Round(AVG(age),2) AS AvgAe
    FROM df
""").to_df()


# In[15]:


# Average age by education field #
db.query("""
    SELECT  EducationField, Round(AVG(age),2) AS EFAvgAge
    FROM df
    GROUP BY EducationField
""").to_df()


# In[18]:


# Average age by job role #
db.query("""
    SELECT JobRole, Round(AVG(age),2) AS JRAvgAge
    FROM df
    GROUP BY JobRole
""").to_df()


# In[16]:


# Number of person by department #
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


# In[77]:


db.query("""
WITH AgeGroups AS (
    SELECT 
        CASE 
            WHEN Age < 30 THEN '18-29'
            WHEN Age BETWEEN 30 AND 39 THEN '30-39'
            WHEN Age BETWEEN 40 AND 49 THEN '40-49'
            ELSE '50+' 
        END AS AgeGroup,
        OverTime,
        Attrition,
        MonthlyIncome,
        JobSatisfaction
    FROM df
),
OverTimeSummary AS (
    SELECT 
        AgeGroup,
        OverTime,
        COUNT(*) AS PersonCount,
        SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS Leavers,
        ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS AttritionRate,
        ROUND(AVG(MonthlyIncome), 2) AS AvgIncome,
        ROUND(AVG(JobSatisfaction), 2) AS AvgSatisfaction
    FROM AgeGroups
    GROUP BY AgeGroup, OverTime
)
SELECT *
FROM OverTimeSummary
ORDER BY AgeGroup ASC, OverTime DESC;
""").to_df()


# In[3]:


df = df.copy()
df["Attrition"] = df["Attrition"].map({"Yes":1, "No":0})


# In[4]:


for c in ["EmployeeID", "EducationField", "EducationField", "TotalWorkingYears", "TrainingTimesLastYear", "YearsWithCurrManager"]:
    if c in df.columns:
        df.drop(columns=c, inplace=True)


# In[5]:


cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
cat_cols.append("JobLevel")
print("Kategorik sütunlar:", cat_cols)


# In[6]:


df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)


# In[7]:


X = df_encoded.drop("Attrition", axis=1)
y = df_encoded["Attrition"]


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[9]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[10]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)


# In[119]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))


# In[83]:


from sklearn.metrics import confusion_matrix as cm
cm = confusion_matrix(y_test, y_pred)
cm = np.array(cm)
plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
for (i, j), val in np.ndenumerate(cm):
    plt.text(j, i, val, ha='center', va='center')
plt.tight_layout()
plt.show()


# In[84]:


from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_test, y_proba):.3f}")
plt.plot([0,1],[0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()


# In[85]:


feature_names = X.columns.tolist()
coefs = pd.Series(model.coef_[0], index=feature_names).sort_values(ascending=False)
print("\nTop positive coefficients (increase attrition odds):")
print(coefs.head(10))
print("\nTop negative coefficients (decrease attrition odds):")
print(coefs.tail(10))


# In[11]:


feature_columns = df_encoded.drop("Attrition", axis=1).columns

feature_columns = feature_columns[:X_test.shape[1]]

risk_df = pd.DataFrame(X_test, columns=feature_columns)

# Tahmin olasılıkları ve gerçek değerleri ekle
y_pred_proba = model.predict_proba(X_test)[:, 1]
risk_df['Attrition_Prob'] = y_pred_proba
risk_df['Actual_Attrition'] = y_test.values

print(risk_df.head())


# In[12]:


current_employees = risk_df[risk_df['Actual_Attrition'] == 0]
high_risk_current = current_employees.sort_values('Attrition_Prob', ascending=False)
high_risk_current = high_risk_current[high_risk_current['Attrition_Prob'] > 0.7]

print("Şirkette hâlâ çalışıp yüksek ayrılma riski taşıyan kişi sayısı:", len(high_risk_current))
high_risk_current


# In[ ]:




