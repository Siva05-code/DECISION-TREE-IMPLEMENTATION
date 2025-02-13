#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[3]:


# Reading the datasets
df = pd.read_csv("/Users/sivakarthick/Hub/Winequality/Decision_Tree/winequality-red.csv", delimiter=";")


# In[4]:


print(df.info())
print('\n')
print(df.describe())  
print('\n')
print(df.isnull().sum())  


# In[5]:


def categorize_quality(q):
    if q <= 5:
        return "Low"
    elif q == 6:
        return "Medium"
    else:
        return "High"

df["quality_label"] = df["quality"].apply(categorize_quality)


# In[6]:


# categorical labels into numerical values
label_mapping = {"Low": 0, "Medium": 1, "High": 2}
df["quality_label"] = df["quality_label"].map(label_mapping)


# In[7]:


#Fixing Features (X) and Target (y)
X = df.drop(columns=["quality", "quality_label"])  
y = df["quality_label"]


# In[8]:


# Training (80%) & Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26, stratify=y)


# In[9]:


# Normalize features 
s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)


# In[10]:


# Train Decision Tree Classifier
model = DecisionTreeClassifier(criterion="gini", max_depth=10, random_state=26)
model.fit(X_train, y_train)


# In[11]:


# Predict on Test Data
y_pred = model.predict(X_test)


# In[12]:


# Evaluating Model Performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[13]:


# Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[14]:


# Visualize Decision Tree
plt.figure(figsize=(16,10))
plot_tree(model, feature_names=X.columns, class_names=["Low", "Medium", "High"], filled=True, rounded=True)
plt.title("Decision Tree for Wine Quality Classification")
plt.show()

