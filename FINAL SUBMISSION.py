# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light,md
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # **Project 4 - Analysing Data Relating Job Postings**

# **CODE IMPORTS**

# +
# Standard Imports
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

# Natural Language Processing Modules
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

# Machine Learning Building Modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFdr
from sklearn.model_selection import train_test_split, cross_val_score

# Machine Learning Evaulation Metrics
import sklearn.metrics as metrics
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve, plot_confusion_matrix
from sklearn.metrics import classification_report

# Eli5
import eli5
# -

# ## Executive Summary

# The aim of this report is to summarise, describe, and assess the process of solving two questions using a Data Science approach: Using information from job postings/listings, what factors impact the listed salary of data-related jobs the most (1), and what factors correlate with the industry/job type of the listed data job(2).
#
# The first section of the report outlines the methodology and computer code that was designed and used  to collect enough data to answer the project questions. Code was written to rapidly extract important information from over 18 000 job listings from SEEK.com, and more was written to 'clean up' the retrieved information for efficent Data Science analysis. Technically, these two processes are referred to as 'Web Scraping' and 'Data Cleaning'.
#
# The second section describes the data science approach to answering question 1. We classified job listings as either high salary or low salary jobs based on whether the salary was above the aggregate median. This step allows us to use the data to train two classification Machine Learning models (RandomForestDecisionTreeClassifier & LogisticRegression) to their optimal accuracy levels. We can then perform analysis on our well-trained models to understand the key features/factors that the models used to accurately classify high and low salary jobs, and use these insights to futher understand the current data job market. Technically, this process is referred to as 'Inferencing'.
#
# Section three describes the same inferencing approach to answering question 2. First, data jobs were classified as either being from Science-related industries or not. We also use difference Machine Learning models to perform inferencing to answer this question (XDJksljdlsa).  



# ## Question 1: Factors that Impact Salary

# ### Loading and Analysing Data

# +
# Loading and Analysing Data
df = pd.read_csv('./SubmissionProject4/data_science.csv')
print(f"""
Shape: 
{df.shape}
=======================
Jobs below (0) and above (1) median salary:
{df.above_ave_salary.value_counts()}

Baseline Score:
{df.above_ave_salary.value_counts(normalize=True)[0]}

""")


df.head()
#     Quick Overview of Data
# -

# ### Building K-Nearest Neighbours (KNN) Classifier

print("===={:=<15}".format('Neighbours'))

# +
# KNN Classifer on whether job is above salary
# Features and Target
X = df.job_description
y = df.above_ave_salary
print("===={:=<60}".format('Initialising Model: KNN Classifier'))
print("===={:=<60}".format(""))

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
print("===={:=<60}".format('Splitting Data into Training and Validation Models'))

# Count Vectorising
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)
print("===={:=<60}".format('Count Vectorising Text Data'))
print("===={:=<60}".format(f"No. of Features: {X_train.shape[1]}"))


# Reducing Demensionality based on Fase Discovery Rate
fdr = SelectFdr()
X_train = fdr.fit_transform(X_train, y_train)
X_test = fdr.transform(X_test)
print("===={:=<60}".format('Reducing Feature Dimensions using FDR'))
print("===={:=<60}".format(f"No. of Features: {X_train.shape[1]}"))
print("===={:=<60}".format(""))

# KNN Classifier
print("===={:=<60}".format('Building and Fitting KNN Classifier'))
knn_c = KNeighborsClassifier()
knn_c.fit(X_train, y_train)

# Create Model Predictions
y_pred = knn_c.predict(X_test)
print("===={:=<60}".format('Creating Model Predictions'))
print("===={:=<60}".format(""))

# Validation of Our Model
print("===={:=<60}".format('Validating Training Model'))
print("===={:=<60}".format('Validation Scores'))
print(cross_val_score(knn_c, X_train, y_train, cv=5, n_jobs=2))
print("===={:=<60}".format('Mean'))
print(cross_val_score(knn_c, X_train, y_train, cv=5, n_jobs=2).mean())

# +
# KNN Classifier Metrics
print("===={:=<60}".format('Classification Report: Bagging Decision Tree Classifier'))
print(metrics.classification_report(y_test, y_pred ))

print("===={:=<60}".format('ROC Curve & Precision Recall Curve'))
metrics.plot_roc_curve(knn_c, X_test, y_test)
metrics.plot_precision_recall_curve(knn_c, X_test, y_test)
plt.show()
# -

# ### Bagging Decision Tree Classifier

# +
# KNN Classifer on whether job is above salary
# Features and Target
X = df.job_description
y = df.above_ave_salary
print("===={:=<60}".format('Initialising Model: Bagging Decision Tree Classifier'))
print("===={:=<60}".format(""))


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
print("===={:=<60}".format('Splitting Data into Training and Validation Models'))

# Count Vectorising
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)
print("===={:=<60}".format('Count Vectorising Text Data'))
print("===={:=<60}".format(f"No. of Features: {X_train.shape[1]}"))


# Reducing Demensionality based on Fase Discovery Rate
fdr = SelectFdr()
X_train = fdr.fit_transform(X_train, y_train)
X_test = fdr.transform(X_test)
print("===={:=<60}".format('Reducing Feature Dimensions using FDR'))
print("===={:=<60}".format(f"No. of Features: {X_train.shape[1]}"))
print("===={:=<60}".format(""))

# Bagging Tree Classifier
print("===={:=<60}".format('Building and Fitting Bagging Decision Tree Classifier'))
tree_c = DecisionTreeClassifier()
bag_c = BaggingClassifier(base_estimator=tree_c,
                         n_estimators=100,
                         n_jobs=2)
bag_c.fit(X_train, y_train)

# Create Model Predictions
y_pred = bag_c.predict(X_test)
print("===={:=<60}".format('Creating Model Predictions'))
print("===={:=<60}".format(""))

# Validation of Our Model
print("===={:=<60}".format('Validating Training Model'))
print("===={:=<60}".format('Validation Scores'))
print(cross_val_score(bag_c, X_train, y_train, cv=5, n_jobs=2))
print("===={:=<60}".format('Mean'))
print(cross_val_score(bag_c, X_train, y_train, cv=5, n_jobs=2).mean())

# +
# Bagging Decision Tree Classifier Metrics
print("===={:=<60}".format('Classification Report: Bagging Decision Tree Classifier'))
print(metrics.classification_report(y_test, y_pred))

print("===={:=<60}".format('ROC Curve & Precision Recall Curve'))
metrics.plot_roc_curve(bag_c, X_test, y_test)
metrics.plot_precision_recall_curve(bag_c, X_test, y_test)
plt.show()
# -

# ### LogisticRegression Classifier

# +
# LogisticRegression Classifer on whether job is above salary
# Features and Target
X = df.job_description
y = df.above_ave_salary
print("===={:=<60}".format('Initialising Model: Logistic Regression CV'))
print("===={:=<60}".format(""))

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
print("===={:=<60}".format('Splitting Data into Training and Validation Models'))

# Count Vectorising
cv = CountVectorizer(ngram_range=(1,3))
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)
print("===={:=<60}".format('Count Vectorising Text Data'))
print("===={:=<60}".format(f"No. of Features: {X_train.shape[1]}"))


# Reducing Demensionality based on Fase Discovery Rate
# fdr = SelectFdr()
# X_train = fdr.fit_transform(X_train, y_train)
# X_test = fdr.transform(X_test)
# print("===={:=<60}".format('Reducing Feature Dimensions using FDR'))
# print("===={:=<60}".format(f"No. of Features: {X_train.shape[1]}"))
# print("===={:=<60}".format(""))


# Logistic Regression Classifier
print("===={:=<60}".format('Building and Fitting Logistic Regression CV'))
lr_cv = LogisticRegression(max_iter=500)
lr_cv.fit(X_train, y_train)

# Create Model Predictions
y_pred = lr_cv.predict(X_test)
print("===={:=<60}".format('Creating Model Predictions'))
print("===={:=<60}".format(""))

# Validation of Our Model
print("===={:=<60}".format('Validating Training Model'))
print("===={:=<60}".format('Validation Scores'))
print(cross_val_score(lr_cv, X_train, y_train, cv=5, n_jobs=2))
print("===={:=<60}".format('Mean'))
print(cross_val_score(lr_cv, X_train, y_train, cv=5, n_jobs=2).mean())

# +
# Logistic Regression Metrics
print("===={:=<60}".format('Classification Report: Bagging Decision Tree Classifier'))
print(metrics.classification_report(y_test, y_pred))

print("===={:=<60}".format('ROC Curve & Precision Recall Curve'))
metrics.plot_roc_curve(lr_cv, X_test, y_test)
metrics.plot_precision_recall_curve(lr_cv, X_test, y_test)
plt.show()
# -

# ### RandomForest

# +
# KNN Classifer on whether job is above salary
# Features and Target
X = df.job_description
y = df.above_ave_salary
print("===={:=<60}".format('Initialising Model: RandomForestClassifier'))
print("===={:=<60}".format(""))


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
print("===={:=<60}".format('Splitting Data into Training and Validation Models'))

# Count Vectorising
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)
print("===={:=<60}".format('Count Vectorising Text Data'))
print("===={:=<60}".format(f"No. of Features: {X_train.shape[1]}"))

# Reducing Demensionality based on Fase Discovery Rate
fdr = SelectFdr()
X_train = fdr.fit_transform(X_train, y_train)
X_test = fdr.transform(X_test)
print("===={:=<60}".format('Reducing Feature Dimensions using FDR'))
print("===={:=<60}".format(f"No. of Features: {X_train.shape[1]}"))
print("===={:=<60}".format(""))

# Bagging Tree Classifier
print("===={:=<60}".format('Building and Fitting RandomForestClassifier'))
tree_c = RandomForestClassifier()
tree_c.fit(X_train, y_train)

# Create Model Predictions
y_pred = tree_c.predict(X_test)
print("===={:=<60}".format('Creating Model Predictions'))
print("===={:=<60}".format(""))

# Validation of Our Model
print("===={:=<60}".format('Validating Training Model'))
print("===={:=<60}".format('Validation Scores'))
print(cross_val_score(tree_c, X_train, y_train, cv=5, n_jobs=2))
print("===={:=<60}".format('Mean'))
print(cross_val_score(tree_c, X_train, y_train, cv=5, n_jobs=2).mean())

# +
# Logistic Regression Metrics
print("===={:=<60}".format('Classification Report: RandomForestClassifier'))
print(metrics.classification_report(y_test, y_pred))

print("===={:=<60}".format('ROC Curve & Precision Recall Curve'))
metrics.plot_roc_curve(tree_c, X_test, y_test)
metrics.plot_precision_recall_curve(tree_c, X_test, y_test)
plt.show()
# -

# ### Inferencing Models for Insights

# ELI5 KNN
eli5.show_weights(estimator=lr_cv, top=30, feature_names=cv.get_feature_names(),
                 target_names=['Low','High Salary Data Job'])

# ELI5 KNN
eli5.show_weights(estimator=lr_cv, top=30, feature_names=cv.get_feature_names(),
                 target_names=['Low','High Salary Data Job'])

eli5.show_weights(estimator=tree_c, top=(20), feature_names=cv.get_feature_names(),)

eli5.show_weights(estimator=tree_c, top=(20), feature_names=fdr.(),)

# BEST QUESTION 1 MODEL

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

# +
# Optimising the Best Classification Pipeline using GridsearchCV

cv = CountVectorizer()
tf = TfidfTransformer()
rdf = SelectFdr()
tree_c = DecisionTreeClassifier()
bag_c = BaggingClassifier(tree_c, n_jobs=-1, verbose=1)

pipe = Pipeline([('cv', cv), 
                ('tf', None),
                ('rdf', None),
                ('tree', tree_c)])
# -

param = {'cv__stop_words': [None, 'english'],
        'cv__max_df': [1.0, 0.7, 0.5, 0.2],
        'cv__min_df': [1, 0.1, 10],
         'cv__ngram_range': [(1,1), (1,3)],
        'tf': [None, tf],
        'tf__norm': ['l1', 'l2'],
        'rdf': [None, rdf],
        'rdf__alpha': [0.05, 0.1],
        'bag__base_estimator__criterion': ['gini', 'entropy'],
        'bag__base_estimator__max_depth': [None, 10, 5, 2],
        'bag__base_estimator__max_features': [None, 'auto', 'log2'],
        'bag__n_estimators': [10, 50, 100]}

param = {'cv__stop_words': [None, 'english'],
        'cv__max_df': [1.0, 0.7, 0.5, 0.2],
        'cv__min_df': [1, 0.1, 10],
         'cv__ngram_range': [(1,1), (1,3)],
        'tf': [TfidfTransformer()],
        'tf__norm': ['l1', 'l2'],
        'rdf': [SelectFdr()],
        'rdf__alpha': [0.05, 0.1],
        'tree__criterion': ['gini', 'entropy'],
        'tree__max_depth': [None, 10, 5, 2],
        'tree__max_features': [None, 'auto', 'log2']}

# +
X = df.job_description
y = df.above_ave_salary

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
# -

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(estimator=pipe, param_grid=param, cv=5, n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)
