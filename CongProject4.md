---
jupyter:
  jupytext:
    formats: ipynb,md,py:light
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Executive Summary
The aim of this report is to summarise, describe, and assess the process of solving two questions using a Data Science approach: Using information from job postings/listings, what factors impact the listed salary of data-related jobs the most (1), and what factors correlate with the industry/job type of the listed data job(2).

The first section of the report outlines the methodology and computer code that was designed and used to collect enough data to answer the project questions. Code was written to rapidly extract important information from over 18 000 job listings from SEEK.com, and more was written to 'clean up' the retrieved information for efficent Data Science analysis. Technically, these two processes are referred to as 'Web Scraping' and 'Data Cleaning'.

The second section describes the data science approach to answering question 1. We classified job listings as either high salary or low salary jobs based on whether the salary was above the aggregate median. This step allows us to use the data to train two classification Machine Learning models (BaggingClassifier & LogisticRegression) to their optimal accuracy levels. We can then perform analysis on our well-trained models to understand the key features/factors that the models used to accurately classify high and low salary jobs, and use these insights to futher understand the current data job market. Technically, this process is referred to as 'Inferencing'.

Section three describes the same inferencing approach to answering question 2. First, data jobs were classified as either being from Science-related industries or not. We also use difference Machine Learning models to perform inferencing to answer this question (SupportVectorClassification and AdaBoost ExtraTreeClassifier).  

**KeyFindings**  
Higher Paying Jobs are usually:
- Senior, more leadership type roles
- Experience with cloud computing and data infrastructure/architecture
- Involved with stakeholders
- Experience with SQL  

Lower Paying Jobs are usually:
- Analyst Roles
- Have some type of training

Science Related Data Jobs are usually:
- Have something to do with childcare
- Deal with equity

Non-Science Related Data Jobs:
- Deal with acquisitions
- Risk
- Transactions




## WebScraping Seek.com

This section contains the code to scrape around 9000 job postings. Data Munging and Data Cleaning will be in a separate notebook.

The impetus of this section was to build a holistic bundle of code. It even sends text messages when the web scraping is completed!

```python
# List of data job and page amounts to scrape

job_search_links = [['data-analyst', 200],
                   ['data-engineer', 80],
                   ['database-administrator', 40],
                   ['database-admin', 30],
                   ['machine-learning-engineer', 7],
                   ['machine-learning-developer', 10],
                   ['data-scientist', 15],
                   ['data-architect', 30],
                   ['big-data-architect', 5],
                   ['business-analyst', 200],
                   ['machine-learning-scientist', 5],
                   ['data-warehouse', 35],
                   ['business-intelligence-analyst', 40],
                   ['data-science-manager', 35],
                   ['data-science-consultant', 8],
                   ['data', 200]]
```

```python jupyter={"outputs_hidden": true}
# Gather ONLY the Links of Data Job Postings
data = []
for title, pages in job_search_links:    
    for i in range(1,(pages + 1)):
    # Getting URL INTO BTS OBJECT
        url = f"https://www.seek.com.au/{title}-jobs?page={i}"
        # Set the url
        url_data = requests.get(url)
        # Reqeuest url data
        url_bts = BeautifulSoup(url_data.content, 'lxml')
        # Turn url data into BS object

    # GETTING JOB LINKS FROM BTS OBJECT
        page_links = url_bts.find_all("a", {'class': "_2iNL7wI"})
        hiring_company = url_bts.find_all('a', {'class':'_3AMdmRg'})
        
        # Get all links
        for i in range(len(page_links)):
            row = {}
            row['Title'] = page_links[i].text
            row['Link'] = page_links[i].attrs['href'].split('?')[0]
        
            data.append(row)
        
            if len(data)%100 == 0:
                print(f"{len(data)} links scraped")

data_df = pd.DataFrame(data=data)
```

```python
# Drop Duplicate Links
data = data_df.drop_duplicates()
```

```python
# IMPORT Twilio Rest API. Will send a text message to my phone during the scraping process
from twilio.rest import Client
def send_sms(message="Default SMS"):
    account_sid = 'ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    auth_token = '017dXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    client = Client(account_sid, auth_token)

    message = client.messages \
        .create(
             body=f'{message} - Cong',
             from_='+1XXXXXXXXXXX',
             to='+61XXXXXXXXXX'
         )

    print(message.sid)
    
```

```python
# MAIN BLOCK OF WEB SCRAPING CODE. PERFORMED ALL WEBSCRAPING OF RELEVANT CODE.
# Create empty lists
data_details = []
data_questions = []
data_core = []
data_hirer = []

list_job_types = ['Full Time','Contract/Temp','Part Time','Casual/Vacation','Dail Rate Contract']
for link in data['Link']:
# GRAB THE WEBPAGE 
    url = f"https://www.seek.com.au{link}"
    url_data = requests.get(url)
    bts = BeautifulSoup(url_data.content, 'lxml')

# ESSENTIAL JOB INFO SECTION
    pandas_row = {}
    pandas_row['Link'] = link
    
    try:
        core_job_details = bts.find('div', {'class': 'K1Fdmkw JyFVSRZ'}).find('div', {'class': "Pdwn1mb"})
        core_job_details = core_job_details.find_all('dd')
    except:
        pass
    
    # Seek will always includes Date Posted and Job Type info into each Job Posting
    try:
        if core_job_details[3].text in list_job_types: # If a reward is present in the description, Job Type will ALWAYS the 3rd iterable
            pandas_row['Date_Posted'] = core_job_details[0].text
            pandas_row['Location_City'] = core_job_details[1].find_all('span', {'class':""})[0].find('strong').text
            try:
                pandas_row['Location_Region'] = core_job_details[1].find_all('span', {'class':""}
                                                                        )[0].find('span').text.replace(",","",1).strip()
            except:
                pass
            pandas_row['Reward'] = core_job_details[2].text
            pandas_row['Job_Type'] = core_job_details[3].text
            pandas_row['Industry'] = core_job_details[4].find_all('span', {'class':""})[0].find('strong').text
            pandas_row['Specialisation'] = core_job_details[4].find_all('span', {'class':""}
                                                                       )[0].find('span').text.replace(",","",1).strip()
    
        else:
            pandas_row['Date_Posted'] = core_job_details[0].text
            pandas_row['Location_City'] = core_job_details[1].find_all('span', {'class':""})[0].find('strong').text
            try:
                pandas_row['Location_Region'] = core_job_details[1].find_all('span', {'class':""}
                                                                        )[0].find('span').text.replace(",","",1).strip()
            except:
                pass
            pandas_row['Job_Type'] = core_job_details[2].text
            pandas_row['Industry'] = core_job_details[3].find_all('span', {'class':""})[0].find('strong').text
            pandas_row['Specialisation'] = core_job_details[3].find_all('span', {'class':""}
                                                                       )[0].find('span').text.replace(",","",1).strip()
    except:
        pass
    
    data_details.append(pandas_row)
        
# JOB QUESTION INFO
    new_row = {}
    new_row['Link'] = link
    
    try:
        questions = bts.find("ul",{'class': "_34zKk91"}).find_all('span', {'class':''})
        for i in range(len(questions)):
            new_row[f"Q{i+1}"] = questions[i].text
        data_questions.append(new_row)
    except:
        pass
    

# JOB DESCRIPTION TEXT INFO
    new_row = {}
    new_row['Link'] = link
    try:
        new_row['Text'] = bts.find('div', {'data-automation': "jobDescription"}).text
        new_row['Bullet_Text'] = [x.text for x in bts.find('div', {'data-automation': "jobDescription"}).find_all('li')]
        new_row['Strong_Text'] = [x.text for x in bts.find('div', {'data-automation': "jobDescription"}).find_all('strong')]
        new_row['Par_Text'] = [x.text for x in bts.find('div', {'data-automation': "jobDescription"}).find_all('p')]
    except:
        pass
    
    data_core.append(new_row)
    
# JOB HIRER INFO
    new_row = {}
    new_row['Link'] = link
    try:
        new_row['Hirer'] = bts.find('span' ,{'class': "_3FrNV7v _2QG7TNq E6m4BZb"}).text
    except:
        pass
    
    data_hirer.append(new_row)

# PRINT STATEMENTS AND SMS TESTING
    if len(data_details)%50 == 0:
        print(f"{len(data_details)} links scraped")
            
    if len(data_details)%4300 == 0:
        send_sms(f"{len(data_details)} links scraped")
        

# FINAL PRINT STATEMENT AND SMS SENDING
print("ALL LINKS SCRAPED")
send_sms('ALL LINKS SCRAPED')
    
```

```python
# Send scraped data into DataFrames and then into csv's
data_hirier = pd.DataFrame(data=data_hirer)
data_details = pd.DataFrame(data=data_details)
data_core = pd.DataFrame(data=data_core)
data_questions = pd.DataFrame(data=data_questions)

data.to_csv('data_.csv')
data_hirier.to_csv('data_hirer_.csv')
data_details.to_csv('data_details_.csv')
data_core.to_csv('data_core_.csv')
data_questions.to_csv('data_questions_.csv')
```

## Question 1: Determining Factors that Impact on Salary  
In this section I:
- Performed a brute force method to create the 'best' possible decision tree classifier model. 
- Created a Logistical Regression Classifier model that is easily intrepretable using ELI5 to gain key word insights
- Describe insights gained from inferencing our Logistical Regression Model

The impetus of this section was to understand and constrast computationally construsted model and relatively simple to build (and understand) models. 


**VIEWING DATASET**

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

```python
df = pd.read_csv("./SubmissionProject4/data_science.csv")
df.head()
```

### Bagging Classifier Optimisiation using GridSearchCV (BruteForce)

I performed two GridSearchCV's on two pipelines. 
1. To find the optimal CountVectorizer, TfidTransformer, SelectFdr, and DecisionTreeClassifier hyperparameters
2. To find the optimal BaggingClassifier hyperparameters that used the above Pipeline as it's base estimator

```python
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectFdr
from sklearn.metrics import plot_precision_recall_curve, plot_roc_curve, classification_report
import sklearn.metrics as metrics
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
```

#### GridSearching the BestDecisionTreeClassifier

```python
# Pipeline and GridSearchCV for best DecisionTreeClassifier
# Features and Target
X = df.job_description
y = df.above_ave_salary
print("===={:=<60}".format('Initialising GridSearch: DecisionTree Classifier'))
print("===={:=<60}".format(""))

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
print("===={:=<60}".format('Splitting Data into Training and Validation Models'))
print("===={:=<60}".format(""))

# Making Pipeline
print("===={:=<60}".format('Making Pipeline'))
cv = CountVectorizer()
tf = TfidfTransformer()
rdf = SelectFdr()
tree_c = DecisionTreeClassifier()

pipe = Pipeline([('cv', cv), 
                ('tf', None),
                ('rdf', None),
                ('tree', tree_c)])

print("===={:=<60}".format(""))

# Defining GridSearch Parameters
print("===={:=<60}".format('Performing GridSearch'))
param = {'cv__stop_words': [None, 'english'],
        'cv__max_df': [1.0, 0.5],
        'cv__min_df': [1, 0.1],
         'cv__ngram_range': [(1,1), (1,3)],
        'tf': [TfidfTransformer()],
        'tf__norm': ['l1', 'l2'],
        'rdf': [SelectFdr()],
        'rdf__alpha': [0.05, 0.1],
        'tree__criterion': ['gini', 'entropy']}

# Performing GridSearch
grid_search = GridSearchCV(estimator=pipe, param_grid=param, cv=5,
                          n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
         
```

```python
print(grid_search.best_params_)
```

```python
grid_search.best_score_
```

#### GridSearching the Bagging Classifier

```python
# Pipeline and GridSearchCV for BaggingClassifier
# Features and Target
X = df.job_description
y = df.above_ave_salary
print("===={:=<60}".format('Initialising GridSearch: BaggingClassifier'))
print("===={:=<60}".format(""))

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
print("===={:=<60}".format('Splitting Data into Training and Validation Models'))
print("===={:=<60}".format(""))

# Making Pipeline
print("===={:=<60}".format('Making Pipeline'))
cv = CountVectorizer(max_df=0.5, min_df=1, ngram_range=(1,1), stop_words=None)
tf = TfidfTransformer(norm= 'l2')
rdf = SelectFdr(alpha=0.1)
tree_c = DecisionTreeClassifier(criterion='entropy')
bag_c = BaggingClassifier(tree_c)

pipe = Pipeline([('cv', cv), 
                ('tf', tf),
                ('rdf', rdf),
                ('bag', bag_c)])

print("===={:=<60}".format(""))

# Defining GridSearch Parameters
print("===={:=<60}".format('Performing GridSearch'))
param = {'bag__n_estimators': [10, 50, 100, 400],
         'bag__max_samples': [1.0, 0.5, 0.2],
         'bag__max_features': [1.0, 0.5, 0.2]}

# Performing GridSearch
grid_search_2 = GridSearchCV(estimator=pipe, param_grid=param, cv=5,
                          n_jobs=-1, verbose=1)
grid_search_2.fit(X_train, y_train)

```

```python
grid_search_2.best_params_
```

```python
grid_search_2.best_score_
```

#### Creating the 'Best' Bagging Classifier

```python
# 'Best' Bagging Classifier on whether job is above salary

# Features and Target
X = df.job_description
y = df.above_ave_salary
print("===={:=<60}".format('Initialising Model: Bagging Classifier'))
print("===={:=<60}".format(""))

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
print("===={:=<60}".format('Splitting Data into Training and Validation Models'))
print("===={:=<60}".format(""))

# Count Vectorising and Tfid Vectorising
cv = CountVectorizer(ngram_range=(1,1), stop_words=None, max_df=0.5, min_df=1)
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)
print("===={:=<60}".format('Count Vectorising Text Data'))
print("===={:=<60}".format(f"No. of Features: {X_train.shape[1]}"))
print("===={:=<60}".format(""))

tf = TfidfTransformer(norm='l2')
X_train = tf.fit_transform(X_train)
X_test = tf.transform(X_test)
print("===={:=<60}".format('Tfidf Vectorising Text Data'))
print("===={:=<60}".format(f"No. of Features: {X_train.shape[1]}"))
print("===={:=<60}".format(""))

# Reducing Demensionality based on Fase Discovery Rate
fdr = SelectFdr(alpha=0.1)
X_train = fdr.fit_transform(X_train, y_train)
X_test = fdr.transform(X_test)
print("===={:=<60}".format('Reducing Feature Dimensions using FDR'))
print("===={:=<60}".format(f"No. of Features: {X_train.shape[1]}"))
print("===={:=<60}".format(""))

# Bagging Tree Classifier
print("===={:=<60}".format('Fitting Bagging Decision Tree Classifier'))
tree_c = DecisionTreeClassifier(criterion='entropy')
bag_c = BaggingClassifier(base_estimator=tree_c,
                         n_estimators=100,
                          max_features=1.0,
                          max_samples=0.5,
                         n_jobs=2)
bag_c.fit(X_train, y_train)

# Create Model Predictions
y_pred = bag_c.predict(X_test)
print("===={:=<60}".format('Creating Model Predictions'))
print("===={:=<60}".format(""))

# Validation of Our Model
print("===={:=<60}".format('Validating Training Model'))
print("===={:=<60}".format('Validation Scores'))
scores = cross_val_score(bag_c, X_train, y_train, cv=5, n_jobs=2)
print(scores)
print("===={:=<60}".format('Mean'))
print(scores.mean())
```

```python
# Bagging Decision Tree Classifier Metrics
print("===={:=<60}".format('Classification Report: Bagging Decision Tree Classifier'))
print(metrics.classification_report(y_test, y_pred))

print("===={:=<60}".format('ROC Curve & Precision Recall Curve'))
metrics.plot_roc_curve(bag_c, X_test, y_test)
metrics.plot_precision_recall_curve(bag_c, X_test, y_test)
plt.show()
```

### LogisticRegression Classifier
I created two Logistical Regression Models. Only a CountVectorizer was used for feature extraction and preprocessing to maintain intrepretability of the model using eli5. The two models differ based on the ngram_range parameter in the CountVectorizer: The length, in words, that a feature extracted from the text can be. This was done to corroborate the insights gained from both models. 

```python
# Logistic Regression Classifier on whether job is above salary

# Features and Target
X = df.job_description
y = df.above_ave_salary
print("===={:=<60}".format('Initialising Model: Logistic Regression'))
print("===={:=<60}".format(""))

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
print("===={:=<60}".format('Splitting Data into Training and Validation Models'))
print("===={:=<60}".format(""))

# Count Vectorising
cv = CountVectorizer(ngram_range=(1,1))
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)
print("===={:=<60}".format('Count Vectorising Text Data'))
print("===={:=<60}".format(f"No. of Features: {X_train.shape[1]}"))
print("===={:=<60}".format(""))


# Logistic Regression Classifier
print("===={:=<60}".format('Fitting Logistic Regression'))
lr = LogisticRegression(max_iter=500)
lr.fit(X_train, y_train)

# Create Model Predictions
y_pred = lr.predict(X_test)
print("===={:=<60}".format('Creating Model Predictions'))
print("===={:=<60}".format(""))

# Validation of Our Model
print("===={:=<60}".format('Validating Training Model'))
print("===={:=<60}".format('Validation Scores'))
print(cross_val_score(lr, X_train, y_train, cv=5, n_jobs=2))
print("===={:=<60}".format('Mean'))
print(cross_val_score(lr, X_train, y_train, cv=5, n_jobs=2).mean())
```

#### CountVectorizer ngram_range = (1,1)

```python
from sklearn.linear_model import LogisticRegression
```

```python
# Logistic Regression Classifier on whether job is above salary

# Features and Target
X = df.job_description
y = df.above_ave_salary
print("===={:=<60}".format('Initialising Model: Logistic Regression'))
print("===={:=<60}".format(""))

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
print("===={:=<60}".format('Splitting Data into Training and Validation Models'))
print("===={:=<60}".format(""))

# Count Vectorising
cv = CountVectorizer(ngram_range=(1,1))
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)
print("===={:=<60}".format('Count Vectorising Text Data'))
print("===={:=<60}".format(f"No. of Features: {X_train.shape[1]}"))
print("===={:=<60}".format(""))


# Logistic Regression Classifier
print("===={:=<60}".format('Fitting Logistic Regression'))
lr = LogisticRegression(max_iter=500)
lr.fit(X_train, y_train)

# Create Model Predictions
y_pred = lr.predict(X_test)
print("===={:=<60}".format('Creating Model Predictions'))
print("===={:=<60}".format(""))

# Validation of Our Model
print("===={:=<60}".format('Validating Training Model'))
print("===={:=<60}".format('Validation Scores'))
print(cross_val_score(lr, X_train, y_train, cv=5, n_jobs=2))
print("===={:=<60}".format('Mean'))
print(cross_val_score(lr, X_train, y_train, cv=5, n_jobs=2).mean())
```

```python
# Logistic Regression Metrics
print("===={:=<60}".format('Classification Report: Logistic Regression'))
print(metrics.classification_report(y_test, y_pred))

print("===={:=<60}".format('ROC Curve & Precision Recall Curve'))
metrics.plot_roc_curve(lr, X_test, y_test)
metrics.plot_precision_recall_curve(lr, X_test, y_test)
plt.show()
```

#### CountVectorizer ngram_range = (1,3)

```python jupyter={"source_hidden": true}
# Logistic Regression Classifier on whether job is above salary

# Features and Target
X = df.job_description
y = df.above_ave_salary
print("===={:=<60}".format('Initialising Model: Logistic Regression, ngram_range=(1,3)'))
print("===={:=<60}".format(""))

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
print("===={:=<60}".format('Splitting Data into Training and Validation Models'))
print("===={:=<60}".format(""))

# Count Vectorising
cv_2 = CountVectorizer(ngram_range=(1,3))
X_train = cv_2.fit_transform(X_train)
X_test = cv_2.transform(X_test)
print("===={:=<60}".format('Count Vectorising Text Data'))
print("===={:=<60}".format(f"No. of Features: {X_train.shape[1]}"))
print("===={:=<60}".format(""))

# Logistic Regression Classifier
print("===={:=<60}".format('Fitting Logistic Regression, ngram_range=(1,3)'))
lr_2 = LogisticRegression(max_iter=500)
lr_2.fit(X_train, y_train)

# Create Model Predictions
y_pred = lr_2.predict(X_test)
print("===={:=<60}".format('Creating Model Predictions'))
print("===={:=<60}".format(""))

# Validation of Our Model
print("===={:=<60}".format('Validating Training Model'))
print("===={:=<60}".format('Validation Scores'))
print(cross_val_score(lr_2, X_train, y_train, cv=5, n_jobs=2))
print("===={:=<60}".format('Mean'))
print(cross_val_score(lr_2, X_train, y_train, cv=5, n_jobs=2).mean())
```

```python jupyter={"source_hidden": true}
# Logistic Regression Metrics
print("===={:=<60}".format('Classification Report: Logistic Regression ngram_range=(1,3)'))
print(metrics.classification_report(y_test, y_pred))

print("===={:=<60}".format('ROC Curve & Precision Recall Curve'))
metrics.plot_roc_curve(lr_2, X_test, y_test)
metrics.plot_precision_recall_curve(lr_2, X_test, y_test)
plt.show()
```

### Inferencing from LogisticRegression Models
ELI5 is a simple library that lets you quickly view the heaviest weights of a model. For NLP, the weights reveal insights on word association to the target variable. 

```python jupyter={"outputs_hidden": true}
import eli5 as eli5
```

```python
# Showing key features from LogisticRegression
eli5.show_weights(estimator=lr, top=30, feature_names=cv.get_feature_names(),
                 target_names=['Low','High Salary Data Job'])
```

```python
# Showing key feautres from LogisticRegression ngram_range=(1,3)
eli5.show_weights(estimator=lr_2, top=30, feature_names=cv_2.get_feature_names(),
                 target_names=['Low','High Salary Data Job'])
```

## Question 2: Factors that affect Job Industry

In this section I attempted to gain insights into the textual differences between science relate and non-science related data jobs.  
I used two models with emphasis on intrepretability using ELI5:
1. Support Vector Classifier
2. AdaBoost ExtraTreeClassifier

The impetus of this section was to understand and create models I typically wouldn't use for this approach.


**VIEWING DATASET**

```python jupyter={"outputs_hidden": true}
df.head()
```

```python
df.classification.value_counts()
```

```python
# Why not classify jobs based on whether they are from a science related industry
science = ["Information & Communication Technology", "Science & Technology",
           "Healthcare & Medical", "Mining, Resources & Energy", 
           "Farming, Animals & Conservation"]

y = [1 if industry in science else 0 for industry in df['classification']]
print("===={:=<60}".format('Baseline'))
np.mean(y)
```

### SVM Classifier

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
```

```python
# SVM Classifier on whether job is science related

# Features and Target
science = ["Information & Communication Technology", "Science & Technology", "Healthcare & Medical", "Mining, Resources & Energy", "Farming, Animals & Conservation"]
X = df.job_description
y = [1 if industry in science else 0 for industry in df['classification']]
print("===={:=<60}".format('Initialising Model: SVC'))
print("===={:=<60}".format(""))

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
print("===={:=<60}".format('Splitting Data into Training and Validation Models'))
print("===={:=<60}".format(""))

# Count Vectorising
cv = CountVectorizer(ngram_range=(1,1))
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)
print("===={:=<60}".format('Count Vectorising Text Data'))
print("===={:=<60}".format(f"No. of Features: {X_train.shape[1]}"))
print("===={:=<60}".format(""))

# Scaling our Data
print("===={:=<60}".format('Scaling Data'))
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.todense())
X_test = scaler.transform(X_test.todense())
print("===={:=<60}".format(""))

# SVM Classifier
print("===={:=<60}".format('Fitting SVC'))
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

# Create Model Predictions
y_pred = svc.predict(X_test)
print("===={:=<60}".format('Creating Model Predictions'))
print("===={:=<60}".format(""))

# Validation of Our Model
print("===={:=<60}".format('Validating Training Model'))
print("===={:=<60}".format('Validation Scores'))
scores = cross_val_score(svc, X_train, y_train, cv=5, n_jobs=2)
print(scores)
print("===={:=<60}".format('Mean'))
print(scores.mean())
```

```python
# SVC Metrics
print("===={:=<60}".format('Classification Report: SVC'))
print(metrics.classification_report(y_test, y_pred))

print("===={:=<60}".format('ROC Curve & Precision Recall Curve'))
metrics.plot_roc_curve(svc, X_test, y_test)
metrics.plot_precision_recall_curve(svc, X_test, y_test)
plt.show()
```

### Inferencing from SVC

```python
eli5.show_weights(estimator=svc, top=(15, 15), feature_names=cv.get_feature_names(),
                 target_names=['Non Science Data Job','Science Data Job'])
```

### AdaBoost Classifier

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import ExtraTreeClassifier
```

```python
# AdaBoost Classifier on whether job is science related

# Features and Target
science = ["Information & Communication Technology", "Science & Technology", "Healthcare & Medical", "Mining, Resources & Energy", "Farming, Animals & Conservation"]
X = df.job_description
y = [1 if industry in science else 0 for industry in df['classification']]
print("===={:=<60}".format('Initialising Model: AdaBoost Classifier (ExtraTreeClassifier)'))
print("===={:=<60}".format(""))

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
print("===={:=<60}".format('Splitting Data into Training and Validation Models'))
print("===={:=<60}".format(""))

# Count Vectorising
cv = CountVectorizer(ngram_range=(1,1))
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)
print("===={:=<60}".format('Count Vectorising Text Data'))
print("===={:=<60}".format(f"No. of Features: {X_train.shape[1]}"))
print("===={:=<60}".format(""))

# # Scaling our Data
# print("===={:=<60}".format('Scaling Data'))
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train.todense())
# X_test = scaler.transform(X_test.todense())
# print("===={:=<60}".format(""))

# AdaBoost Classifier (ExtraTreeClassifier)
print("===={:=<60}".format('Fitting AdaBoost Clasifier (ExtraTreeClassifier)'))
e_tree = ExtraTreeClassifier()
ada = AdaBoostClassifier(e_tree, n_estimators=200)
ada.fit(X_train, y_train)

# Create Model Predictions
y_pred = ada.predict(X_test)
print("===={:=<60}".format('Creating Model Predictions'))
print("===={:=<60}".format(""))

# Validation of Our Model
print("===={:=<60}".format('Validating Training Model'))
print("===={:=<60}".format('Validation Scores'))
scores = cross_val_score(ada, X_train, y_train, cv=5, n_jobs=2)
print(scores)
print("===={:=<60}".format('Mean'))
print(scores.mean())
```

```python
# AdaBoost (ExtraTreeClassifier) Metrics
print("===={:=<60}".format('Classification Report: AdaBoost Classifier (ExtraTreeClassifier)'))
print(metrics.classification_report(y_test, y_pred))

print("===={:=<60}".format('ROC Curve & Precision Recall Curve'))
metrics.plot_roc_curve(ada, X_test, y_test)
metrics.plot_precision_recall_curve(ada, X_test, y_test)
plt.show()
```

### Inferencing from AdaBoost Classifier (ExtraTreeClassifier)

```python
eli5.show_weights(estimator=ada, top=30, feature_names=cv.get_feature_names(),
                 target_names=['Non Science Data Job','Science Data Job'])
```
