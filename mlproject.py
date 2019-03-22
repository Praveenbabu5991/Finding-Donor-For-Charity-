# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 18:42:49 2018

@author: Bolt
"""

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames
from __future__ import division #for division
import matplotlib.pyplot as plt
import seaborn as sns



# Load the Census dataset
data = pd.read_csv("d1.csv")

# Success - Display the first record
display(data.head(10))
data.columns
#lets drop fnlwgt
data=data.drop('fnlwgt',axis=1) 



#summary stats
data.describe()
print(data.shape)
print(data.columns)
print(data.dtypes)
#to check is there any null values
data.isnull().sum().sum()
data.summary()
 #DATA EXPLORATION
# Total number of records
n_records = data.shape[0]

# Number of records where individual's income is more than $50,000
n_greater_50k = data.income.value_counts()[1].astype(int)

# Number of records where individual's income is at most $50,000
n_at_most_50k = data.income.value_counts()[0].astype(int)

# Percentage of individuals whose income is more than $50,000
greater_percent = n_greater_50k / n_records

# Print the results
print ("Total number of records: {}".format(n_records))
print ("Individuals making more than $50,000: {}".format(n_greater_50k))
print ("Individuals making at most $50,000: {}".format(n_at_most_50k))
print ("Percentage of individuals making more than $50,000: {:.2F}%".format(greater_percent * 100))


# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)
type(income_raw)

#histogram
data['capital-gain'].hist()
data['capital-loss'].hist()
#other way to create a histogram
data['capital-gain'].plot(kind='hist',bins=10)
#to find area 
data.plot.area()
data.plot.bar()



#joint plot
sns.jointplot(x='capital-gain',y='capital-loss',data=data,kind='hex')
#pairplot for numeric value
sns.pairplot(data,hue='sex')
sns.barplot(x='sex',y='education-num',data=data)
#count plot  count the no of occurences
sns.countplot(x='sex',data=data)
sns.countplot(x='workclass',data=data)
sns.countplot(x='education_level',data=data)
sns.countplot(x='marital-status',data=data)
sns.countplot(x='occuption',data=data)
sns.countplot(x='relationship',data=data)
sns.countplot(x='race',data=data)
sns.countplot(x='native-country',data=data)


#matrixplot give corelation matrix and heatmap
tc=data.corr()
sns.heatmap(tc)

#distribution plot
sns.distplot(data['capital-gain'],bins=40)
sns.distplot(data['capital-loss'],bins=40)

#Notice the strong positive skew present in the capital-gain and 
#capital-loss features. In order to compress the range of our dataset
# and deal with outliers we will perform a log transformation using
# np.log(). However, it's important to remember that the log of zero
# is undefined so we will add 1 to each sample
# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1)) #add 1
features_raw.head()
# Visualize the new log distributions
features_raw['capital-gain'].hist()
features_raw['capital-loss'].hist()


sns.distplot(features_raw['capital-gain'])
sns.distplot(features_raw['capital-loss'])


#preprocessing(normalization)
#After implementing our log transformation, it's good practice to perform
#scaling on numerical features so that each feature will be weighted
#equally when we have our algorithm ingest it. once scaling has
#been applied, the features will not be recognizable.

#To do this we'll employ sklearn's sklearn.preprocessing.MinMaxScaler.
#Any outliers will dramatically effect the results of the scaling, that's 
#why we handled them with a log transformation in the previous step.
#What's happening under-the-hood of this function is a simple division 
#and subtraction to re-weight each sample within each feature
#such that they all fall within the range (0,1)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical]=scaler.fit_transform(data[numerical])

display(features_raw.head(1))

#enoding categorical data,ml algo likes to work with numerical data
features=pd.get_dummies(features_raw)
features.head()
# Encode the 'income_raw' data to numerical values

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
income = labelencoder_X.fit_transform(income_raw)

# Print the number of features after one-hot encoding
encoded = list(features.columns)
print ("{} total features after encoding.".format(len(encoded)))

#splitting train and test data
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(features,income,test_size=0.2,random_state=0)

# Show the results of the split
print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))

#establishing model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#naive classifier
y_pred = np.ones(32561)

#confusion matrix
cm = confusion_matrix(income,y_pred)

#metrics
tn = cm[0,0]
fp = cm[0,1]
fn = cm[1,0]
tp = cm[1,1]

recall = tp / (tp + fn)
precision = tp / (tp + fp)
'''
* denote precision and recallin one score we use average(f1 score)
*  eg.precision=100 recall=0 average =50
*f1 score exist between two values

* f1 score uses harmonic mean(2xy/x+y) eg=p=0.2,r=0.8,avg=0.5,hm=0.32
'''
beta = .5

# Calculate accuracy
accuracy = (tp + tn) / (tp + tn + fp + fn)
'''
accuracy is not a good predictor
credit card transcation
accuracy=284335/284887=99.83%
'''

# Calculate F-score using the formula above for beta = 0.5

fscore = (1 + beta**2) * ((precision * recall) / ((beta**2 * precision) + recall))

# Print the results
print ("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))


from sklearn.metrics import accuracy_score, fbeta_score #metrics / scoring
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from time import time

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''

    results = {}

    # Fit the learner to the training data using slicing with 'sample_size'
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size]) #sample_weight=sample_size
    end = time() # Get end time

    # Calculate the training time
    results['train_time'] = (end - start)

    # Get the predictions on the test set,
    #       then get predictions on the first 300 training samples
    start = time() # Get start time
    predictions_test = learner.predict(X_test) #pred = clf.predict(features_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time

    # Total prediction time
    results['pred_time'] = (end - start)

    # Compute accuracy on 300 training samples
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)

    # Compute accuracy on test set
    results['acc_test']  = accuracy_score(y_test, predictions_test)


    # Compute F-score on 300 training samples
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=.5)


    # Compute F-score on the test set
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=.5)

    # Success
    print ("{} trained on {} samples.".format(learner, sample_size))

    # Return the results
    return results

mod1=train_predict(GaussianNB(), 26048, X_train, y_train,X_test, y_test)
mod1
mod2=train_predict(LogisticRegression(random_state = 0), 26048, X_train, y_train,X_test, y_test)
mod2
mod3=train_predict(RandomForestClassifier(random_state = 0), 26048, X_train, y_train,X_test, y_test)
mod3
print(mod1,mod2,mod3)

#k fold cross validation (evective way to evaluate model performance)
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=LogisticRegression(random_state = 0),X=X_train,y=y_train,cv=10)
accuracies.mean()
accuracies.std()
#model tuning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.grid_search import GridSearchCV

clf = LogisticRegression()
parameters = [{'C': [0.01, 0.1, 1, 10],"solver" : ['newton-cg','liblinear']}] 
'''error=C classification error+margin error
  large c=focus on classifing points
          may have a small margin
  small focus on large margin
  may make classification error
'''
grid_obj =  GridSearchCV(LogisticRegression(penalty='l2', random_state=0),parameters ,scoring='accuracy',cv=10,n_jobs=-1)
# Fit the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(X_train, y_train)
best_accuracy=grid_fit.best_score_
best_parameter=grid_fit.best_params_




