
"""
Program summary: "This program is to implement Naive Bayes Classifier to classify the Iris data set. 
The data 
set is split into 2 folds, 50% of the data is used for training and the other 50% is used for testing.
The program will print out the accuracy score, the confusion matrix and the classification report."

"""





from pandas.core.arrays.arrow import array
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from pandas import read_csv
import numpy as np

#Loading the data set
url = "iris.csv" #data path
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] #data name
dataset = read_csv( url , names=names ) # Using pandas function to read the data


# Create an array to hold features and classes
array = dataset.values
X = array[:,0:4] # X contain the data of each feature
y = array[:,4] # Y is the name of the flower

#Split Data into 2 folds
X_fold1, X_fold2, y_fold1, y_fold2 = train_test_split(X, y, test_size = 0.50, random_state = 1)
# X_fold1 and y_fold1 are the first fold
# X_fold2 and y_fold2 are the second fold
# test_size = 0.50 means 50% of the data will be used for testing
# random_state = 1 means the data will be split randomly

#Create model for train and test
model = GaussianNB() # Using Gaussian Naive Bayes model

# Giving data to model
model.fit(X_fold1, y_fold1) # Training the first fold
predict1 = model.predict(X_fold2) # Then test the model with fold 2
model.fit(X_fold2, y_fold2)  # Training the second fold
predict2 = model.predict(X_fold1) # test the model with fold 1

# Concatenating the actual test results from the whole set
actual = np.concatenate([y_fold2, y_fold1]) 

# Concatenating the predicted results
predicted = np.concatenate([predict1, predict2])

# Print out the accuracy score, the confusion matrix and Classification report
print(f"The accuracy score is {accuracy_score(actual,predicted)}") 
print(f"The confusion matrix is \n {confusion_matrix(actual, predicted)}")
print("\n~~~~           THE CLASSIFICATION REPORT         ~~~~~")
print(classification_report(actual, predicted))








