"""
Program summary: This program is to compare how well different ML classifiers perform on the Iris dataset.
The dataset is uses 2-fold cross validation to train and test the model. The program will print out the 
accuracy score and the confusion matrix of the ML models that need to be compare.
"""
from pandas.core.arrays.arrow import array
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from pandas import read_csv
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression


#Loading the data set
url = "iris.csv" #data path
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] #data name
dataset = read_csv( url , names=names )


# Create an array to hold features and classes
array = dataset.values
X = array[:,0:4] # X contain the data of each feature
y = array[:,4] # Y is the name of the flower


#Since the fifth column of the dataset is strings, change it to numeric values
class_mapping = {'Iris-setosa': 0, 'Iris-versicolor':1, 'Iris-virginica': 2}
y = np.array([class_mapping[name] for name in y])


#Split Data into 2 folds
X_fold1, X_fold2, y_fold1, y_fold2 = train_test_split(X, y, test_size = 0.50, random_state = 1)

# First part of the requirement ML model
LR = [PolynomialFeatures(degree=1), PolynomialFeatures(degree=2), PolynomialFeatures(degree=3)]
LRname = ["Linear regression", "Polynomial of degree 2 regression", "Polynomial of degree 3 regression"]
n = 0
for Poly_R in LR:
  model = LinearRegression() 

  #Create polynomial Features for folds
  X_Poly1 = Poly_R.fit_transform(X_fold1)
  X_Poly2 = Poly_R.fit_transform(X_fold2)

  #Giving data to model for training and testing
  
  model.fit(X_Poly1, y_fold1) # Training the first fold
  pred1 = model.predict(X_Poly2).round() # Test the model with fold 2
  # Since the predict value is float, we need to change it to integer
  # Regressor model will predict the value between -1 and 3
  pred1 = np.where(pred1 >= 3.0, 2.0, pred1) # If the value is greater than 3, change it to 2
  pred1 = np.where(pred1 <= -1.0, 0.0, pred1) # If the value is less than -1, change it to 0

  model.fit(X_Poly2, y_fold2) # Training the second fold
  pred2 = model.predict(X_Poly1).round() # Test the model with fold 1
  pred2 = np.where(pred2 >= 3.0, 2.0, pred2) # If the value is greater than 3, change it to 2
  pred2 = np.where(pred2 <= -1.0, 0.0, pred2) # If the value is less than -1, change it to 0

   # Concatenating the actual test results from the whole set
  actual = np.concatenate([y_fold2, y_fold1])
  # Concatenating the predicted results
  predicted = np.concatenate([pred1, pred2])

  # Print out the accuracy score, the confusion matrix and Classification report
  print(f"~~~~~~~ {LRname[n]} ~~~~~~~")
  print(f"The accuracy score is {accuracy_score(actual,predicted)}")
  print(f"The confusion matrix is \n {confusion_matrix(actual, predicted)}\n")
  n += 1 


# Second part of the requirement ML model
M = [GaussianNB(), KNeighborsClassifier(n_neighbors=10), LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis()]
Mname = ["Naive Baysian", "kNN", "LDA", "QDA"]
i = 0
# For other ML model that not Linear Regression
for model in M:
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
  print(f"~~~~~~~ {Mname[i]} ~~~~~~~")
  print(f"The accuracy score is {accuracy_score(actual,predicted)}")
  print(f"The confusion matrix is \n {confusion_matrix(actual, predicted)}\n")
  i += 1

