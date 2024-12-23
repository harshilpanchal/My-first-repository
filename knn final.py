#Importing the required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filename = "E:\SICSR\Sem5\Data Science\LoanApproval.csv"

names = ['Name', 'Age', 'College Tier', 'CGPA', 'Annual Income', 'Laon Amount', 'Course Fees', 'Stem Course','Laon Approval']
dataset = pd.read_csv(filename, names=names) # reading the provided dataset

print(dataset.head())

#dividing the data into dependent and independent variable 
X = dataset.iloc[:,1:8].values
y = dataset.iloc[:, 8].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  #Spliting the training and testing data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=20)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))




no_neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(no_neighbors))
test_accuracy = np.empty(len(no_neighbors))


# data for plot
for i, k in enumerate(no_neighbors):
    # We instantiate the classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    
    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
       

    # Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)
    
    
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(no_neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(no_neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show() 
    
#import the validation packages
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#generate the confusion matrixs
cm=confusion_matrix(y_test,y_pred)
print(cm)

#print the accuracy
print("Accuracy = {0}".format(accuracy_score(y_test,y_pred)))



