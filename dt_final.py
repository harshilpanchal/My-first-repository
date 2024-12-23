import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

filename = "D:\Research\Maternal Health Risk Data Set.csv"

names = ['age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate', 'RiskLevel']
dataset = pd.read_csv(filename, names=names)

print(dataset.head())

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 6].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
model = classifier.fit(X_train, y_train)

text_representation = tree.export_text(classifier)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#print the accuracy
print("Accuracy = {0}".format(accuracy_score(y_test,y_pred)))



# Fit and show the visualizer
viz.fit(X, y)
viz.show()



fig = plt.figure(figsize=(5,5))
_ = tree.plot_tree(classifier, 
                   feature_names=names,  
                   filled=True)
