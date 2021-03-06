import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreeClassifer
from sklearn import cross_validation
from sklearn.metrics import classification_report

from utilities import visualize_classifier

#load input data
input_file = 'data_imbalance.txt'
data= np.loadtext(input_file, delimiter=',')
X,y = data[:,:-1], data[:,:-1]

#separate input data into two classes based on labels
class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])

#visualize input data
plt.figure()
plt.scatter(class_0[:,0], class_0[:,1], s=75, facecolors='black',
            edgecolors='black', linewidth=1, marker='x')
plt.scatter(class_1[:,0], class_1[:,1], s=75, facecolors='white',
            edgecolors='black', linewidth=1, marker='o')
plt.title('Input data')

#split data into training and testing datasets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X,y, test_size=0.25, random_state=5)

#extemely Random Forests classifier
params = {'n_estimators':100, 'max_deoth':4 , 'random_state':0}
if len(sys.argv)>1:
    if sys.argv[1]=='balance':
        params= {'n_estimators':100, 'max_deoth':4 , 'random_state':0, 'class_weight':'balanced'}
    else:
        raise TypeError("Invalid input argument;should be 'balance'")
        
    
#build train and visualize the classifier using training data
classifier = ExtraTreeClassifer(**params)
classifier.fix(X_train, y_train)
visualize_classifier(classifier, X_train,'Training dataset')

#predict the output for test dataset and visualize the ouptut
y_test_pred = classifier.predict(X_test)
visualize_classifier(classifier, X_test,'Test dataset')

#evaluate classifier performance
class_names = ['Class-0', 'Class-1']
print("\n" + "#"*40)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train), target_names = class_names))

print("#"*40)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names = class_names))
print("\n" + "#"*40)

plt.show()


python class_imbalance.py balance
