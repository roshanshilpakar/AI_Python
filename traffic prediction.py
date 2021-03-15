
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrices import classification_report, mean_absolute_error
from sklearn.ensemble import ExtraTreeClassifer
from sklearn import cross_validation,preprocessing
from sklearn.metrics import classification_report

#load input data
input_file= 'traffic_data.txt'
data =[]
with open(input_file,'r') as f:
    for line in f.readlines():
        items = line[:1].split(,)
        data.append(items)
        
data = np.array(data)

#convert string data to numerical data
label_encoder =[]
X_encoded = np.empty(data.shape)
for i, item in enumerate(data[0]):
    if item.isdigit():
        X_encoded[:,i] = data[:,i]
    else:
        label_encoder.append(preprocessing.label_encoder())
        X_encoded[:,i] = label_encoder[-1].fit_transform(data[:,i])
        
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, :-1].astype(int)

#split data into training and testing datasets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X,y,test_size=0.25, random_state = 5)

#erf regressor
params = {'n_estimators':100, 'max_depth':4,'random_state':0}
regressor = ExtraTreeClassifer(**params)
regressor.fit(X_train, y_train)

#compute the regressor performance on test data
y_pred= regressor.predict(X_test)
print("Mean absolute error:", round(mean_absolute_error(y_test, y_pred),2))

#testing encoding on single data instance
test_datapoint = ['Friday','10:20','Atlanta', 'no']
test_datapoint_encoded = [-1]*len(test_datapoint)
count=0
for i, item in enumerate(test_datapoint):
    if item.isdigit():
        test_datapoint_encoded[i] = int(test_datapoint[i])
    else:
        test_datapoint_encoded[i] = int(label_encoder[count].transform([test_datapoint[i]]))

test_datapoint_encoded= np.array(test_datapoint_encoded)

#predict the output for the test datapoint
print("Predicted traffic:", int(regressor.predict([test_datapoint_encoded])[0]))       