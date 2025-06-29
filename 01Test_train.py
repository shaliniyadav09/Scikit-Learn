from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm

iris = datasets.load_iris()
# split data into feature and labels

x= iris.data
y= iris.target
print(x.shape)


#hours of studying vs good/bad grades
#10 different stidents
#train a model with 8 students
#predict with the remaining 2
#level of accuruarcy of our model

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

model = svm.SVC()
model.fit(x_train, y_train)#train
print("d")
print(model)

predictions = model.predict(x_test)#predict