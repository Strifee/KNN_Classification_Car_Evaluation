import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
clss = le.fit_transform(list(data["class"]))
print(buying)

predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))
Y = list(clss)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors=9)

acc = model.fit(x_train, y_train)
print(acc)

predicted = model.predict(x_test)
names = ["unacc","acc","good","vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], " | Data: ", x_test[x], " | Actual: ", names[y_test[x]])