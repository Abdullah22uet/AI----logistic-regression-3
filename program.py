import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("IRIS.csv")
# drop null values
data.dropna(inplace=True)
# drop duplicate values
data.drop_duplicates(inplace=True)

# encoding target variable into numeric values
data["species"].replace({"Iris-setosa":"1","Iris-versicolor":"2","Iris-virgincia":"3"} , inplace=True)

x = data.drop("species",axis=1)
y = data["species"]

# now data is ready for model training ===================================================
# splitting data into four parts
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=.20 , random_state=0)

# traning the model
model = LogisticRegression()
model.fit(x_train,y_train)

# predicting results from the model
prediction = model.predict(x_test)

# finding the accuracy score of the model
score = model.score(x_test , y_test)
print("Accuracy :" , score*100)