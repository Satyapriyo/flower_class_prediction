import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("iris.csv")
print(df.head())


X = df[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]]
Y = df[["Class"]]

X_train , X_test, Y_train ,Y_test = train_test_split(X,Y,test_size=0.3,random_state=50)


sc =StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

clasifier = RandomForestClassifier()

clasifier.fit(X_train, Y_train)

pickle.dump(clasifier,open("model.pkl", "wb"))




