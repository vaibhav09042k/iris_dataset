import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("Iris (1).csv")

x = df.iloc[:, 1:5]
y = df.iloc[:, 5]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

###################### Scaling data ######################
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

###################### Applying Knn Algorithm ######################

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(x_train_std, y_train)
#pred_y = knn.predict(x_test_std)
#print(pred_y)
#print(y_test)
knn_train_score = knn.score(x_train_std, y_train)
knn_test_score = knn.score(x_test_std, y_test)
#print(f"The accuracy score of the Knn classifier training data - {knn_train_score}")
#print(f"The accuracy score of the Knn classifier test data - {knn_test_score}")

###################### Applying Decision Tree Algorithm ######################

from sklearn import tree

dct = tree.DecisionTreeClassifier(criterion='gini')
dct.fit(x_train_std, y_train)

dct_train_score = dct.score(x_train_std, y_train)
dct_test_score = dct.score(x_test_std, y_test)
print(f"The accuracy of the Decision Tree classifier on training data is {dct_train_score}")
print(f"The accuracy of the Decision Tree classifier on test data is {dct_test_score}")