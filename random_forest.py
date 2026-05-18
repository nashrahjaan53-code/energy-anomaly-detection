from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
data = load_iris()

X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print(prediction)