from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 44)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# print(y_pred, y_test)
accuracy = accuracy_score(y_pred, y_test)
print(accuracy)
cm = confusion_matrix(y_test, y_pred)
print(type(cm))
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    cmap="viridis",
    fmt="d",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
