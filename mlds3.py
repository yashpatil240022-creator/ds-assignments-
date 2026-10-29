import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Load Dataset
iris = load_iris()
X = iris.data
y = iris.target
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.25,
random_state=42,
stratify=y
)
# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_prediction = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_prediction)
print("Decision Tree Accuracy =", dt_accuracy)
# Random Forest
rf = RandomForestClassifier(
n_estimators=100,
random_state=42
)
rf.fit(X_train, y_train)
rf_prediction = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_prediction)
print("Random Forest Accuracy =", rf_accuracy)
print("\nEffect of Number of Trees")
for trees in [10, 50, 100, 200]:
    model = RandomForestClassifier(
    n_estimators=trees,
    random_state=42
    )
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    print("Trees =", trees,
    " Accuracy =", round(accuracy,4))
# Feature Importance
importance = rf.feature_importances_
indices = np.argsort(importance)[::-1]
plt.figure(figsize=(8,5))
plt.bar(
range(X.shape[1]),
importance[indices]
)
plt.xticks(
range(X.shape[1]),
np.array(iris.feature_names)[indices],
rotation=45
)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()