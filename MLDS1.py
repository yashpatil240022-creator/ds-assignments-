import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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
# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
k_values = range(1, 16)
accuracies = []
for k in k_values:
    model = KNeighborsClassifier(
        n_neighbors=k
    )
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    accuracies.append(accuracy)
    print(f"K = {k:2d}  Accuracy = {accuracy:.4f}")
best_k = k_values[np.argmax(accuracies)]
print("\nBest Value of K =", best_k)
print("Best Accuracy =", max(accuracies))
plt.figure(figsize=(7,5))
plt.plot(
    list(k_values),
    accuracies,
    marker='o'
)
plt.xlabel("Value of K")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs K")
plt.grid(True)
plt.show()
