import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
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
print("Effect of Tree Depth on Accuracy\n")
for depth in range(1, 11):
    model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=depth,
    random_state=42
    )
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    print(f"Depth = {depth}  Accuracy = {accuracy:.4f}")
print("\nComparison of Splitting Criteria\n")
for criterion in ["gini", "entropy"]:
    model = DecisionTreeClassifier(
    criterion=criterion,
    random_state=42
    )
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    print(f"{criterion} : {accuracy:.4f}")
# Final Tree
final_tree = DecisionTreeClassifier(
criterion="gini",
max_depth=3,
random_state=42
)
final_tree.fit(X_train, y_train)
plt.figure(figsize=(12,8))
plot_tree(
final_tree,
feature_names=iris.feature_names,
class_names=iris.target_names,
filled=True
)
plt.title("Decision Tree")
plt.show()