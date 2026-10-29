from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
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
kernels = ['linear', 'poly', 'rbf']
results = {}
for kernel in kernels:
    svm = SVC(
    kernel=kernel,
    C=1.0,
    gamma='scale',
    random_state=42
    )
    svm.fit(X_train, y_train)
    prediction = svm.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    results[kernel] = accuracy
    print("\nKernel :", kernel)
    print("Accuracy :", accuracy)
    print(classification_report(
    y_test,
    prediction,
    target_names=iris.target_names
    ))
print("\nSummary")
for kernel, accuracy in results.items():
    print(kernel, ":", accuracy)
best_kernel = max(results, key=results.get)
print("\nBest Kernel =", best_kernel)
