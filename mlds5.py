import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# Load Dataset
iris = load_iris()
X = iris.data
y = iris.target
# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# PCA with all components
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
# Explained Variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
print("Explained Variance Ratio")
for i, variance in enumerate(explained_variance):
    print(f"PC{i+1}: {variance:.4f}")
# Plot Cumulative Variance
plt.figure(figsize=(7,5))
plt.plot(
range(1, len(cumulative_variance)+1),
cumulative_variance,
marker='o'
)
plt.xlabel("Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid(True)
plt.show()
# PCA with 2 Components
pca2 = PCA(n_components=2)
X_new = pca2.fit_transform(X_scaled)
plt.figure(figsize=(7,5))
for label in np.unique(y):
    plt.scatter(
X_new[y==label,0],
X_new[y==label,1],
label=iris.target_names[label]
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA 2D Projection")
plt.legend()
plt.show()