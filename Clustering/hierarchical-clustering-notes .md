# Hierarchical Clustering: Agglomerative and Divisive Approaches

## 1. Introduction to Hierarchical Clustering

Hierarchical Clustering is a type of cluster analysis that builds a hierarchy of clusters, creating a tree-like structure called a dendrogram. It differs from partitional methods like K-Means by not requiring a preset number of clusters.

### Two Primary Approaches
1. **Agglomerative (Bottom-Up)**
2. **Divisive (Top-Down)**

## 2. Agglomerative Hierarchical Clustering

### Core Concept
- Starts with each data point as a separate cluster
- Progressively merges closest clusters
- Continues until all points are in a single cluster

#### Key Steps:
1. Compute distance matrix between all points
2. Find two closest clusters
3. Merge the closest clusters
4. Repeat steps 2-3 until desired number of clusters

### Distance Metrics
1. **Euclidean Distance**
2. **Manhattan Distance**
3. **Cosine Similarity**

### Linkage Criteria
- **Single Linkage**: Minimum distance between clusters
- **Complete Linkage**: Maximum distance between clusters
- **Average Linkage**: Mean distance between clusters
- **Ward's Method**: Minimizes variance within clusters

## 3. Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

# Generate sample data
X, y = make_blobs(n_samples=50, centers=3, random_state=42)

# Perform Hierarchical Clustering
def plot_dendrogram(X, method='ward'):
    plt.figure(figsize=(10, 7))
    
    # Compute linkage matrix
    Z = linkage(X, method=method)
    
    # Plot dendrogram
    plt.title(f'Hierarchical Clustering Dendrogram ({method.capitalize()} Linkage)')
    dendrogram(
        Z,
        leaf_rotation=90,
        leaf_font_size=8
    )
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()

# Visualize different linkage methods
linkage_methods = ['single', 'complete', 'average', 'ward']
for method in linkage_methods:
    plot_dendrogram(X, method)
```

## 4. Divisive Hierarchical Clustering

### Core Concept
- Starts with all data points in one cluster
- Progressively splits clusters
- Continues until each point is in its own cluster

#### Challenges
- Computationally more expensive
- Less commonly implemented
- Requires defining splitting criteria

### Simplified Divisive Algorithm
1. Start with entire dataset as one cluster
2. Find most heterogeneous cluster
3. Split cluster into sub-clusters
4. Repeat until stopping condition met

## 5. Code Example of Divisive Clustering

```python
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt

def divisive_clustering(X, max_clusters=5):
    plt.figure(figsize=(15, 3))
    
    for n_clusters in range(1, max_clusters + 1):
        plt.subplot(1, max_clusters, n_clusters)
        
        # Simulate divisive clustering by reducing clusters
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, 
            linkage='ward'
        )
        clustering.fit(X)
        
        plt.scatter(X[:, 0], X[:, 1], c=clustering.labels_, cmap='viridis')
        plt.title(f'{n_clusters} Clusters')
    
    plt.tight_layout()
    plt.show()

divisive_clustering(X)
```

## 6. Dendrogram Interpretation

### Reading a Dendrogram
- **Vertical Axis**: Distance or Dissimilarity
- **Horizontal Axis**: Individual Data Points
- **Height of Merge**: Indicates cluster separation

### Key Insights from Dendrograms
- Cluster hierarchy
- Cluster distances
- Potential number of meaningful clusters

## 7. Pros and Cons

### Advantages
- No preset number of clusters
- Provides hierarchical view of data
- Works well with small to medium datasets
- Intuitive visualization

### Limitations
- Computationally expensive (O(n²) complexity)
- Sensitive to outliers
- Fixed clustering once performed
- Challenging with high-dimensional data

## 8. Best Practices
1. Normalize/scale features
2. Choose appropriate distance metric
3. Experiment with different linkage methods
4. Use dendrograms for cluster number selection
5. Validate results with domain knowledge

## Recommended Libraries
- scikit-learn
- SciPy
- NumPy
- Matplotlib
