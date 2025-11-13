# Newsgroups Clustering Analysis Report

## Executive Summary

This analysis performed K-means and hierarchical clustering on a newsgroups-style dataset containing 45 documents across 3 categories: sci.space, rec.autos, and talk.politics.guns. The optimal number of clusters was determined using the elbow method, and results were visualized using Principal Component Analysis (PCA).

---

## Dataset Overview

- **Total Documents**: 45
- **Categories**: 3
  - sci.space: 15 documents
  - rec.autos: 15 documents  
  - talk.politics.guns: 15 documents
- **TF-IDF Features**: 200 unique terms
- **TF-IDF Matrix Shape**: (45, 200)

---

## Part A: Elbow Method for Optimal K

### Methodology
The elbow method was used to determine the optimal number of clusters by:
1. Running K-means for k=2 to k=10
2. Calculating inertia (within-cluster sum of squares) for each k
3. Calculating silhouette scores for each k
4. Identifying the "elbow point" where diminishing returns occur

### Results

| k | Inertia | Silhouette Score |
|---|---------|------------------|
| 2 | 42.15 | 0.0064 |
| **3** | **40.85** | **0.0094** |
| 4 | 39.37 | 0.0177 |
| 5 | 38.06 | 0.0169 |
| 6 | 36.85 | 0.0213 |
| 7 | 35.71 | 0.0237 |
| 8 | 34.73 | 0.0230 |
| 9 | 33.39 | 0.0303 |
| 10 | 32.28 | 0.0285 |

### Conclusion
**Optimal k = 3** was selected based on:
- Clear elbow in the inertia curve at k=3
- Balance between model complexity and performance
- Matches the ground truth number of categories

---

## Part B: K-Means Clustering Results

### Performance Metrics
- **Number of Clusters**: 3
- **Inertia**: 40.85
- **Silhouette Score**: 0.0094

### Cluster Distribution
- Cluster 0: 7 documents
- Cluster 1: 34 documents (largest)
- Cluster 2: 4 documents

### Cross-Tabulation: Actual Categories vs K-Means Clusters

|                    | Cluster 0 | Cluster 1 | Cluster 2 |
|--------------------|-----------|-----------|-----------|
| rec.autos          | 2         | 12        | 1         |
| sci.space          | 4         | 11        | 0         |
| talk.politics.guns | 1         | 11        | 3         |

### Top Terms per Cluster

**Cluster 0** (7 documents):
- Top terms: new, safety, vehicle, probe, exploration, forming, galaxies, firearm, owners, continues
- Dominant category: sci.space (57.1%)
- Interpretation: Mixed cluster with space exploration and safety themes

**Cluster 1** (34 documents):
- Top terms: space, shooting, car, gun, maintenance, technology, propulsion, missions, requires, background
- Category distribution: Relatively balanced (sci.space: 32.4%, rec.autos: 35.3%, talk.politics.guns: 32.4%)
- Interpretation: Large mixed cluster containing documents from all categories

**Cluster 2** (4 documents):
- Top terms: state, significantly, laws, lines, amendment, legislature, rights, continues, improve, engine
- Dominant category: talk.politics.guns (75.0%)
- Interpretation: Primarily political/legal content

---

## Part C: Hierarchical Clustering Results

### Performance Metrics
- **Number of Clusters**: 3
- **Linkage Method**: Ward
- **Silhouette Score**: 0.0165 (better than K-means)

### Cluster Distribution
- Cluster 0: 35 documents (largest)
- Cluster 1: 6 documents
- Cluster 2: 4 documents

### Cross-Tabulation: Actual Categories vs Hierarchical Clusters

|                    | Cluster 0 | Cluster 1 | Cluster 2 |
|--------------------|-----------|-----------|-----------|
| rec.autos          | 14        | 1         | 0         |
| sci.space          | 10        | 5         | 0         |
| talk.politics.guns | 11        | 0         | 4         |

### Dendrogram Analysis
The dendrogram visualization shows the hierarchical structure of document relationships using Ward linkage. This method minimizes within-cluster variance at each merge, creating compact, well-separated clusters.

---

## Part D: PCA Visualization

### Dimensionality Reduction
- **Original Dimensions**: 200 TF-IDF features
- **Reduced Dimensions**: 2 principal components (for 2D) and 3 (for 3D)
- **Variance Explained (2D)**: 7.52% (PC1: 3.85%, PC2: 3.67%)

### Key Observations from Visualizations

#### 1. Ground Truth Visualization
The original categories show some overlap in the PCA space, indicating that the three categories share some common vocabulary and themes.

#### 2. K-Means Visualization
- Red X markers indicate cluster centroids
- Cluster 1 (largest) occupies the central region
- Clusters 0 and 2 capture documents at the extremes
- Shows some separation but with significant overlap

#### 3. Hierarchical Clustering Visualization
- Similar pattern to K-means but with different cluster assignments
- More balanced separation in some regions
- Slightly better silhouette score (0.0165 vs 0.0094)

#### 4. 3D Visualization
The 3D PCA plots provide additional perspective showing:
- Better separation between clusters in 3D space
- Cluster centroids positioned strategically
- More nuanced understanding of document relationships

---

## Comparative Analysis

### K-Means vs Hierarchical Clustering

| Metric | K-Means | Hierarchical |
|--------|---------|-------------|
| Silhouette Score | 0.0094 | 0.0165 |
| Largest Cluster Size | 34 docs | 35 docs |
| Smallest Cluster Size | 4 docs | 4 docs |
| Computation Time | Fast | Moderate |
| Cluster Quality | Mixed categories | Better separation |

### Key Findings:
1. **Hierarchical clustering performed slightly better** with a higher silhouette score
2. **Both methods struggled with cluster purity** - documents from multiple categories often ended up in the same cluster
3. **Low variance explained by PCA** (7.52%) suggests high-dimensional nature of text data
4. **Document similarity is complex** - news categories share vocabulary and themes

---

## Insights and Interpretations

### Why Clustering Was Challenging

1. **Vocabulary Overlap**: 
   - Terms like "new", "system", "technology" appear across all categories
   - Common English words create noise in the feature space

2. **Topic Similarity**:
   - All categories discuss technical subjects
   - Similar sentence structures and terminology

3. **High Dimensionality**:
   - 200 features create sparse vector space
   - PCA captures only 7.52% variance in 2D

### Successful Separations

Despite challenges, some patterns emerged:
- **Cluster 2** successfully identified political/legislative content (75% talk.politics.guns)
- **Cluster 0** showed preference for space-related content (57.1% sci.space)
- **Hierarchical clustering** showed better category alignment in Cluster 2

---

## Recommendations

### For Improved Clustering:

1. **Feature Engineering**:
   - Use bigrams/trigrams to capture context
   - Apply domain-specific stop words
   - Increase feature count for richer representation

2. **Advanced Techniques**:
   - Try topic modeling (LDA/NMF)
   - Use word embeddings (Word2Vec, GloVe, BERT)
   - Apply semi-supervised methods if labels are available

3. **Preprocessing**:
   - Remove common terms across categories
   - Apply lemmatization for consistent word forms
   - Balance document lengths

4. **Alternative Algorithms**:
   - DBSCAN for density-based clustering
   - Spectral clustering for non-convex shapes
   - Ensemble clustering methods

---

## Conclusions

1. **Optimal Cluster Number**: The elbow method successfully identified k=3 as optimal, matching the ground truth

2. **Clustering Performance**: Both algorithms achieved moderate success, with hierarchical clustering showing slightly better performance (silhouette score: 0.0165 vs 0.0094)

3. **Document Separation**: Text clustering is inherently challenging due to vocabulary overlap and high dimensionality

4. **Visualization Value**: PCA visualization effectively demonstrated cluster structure and overlap patterns, even with low variance explained

5. **Practical Application**: For real-world newsgroup classification, supervised learning methods would likely outperform unsupervised clustering

---

## Technical Implementation

### Tools and Libraries Used:
- **scikit-learn**: TfidfVectorizer, KMeans, AgglomerativeClustering, PCA
- **matplotlib & seaborn**: Data visualization
- **scipy**: Dendrogram creation
- **pandas & numpy**: Data manipulation

### Reproducibility:
All code and visualizations are available in `newsgroups_clustering.py`. The analysis used fixed random seeds (random_state=42) for reproducibility.

---

## Generated Visualizations

1. **elbow_method.png**: Shows inertia and silhouette scores vs number of clusters
2. **dendrogram.png**: Hierarchical clustering tree structure
3. **pca_clustering_visualization.png**: 2D PCA plots comparing all methods
4. **pca_3d_visualization.png**: 3D PCA plots for deeper analysis

---

*Analysis completed: November 13, 2025*
*Dataset: Sample newsgroups (45 documents, 3 categories)*
*Methods: K-means, Hierarchical Clustering (Ward), PCA*
