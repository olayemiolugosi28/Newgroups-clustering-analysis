# Newsgroups Clustering Analysis

## Project Overview
This project performs text clustering analysis on newsgroups data using machine learning techniques.

## Objectives
- Apply K-means and hierarchical clustering algorithms
- Determine optimal number of clusters using the elbow method
- Visualize results using Principal Component Analysis (PCA)

## Dataset
- **Source**: Sample newsgroups dataset
- **Size**: 45 documents
- **Categories**: 
  - sci.space (space exploration)
  - rec.autos (automotive)
  - talk.politics.guns (politics)

## Technologies Used
- Python 3.x
- scikit-learn (clustering algorithms, TF-IDF)
- matplotlib & seaborn (visualization)
- pandas & numpy (data manipulation)
- scipy (hierarchical clustering)

## Key Results
- **Optimal Clusters**: k=3 (determined by elbow method)
- **K-Means Silhouette Score**: 0.0094
- **Hierarchical Silhouette Score**: 0.0165
- **PCA Variance Explained**: 7.52% (2 components)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/olayemiolugosi28/newsgroups-clustering-analysis.git
cd newsgroups-clustering-analysis
```

2. Install required packages:
```bash
pip install -r requirements_clustering.txt
```

## Usage

Run the analysis:
```bash
python newsgroups_clustering.py
```

This will generate:
- Elbow method plots
- Clustering visualizations
- PCA plots (2D and 3D)
- Dendrogram for hierarchical clustering

## Visualizations

### Elbow Method
![Elbow Method](visualizations/elbow_method.png)

### PCA Clustering Visualization
![PCA Visualization](visualizations/pca_clustering_visualization.png)

### Dendrogram
![Dendrogram](visualizations/dendrogram.png)

## Files Description

| File | Description |
|------|-------------|
| `newsgroups_clustering.py` | Main analysis script |
| `Clustering_Analysis_Report.md` | Detailed analysis report |
| `requirements_clustering.txt` | Python dependencies |
| `visualizations/` | Generated plots and charts |

## Analysis Methods

### 1. Text Preprocessing
- TF-IDF vectorization
- Feature extraction (200 terms)
- Stop word removal

### 2. Clustering Algorithms
- **K-Means**: Partitional clustering with k=3
- **Hierarchical**: Agglomerative clustering with Ward linkage

### 3. Evaluation Metrics
- Elbow method for optimal k
- Silhouette score
- Cross-tabulation with ground truth

### 4. Dimensionality Reduction
- PCA for visualization
- 2D and 3D projections

## Key Findings

1. Both clustering algorithms identified 3 distinct clusters
2. Hierarchical clustering showed better performance (higher silhouette score)
3. Document categories showed vocabulary overlap, making perfect separation challenging
4. PCA effectively visualized cluster structure despite high dimensionality

## Detailed Report

For a comprehensive analysis, see [Clustering_Analysis_Report.md](Clustering_Analysis_Report.md)

## Author

Olayemi Olugosi

## Contact

- GitHub: https://github.com/olayemiolugosi28
- Email: olayemiolugosi@gmail.com
- LinkedIn: www.linkedin.com/in/olayemi-olugosi

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- scikit-learn documentation
- Python data science community
  
---

If you found this project helpful, please give it a star!
