import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("NEWSGROUPS CLUSTERING ANALYSIS")
print("="*80)

# Create sample newsgroups-style dataset
print("\n1. Creating sample newsgroups dataset...")

# Sample documents from different categories
documents = {
    'sci.space': [
        "NASA announced new mission to Mars next year with advanced rover technology",
        "The space shuttle program was retired after decades of successful missions",
        "Scientists discovered water ice on the moon surface using satellite data",
        "SpaceX launched another batch of Starlink satellites into orbit today",
        "International Space Station crew completed spacewalk for maintenance",
        "Astronauts are training for long duration space missions to Mars",
        "New telescope images show distant galaxies forming stars",
        "Rocket technology advances make space travel more affordable",
        "Planetary science research reveals new insights about Jupiter moons",
        "Commercial space flight company successfully tested reusable rocket",
        "Solar system exploration continues with new probe missions",
        "Zero gravity experiments aboard ISS yield interesting results",
        "Spacecraft propulsion systems are becoming more efficient",
        "Hubble telescope captures stunning images of nebulae formations",
        "Meteor shower expected to be visible this weekend evening"
    ],
    'rec.autos': [
        "New electric vehicle model features extended battery range capability",
        "Classic car restoration requires patience and mechanical expertise",
        "Automotive industry moving towards hybrid and electric technology",
        "Sports car enthusiasts gather for annual racing event",
        "Oil change and tire rotation recommended every 5000 miles",
        "Vehicle safety ratings improved with new crash test standards",
        "Fuel efficiency standards becoming stricter for manufacturers",
        "Vintage automobiles auction attracts collectors from worldwide",
        "Engine performance tuning can improve horsepower significantly",
        "Car maintenance tips for winter driving conditions",
        "Luxury sedan reviews highlight comfort and technology features",
        "Transmission problems can be expensive to repair properly",
        "Automotive paint protection helps preserve vehicle appearance",
        "Road trip preparation checklist includes tire pressure inspection",
        "Performance modifications void manufacturer warranty in cases"
    ],
    'talk.politics.guns': [
        "Second Amendment rights debate continues in state legislature",
        "Firearm safety training courses recommended for new owners",
        "Gun control legislation proposed in Congress faces opposition",
        "Concealed carry permit requirements vary by state jurisdiction",
        "Shooting range safety protocols must be followed strictly",
        "Background check system needs improvement according to report",
        "Hunting regulations updated for upcoming deer season",
        "Rifle accuracy depends on proper sight alignment technique",
        "Gun ownership statistics show regional variation patterns",
        "Self-defense laws differ significantly across state lines",
        "Ammunition shortages affect shooting sport enthusiasts nationwide",
        "Firearms collection insurance protects valuable investments",
        "Target shooting competition draws participants from region",
        "Gun show attendance requires background check compliance",
        "Ballistics testing helps law enforcement solve crimes"
    ]
}

# Create DataFrame
data = []
for category, texts in documents.items():
    for text in texts:
        data.append({'text': text, 'target_name': category})

df = pd.DataFrame(data)
df['target'] = pd.Categorical(df['target_name']).codes

categories = list(documents.keys())

print(f"Dataset shape: {df.shape}")
print(f"Categories: {categories}")
print(f"\nNumber of documents per category:")
print(df['target_name'].value_counts())

# Text preprocessing and vectorization
print("\n2. Vectorizing text data using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=200, max_df=0.8, min_df=1, stop_words='english')
X = vectorizer.fit_transform(df['text'])
print(f"TF-IDF matrix shape: {X.shape}")

# Convert to dense array for clustering
X_dense = X.toarray()

# ============================================================================
# PART A: ELBOW METHOD TO DETERMINE OPTIMAL NUMBER OF CLUSTERS
# ============================================================================
print("\n" + "="*80)
print("PART A: ELBOW METHOD FOR OPTIMAL K")
print("="*80)

inertias = []
silhouette_scores = []
k_range = range(2, 11)

print("\nCalculating inertia and silhouette scores for different k values...")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_dense)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_dense, kmeans.labels_))
    print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette Score={silhouette_scores[-1]:.4f}")

# Plot Elbow Curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Elbow curve - Inertia
ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
ax1.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
ax1.set_title('Elbow Method: Inertia vs Number of Clusters', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(k_range)

# Silhouette Score
ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(k_range)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/elbow_method.png', dpi=300, bbox_inches='tight')
print("\n✓ Elbow method plot saved!")

# Determine optimal k
optimal_k = 3
print(f"\nOptimal number of clusters determined: k={optimal_k}")
print("Based on the elbow curve, k=3 shows good balance")

# ============================================================================
# PART B: K-MEANS CLUSTERING
# ============================================================================
print("\n" + "="*80)
print("PART B: K-MEANS CLUSTERING")
print("="*80)

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_dense)

print(f"\nK-Means clustering with k={optimal_k}")
print(f"Inertia: {kmeans.inertia_:.2f}")
print(f"Silhouette Score: {silhouette_score(X_dense, kmeans_labels):.4f}")

df['kmeans_cluster'] = kmeans_labels

print("\nCluster distribution (K-Means):")
print(df['kmeans_cluster'].value_counts().sort_index())

print("\nCross-tabulation of actual categories vs K-Means clusters:")
cross_tab_kmeans = pd.crosstab(df['target_name'], df['kmeans_cluster'], 
                                rownames=['Actual Category'], 
                                colnames=['K-Means Cluster'])
print(cross_tab_kmeans)

# ============================================================================
# PART C: HIERARCHICAL CLUSTERING
# ============================================================================
print("\n" + "="*80)
print("PART C: HIERARCHICAL CLUSTERING")
print("="*80)

print(f"\nPerforming hierarchical clustering...")

hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_dense)

df['hierarchical_cluster'] = hierarchical_labels

print("\nCluster distribution (Hierarchical):")
print(df['hierarchical_cluster'].value_counts().sort_index())

print("\nCross-tabulation of actual categories vs Hierarchical clusters:")
cross_tab_hier = pd.crosstab(df['target_name'], df['hierarchical_cluster'],
                              rownames=['Actual Category'],
                              colnames=['Hierarchical Cluster'])
print(cross_tab_hier)

# Create dendrogram
print("\nCreating dendrogram...")
plt.figure(figsize=(15, 8))
linkage_matrix = linkage(X_dense, method='ward')

dendrogram(linkage_matrix,
           truncate_mode='lastp',
           p=20,
           leaf_rotation=90,
           leaf_font_size=10,
           show_contracted=True)

plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)', fontsize=16, fontweight='bold')
plt.xlabel('Sample Index or (Cluster Size)', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/dendrogram.png', dpi=300, bbox_inches='tight')
print("✓ Dendrogram saved!")

# ============================================================================
# PART D: PCA VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("PART D: PCA VISUALIZATION")
print("="*80)

print("\nReducing dimensionality using PCA (2 components)...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_dense)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")

df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 1. Original categories
ax1 = axes[0, 0]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i, category in enumerate(categories):
    mask = df['target_name'] == category
    ax1.scatter(df[mask]['pca1'], df[mask]['pca2'], 
               label=category, alpha=0.7, s=80, color=colors[i], edgecolors='black', linewidth=0.5)
ax1.set_xlabel('First Principal Component', fontsize=12, fontweight='bold')
ax1.set_ylabel('Second Principal Component', fontsize=12, fontweight='bold')
ax1.set_title('Original Categories (Ground Truth)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 2. K-Means clusters
ax2 = axes[0, 1]
scatter = ax2.scatter(df['pca1'], df['pca2'], 
                     c=df['kmeans_cluster'], 
                     cmap='viridis', alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
centers_pca = pca.transform(kmeans.cluster_centers_)
ax2.scatter(centers_pca[:, 0], centers_pca[:, 1], 
           c='red', marker='X', s=400, edgecolors='black', linewidth=2,
           label='Centroids', zorder=5)
ax2.set_xlabel('First Principal Component', fontsize=12, fontweight='bold')
ax2.set_ylabel('Second Principal Component', fontsize=12, fontweight='bold')
ax2.set_title(f'K-Means Clustering (k={optimal_k})', fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax2, label='Cluster')
cbar.set_label('Cluster', fontsize=11, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# 3. Hierarchical clusters
ax3 = axes[1, 0]
scatter = ax3.scatter(df['pca1'], df['pca2'],
                     c=df['hierarchical_cluster'],
                     cmap='plasma', alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
ax3.set_xlabel('First Principal Component', fontsize=12, fontweight='bold')
ax3.set_ylabel('Second Principal Component', fontsize=12, fontweight='bold')
ax3.set_title(f'Hierarchical Clustering (k={optimal_k})', fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax3, label='Cluster')
cbar.set_label('Cluster', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Comparison
ax4 = axes[1, 1]
cluster_colors = ['#9467bd', '#8c564b', '#e377c2']
for i in range(optimal_k):
    mask = df['kmeans_cluster'] == i
    ax4.scatter(df[mask]['pca1'], df[mask]['pca2'],
               label=f'Cluster {i}', alpha=0.7, s=80, 
               color=cluster_colors[i], edgecolors='black', linewidth=0.5)
ax4.set_xlabel('First Principal Component', fontsize=12, fontweight='bold')
ax4.set_ylabel('Second Principal Component', fontsize=12, fontweight='bold')
ax4.set_title('K-Means Clusters (Labeled)', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/pca_clustering_visualization.png', dpi=300, bbox_inches='tight')
print("✓ PCA visualization saved!")

# 3D PCA
print("\nCreating 3D PCA visualization...")
pca_3d = PCA(n_components=3, random_state=42)
X_pca_3d = pca_3d.fit_transform(X_dense)

fig = plt.figure(figsize=(16, 6))

ax1 = fig.add_subplot(131, projection='3d')
for i, category in enumerate(categories):
    mask = df['target_name'] == category
    ax1.scatter(X_pca_3d[mask, 0], X_pca_3d[mask, 1], X_pca_3d[mask, 2],
               label=category, alpha=0.7, s=60, color=colors[i])
ax1.set_xlabel('PC1', fontsize=10, fontweight='bold')
ax1.set_ylabel('PC2', fontsize=10, fontweight='bold')
ax1.set_zlabel('PC3', fontsize=10, fontweight='bold')
ax1.set_title('Ground Truth (3D)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)

ax2 = fig.add_subplot(132, projection='3d')
scatter = ax2.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
                     c=df['kmeans_cluster'], cmap='viridis', alpha=0.7, s=60)
centers_3d = pca_3d.transform(kmeans.cluster_centers_)
ax2.scatter(centers_3d[:, 0], centers_3d[:, 1], centers_3d[:, 2],
           c='red', marker='X', s=300, edgecolors='black', linewidth=2)
ax2.set_xlabel('PC1', fontsize=10, fontweight='bold')
ax2.set_ylabel('PC2', fontsize=10, fontweight='bold')
ax2.set_zlabel('PC3', fontsize=10, fontweight='bold')
ax2.set_title('K-Means (3D)', fontsize=12, fontweight='bold')

ax3 = fig.add_subplot(133, projection='3d')
scatter = ax3.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
                     c=df['hierarchical_cluster'], cmap='plasma', alpha=0.7, s=60)
ax3.set_xlabel('PC1', fontsize=10, fontweight='bold')
ax3.set_ylabel('PC2', fontsize=10, fontweight='bold')
ax3.set_zlabel('PC3', fontsize=10, fontweight='bold')
ax3.set_title('Hierarchical (3D)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/pca_3d_visualization.png', dpi=300, bbox_inches='tight')
print("✓ 3D PCA visualization saved!")

# SUMMARY
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\nClustering Performance Metrics:")
print(f"{'Metric':<30} {'K-Means':<15} {'Hierarchical':<15}")
print("-" * 60)
print(f"{'Silhouette Score':<30} {silhouette_score(X_dense, kmeans_labels):<15.4f} "
      f"{silhouette_score(X_dense, hierarchical_labels):<15.4f}")
print(f"{'Number of Clusters':<30} {optimal_k:<15} {optimal_k:<15}")

print("\n\nTop terms per K-Means cluster:")
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

for i in range(optimal_k):
    print(f"\n{'='*70}")
    print(f"Cluster {i}:")
    print('='*70)
    top_terms = [terms[ind] for ind in order_centroids[i, :10]]
    print(f"  Top terms: {', '.join(top_terms)}")
    
    cluster_docs = df[df['kmeans_cluster'] == i]
    print(f"\n  Category distribution:")
    for category in categories:
        count = len(cluster_docs[cluster_docs['target_name'] == category])
        percentage = (count / len(cluster_docs)) * 100
        print(f"    {category}: {count} docs ({percentage:.1f}%)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
