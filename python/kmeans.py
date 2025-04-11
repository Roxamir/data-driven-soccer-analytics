import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Load and prepare the data
df = pd.read_csv('Datasets/playerdata/api_cleaned.csv')

# Select three key features for 3D visualization
selected_features = ['goals', 'rating', 'minutes_played']
X = df[selected_features].copy()
print("")
print(X)

# Handle missing values
X = X.fillna(X.mean())

# Scale the features for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create 3D visualization
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X['goals'], X['rating'], X['minutes_played'],
                    c='blue', alpha=0.05)

ax.set_xlabel('Goals')
ax.set_ylabel('Rating')
ax.set_zlabel('Minutes Played')
ax.set_title('3D Visualization of Player Performance Data')
ax.view_init(azim=-75)

plt.show()

# Function to perform and visualize k-means clustering
def perform_kmeans(k, X_scaled, X, selected_features):
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Get cluster centers
    centers_scaled = kmeans.cluster_centers_
    centers_original = scaler.inverse_transform(centers_scaled)
    
    # Create 3D visualization with clusters
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot data points with lower z-order
    scatter = ax.scatter(X['goals'], X['rating'], X['minutes_played'],
                        c=cluster_labels, cmap='tab10', alpha=0.1,
                        zorder=1)
    
    # Plot all centroids separately to ensure they're on top
    for i in range(len(centers_original)):
        ax.scatter(centers_original[i, 0], centers_original[i, 1], centers_original[i, 2],
                  c='black', marker='o', s=300, linewidth=2,
                  edgecolor='white', zorder=2, alpha=1.0)
    
    # Add a single point for the legend
    ax.scatter([], [], [], c='black', marker='o', s=300, linewidth=2,
              edgecolor='white', label='Centroids', alpha=1.0)
    
    ax.set_xlabel('Goals')
    ax.set_ylabel('Rating')
    ax.set_zlabel('Minutes Played')
    ax.set_title(f'K-means Clustering (k={k})')
    plt.legend()
    
    ax.view_init(azim=-75)
    
    plt.show()
    
    # Print cluster centers
    print(f"\nCluster Centers (k={k}):")
    centers_df = pd.DataFrame(centers_original, columns=selected_features)
    print(centers_df)
    
    # Print number of points in each cluster
    print(f"\nPoints per cluster (k={k}):")
    print(pd.Series(cluster_labels).value_counts().sort_index())
    
    return kmeans, centers_df

# Perform k-means for k=2,3,4
k_values = [2, 3, 4]
kmeans_models = {}

for k in k_values:
    print(f"\n{'='*50}")
    print(f"K-means Clustering with k={k}")
    print('='*50)
    model, centers = perform_kmeans(k, X_scaled, X, selected_features)
    kmeans_models[k] = model

# Example of prediction with new data
print("\nPrediction Example:")
# Create example players with different performance levels
example_players = [
    {'goals': 20, 'rating': 8.5, 'minutes_played': 2700},  # High performer
    {'goals': 5, 'rating': 6.8, 'minutes_played': 1500},   # Average performer
    {'goals': 0, 'rating': 6.0, 'minutes_played': 500}     # Low minutes player
]

# Convert example players to DataFrame
example_df = pd.DataFrame(example_players)

# Scale the example data using the same scaler
example_scaled = scaler.transform(example_df)

print("\nPredicting clusters for example players:")
print("Example Players:")
print(example_df)

for k, model in kmeans_models.items():
    predictions = model.predict(example_scaled)
    print(f"\nPredictions for k={k}:")
    for i, pred in enumerate(predictions):
        print(f"Player {i+1} belongs to cluster {pred}")

# Save the plot descriptions for the paper
plot_description = """
3D Visualization Analysis:
The 3D scatter plot visualizes the relationship between actual goals scored, player ratings, and minutes played. 
This visualization helps us understand how these three key performance metrics relate to each other in their natural scales,
making it easier to interpret the real-world meaning of the clusters.

When examining the plot, we can observe several interesting patterns:
1. There appears to be a positive correlation between minutes played and player ratings, suggesting that players who get more playing time tend to perform better.
2. Goal-scoring shows some clustering effects, with most players scoring few goals (as expected) and a smaller number of high-scoring outliers.
3. The distribution of points in 3D space suggests natural groupings of players, which is why applying k-means clustering can help us identify these distinct player profiles.
"""

print("\nPlot Description for Paper:")
print(plot_description)
