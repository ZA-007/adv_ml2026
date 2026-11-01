#Why is scaling important when features have different ranges or units?
#Because distance-based methods such as K-means can be dominated by features with larger numerical ranges. Scaling makes the features more comparable when distances are calculated.

#6. What does standardisation change, and what does it not change?
#It changes the numerical scale and centres each feature around its mean in standard-deviation units. It does not remove the feature or make all observations identical.

#7. How might the discovered segments differ if you use only income and spending score?
#The clusters would be based only on income and spending. Other characteristics could no longer influence the grouping, so the resulting segments might be different or miss useful customer patterns. Whether they are better depends on the data and the purpose.

"""""
Before fitting K-means, you need to choose how many clusters, represented by K, you want.

For example:
- \(K=2\): divide customers into 2 groups.
- \(K=5\): divide customers into 5 groups.
- \(K=10\): divide customers into 10 groups.
The problem is that you usually don't know the best K in advance. Your practical uses two measures to help: inertia and the silhouette score.

A. Inertia and the elbow plot
Inertia measures how close observations are to the centres of their assigned clusters. In K-means, lower inertia means observations are, overall, closer to their cluster centres.

However, increasing K will generally reduce inertia because you are giving the data more cluster centres. That means the lowest inertia alone does not tell you the best K.
The elbow plot helps you look for a point where adding more clusters stops improving inertia by as much as it did earlier.

The silhouette score looks at how well an observation fits within its own cluster compared with neighbouring clusters.

In general:
- Close to +1: the observation fits its own cluster well and is separated from other clusters.
- Around 0: the observation is near a cluster boundary.
- Below 0: the observation may be closer to another cluster than the one it was assigned to.
A higher average silhouette score generally suggests more clearly separated clusters. But the highest score doesn't automatically mean the clusters are the most useful for a business.

For K-means, focus on these results:
1. Inertia
What to look at: The inertia values and elbow plot.

What it tells you: How close the observations are to their assigned cluster centres. Lower is generally better for compactness, but inertia tends to decrease as K increases.


2. Silhouette score
What to look at: The average score for different K values.

What it tells you: How well observations fit their own clusters compared with neighbouring clusters. A higher score generally suggests better separation.


3. Cluster sizes
What to look at: The number of observations in each cluster.

What it tells you: Whether some clusters are very small or most observations have ended up in one group.


4. Cluster profiles
What to look at: The average feature values for each cluster.

What it tells you: Whether the groups have understandable, useful differences.

- K-means: each customer belongs to one cluster.
- Fuzzy C-means: each customer gets a membership value for every cluster.
FCM is useful when customer groups overlap. A customer might share characteristics with both a high-spending group and a moderate-spending group.

# ==========================================
# PHASE 1: IMPORT LIBRARIES
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from fcmeans import FCM

# ==========================================
# PHASE 2: LOAD & INSPECT DATA
# ==========================================
# <--- CHANGE 'Mall_Customers.csv' to whatever file name your exam dataset has
df = pd.read_csv("Mall_Customers.csv")

print("--- DATA SHAPE ---")
print(df.shape)

print("\n--- FIRST 5 ROWS ---")
display(df.head())

print("\n--- MISSING VALUES CHECK ---")
print(df.isna().sum())  # If there are missing values, you may need to drop or fill them


# ==========================================
# PHASE 3: FEATURE SELECTION & SCALING
# ==========================================
# <--- CHANGE these column names to match the numerical columns in your exam dataset
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features].copy()

# Standardize the features so larger numbers don't dominate
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled data shape:", X_scaled.shape)


# ==========================================
# PHASE 4: FINDING OPTIMAL K (Elbow & Silhouette)
# ==========================================
ks = range(2, 11)
inertias, silhouettes = [], []

for k in ks:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

# Plotting the evidence for your report snapshots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(ks), inertias, marker="o")
axes[0].set(title="Elbow Plot", xlabel="K", ylabel="Inertia")
axes[1].plot(list(ks), silhouettes, marker="o")
axes[1].set(title="Silhouette by K", xlabel="K", ylabel="Silhouette Score")
plt.tight_layout()
plt.show()


# ==========================================
# PHASE 5A: MODEL 1 - K-MEANS (Hard Clustering)
# ==========================================
# <--- CHANGE K based on what your Elbow/Silhouette plots look like in the exam
chosen_k = 5

kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

# 1. Profile clusters in original units (Crucial for report!)
print("\n--- K-MEANS CLUSTER PROFILES ---")
display(df.groupby("kmeans_cluster")[features].mean().round(1))
display(df["kmeans_cluster"].value_counts().sort_index())

# 2. Visualize K-Means (Pick two features from your list to plot x and y)
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x=features[1], y=features[2], hue="kmeans_cluster", palette="tab10")
plt.title(f"K-Means Customer Segments (K={chosen_k})")
plt.show()


# ==========================================
# PHASE 5B: MODEL 2 - FUZZY C-MEANS (Soft Clustering)
# ==========================================
fcm = FCM(n_clusters=chosen_k, random_state=42)
fcm.fit(X_scaled)

# Extract membership matrix (U) and cluster centers
U = fcm.u  # Shape: (n_observations, k)

# Create provisional hard labels using argmax (highest membership percentage)
df["fcm_cluster"] = U.argmax(axis=1)
df["fcm_max_membership"] = U.max(axis=1)

# Calculate membership gap (to find fence-sitters/ambiguous customers)
sorted_U = np.sort(U, axis=1)
df["fcm_membership_gap"] = sorted_U[:, -1] - sorted_U[:, -2]

# 1. Profile FCM clusters in original units
print("\n--- FCM CLUSTER PROFILES ---")
display(df.groupby("fcm_cluster")[features].mean().round(1))

# 2. Visualize FCM Provisional Labels
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x=features[1], y=features[2], hue="fcm_cluster", palette="tab10")
plt.title(f"FCM Provisional Labels (K={chosen_k})")
plt.show()

# 3. Visualize Membership Sharpness (Confidence histogram)
plt.figure(figsize=(7, 4))
sns.histplot(df["fcm_max_membership"], bins=20, kde=True)
plt.title("FCM Membership Sharpness (How confident is the model?)")
plt.show()

# 4. Compare K-Means vs FCM using a Cross-Tabulation Table
print("\n--- CROSS-TABULATION: K-MEANS VS FCM ---")
display(pd.crosstab(df["kmeans_cluster"], df["fcm_cluster"]))

optional
Option 1: Hierarchical Clustering (Agglomerative)What it is: Instead of starting with random centers like K-Means,
hierarchical clustering builds clusters from the bottom up. Every single customer starts as their own tiny cluster, and the algorithm repeatedly merges the closest pairs together until it reaches your target number of clusters ($K$).
Why it's great for the exam: It's deterministic (it gives the exact same result every time you run it, unlike K-Means which can vary slightly based on random initialization).
It's the easiest alternative model to compare against K-Means.
