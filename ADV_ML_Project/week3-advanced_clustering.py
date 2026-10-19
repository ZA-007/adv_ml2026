import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from fcmeans import FCM


#Loading and inspecting the dataset
df = pd.read_csv("Mall_Customers.csv")
print(df.shape)
print(df.head())
print(df.info())
print(df.isna().sum())
print(df.describe(include="all"))


#select features and scales
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features].copy() #We take those three columns from our dataset and store them in X.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaled shape:", X_scaled.shape)
print("Approx. means:", X_scaled.mean(axis=0).round(3))
print("Approx. standard deviations:", X_scaled.std(axis=0).round(3)) #We check whether each feature has been standardised correctly. The values should be close to 1



#Choose K using evidence
ks = range(2, 11) #This tells Python to try different numbers of clusters, starting at 2 and ending at 10. so we know how much groups is best to split at.
inertias, silhouettes = [], [] #This creates two empty lists. Think of them as two notebooks where Python writes down its results.


for k in ks:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled) # A loop. For each value of k, Python creates a K-Means clustering model and groups your customers.
    inertias.append(model.inertia_) # Inertia measures how close customers are to the centre of their assigned group.
    silhouettes.append(silhouette_score(X_scaled, labels))



fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(ks), inertias, marker="o")
axes[0].set(title="Elbow plot", xlabel="K", ylabel="Inertia")
axes[1].plot(list(ks), silhouettes, marker="o")
axes[1].set(title="Silhouette by K", xlabel="K", ylabel="Silhouette score")
plt.tight_layout()
plt.show()

pd.DataFrame({"K": list(ks), "inertia": inertias,
              "silhouette": silhouettes}).round(3)

#fit the hard clustering baseline
k = 5 # replace with the value you can justify
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

# Profile clusters in original feature units
print(df.groupby("kmeans_cluster")[features].mean().round(1))
print(df["kmeans_cluster"].value_counts().sort_index())

#visualize two dimensions
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x=features[1], y=features[2],
                hue="kmeans_cluster", palette="tab10")
plt.title("K-means customer segments")
plt.tight_layout()
plt.show()

# Fit fuzzy c-means
fcm = FCM(n_clusters=k, random_state=42)
fcm.fit(X_scaled)

U = fcm.u                  # shape: n_observations × k
centres_scaled = fcm.centers

print("Membership matrix shape:", U.shape)
print("First five membership vectors:\n", np.round(U[:5], 3))
print("Row sums (should be about 1):", U.sum(axis=1)[:5].round(4))

#Give each customer on emain group
df["fcm_cluster"] = U.argmax(axis=1)  # provisional hard label only
df["fcm_max_membership"] = U.max(axis=1)
sorted_U = np.sort(U, axis=1)
df["fcm_membership_gap"] = sorted_U[:, -1] - sorted_U[:, -2]

#this code was used to find the three observations with the lowest maximum memberships
print(df.nsmallest(3, "fcm_max_membership"))

#seeing how the k means and fuzzy group the customers differently
print(pd.crosstab(df["kmeans_cluster"], df["fcm_cluster"]))

#compare k = 3, 4, 5 and 6
comparison = []
for candidate_k in [3, 4, 5, 6]:
    model = FCM(n_clusters=candidate_k, random_state=42)
    model.fit(X_scaled)
    U_k = model.u
    sorted_memberships = np.sort(U_k, axis=1)
    max_membership = U_k.max(axis=1)
    gap = sorted_memberships[:, -1] - sorted_memberships[:, -2]
    comparison.append({
        "K": candidate_k,
        "mean_max_membership": max_membership.mean(),
        "median_max_membership": np.median(max_membership),
        "mean_top_two_gap": gap.mean()
    })

pd.DataFrame(comparison).round(3)

#Visualize and profile FCM
# Plot FCM's provisional labels against income and spending
plot_df = df.copy()
plt.figure(figsize=(7, 5))
sns.scatterplot(data=plot_df, x=features[1], y=features[2],
                hue="fcm_cluster", palette="tab10")
plt.title("FCM: provisional labels by maximum membership")
plt.tight_layout()
plt.show()

# Membership sharpness
plt.figure(figsize=(7, 4))
sns.histplot(df["fcm_max_membership"], bins=20, kde=True)
plt.xlabel("Maximum FCM membership")
plt.title("How sharply are customers assigned?")
plt.tight_layout()
plt.show()

# Profile by provisional FCM label
print(df.groupby("fcm_cluster")[features].mean().round(1))


