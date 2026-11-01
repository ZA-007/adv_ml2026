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

print("Scaled shape:", X_scaled.shape) #filtered out the customerid and gender , keeping only the 3 numerical features and standarized them
print("Approx. means:", X_scaled.mean(axis=0).round(3))
print("Approx. standard deviations:", X_scaled.std(axis=0).round(3)) #We check whether each feature has been standardised correctly. The values should be close to 1



#Choose K using evidence
ks = range(2, 11) #This tells Python to try different numbers of clusters, starting at 2 and ending at 10. so we know how much groups is best to split at.
inertias, silhouettes = [], [] #This creates two empty lists. Think of them as two notebooks where Python writes down its results.


for k in ks:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled) # A loop. For each value of k, Python creates a K-Means clustering model and groups your customers.
    inertias.append(model.inertia_) # Inertia measures how close customers are to the centre of their assigned group.
    silhouettes.append(silhouette_score(X_scaled, labels)) #calculates the average silhoutte score for those cluster assignemnts


#this plotting code creates two side by side graphs

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(ks), inertias, marker="o")
axes[0].set(title="Elbow plot", xlabel="K", ylabel="Inertia")
axes[1].plot(list(ks), silhouettes, marker="o")
axes[1].set(title="Silhouette by K", xlabel="K", ylabel="Silhouette score")
plt.tight_layout()
plt.show()

#this displays a table of the tested K values and their two score, rounded to three decimal places.
pd.DataFrame({"K": list(ks), "inertia": inertias,
              "silhouette": silhouettes}).round(3)



#after finding out which k cluster is best based on elbow pointor silhoutte score, begin the k means clustering model
#fit the hard clustering baseline

k = 5 # replace with the value you can justify
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

#n_clusters=k: tells K-means how many clusters to create.
#fit_predict(X_scaled): fits the model and returns one label for each row.
#df["kmeans_cluster"] = ...: saves those labels in a new column of your original DataFrame, df.

# Profile clusters in original feature units, it tells u what the groups of clusters mean
print(df.groupby("kmeans_cluster")[features].mean().round(1))
print(df["kmeans_cluster"].value_counts().sort_index())

#This helps you describe the customers in each group using the original units.
#The second line counts how many observations belong to each cluster. sort_index() orders the output by cluster label.
#For example, a hypothetical cluster might have an average income of £60,000 and an average spending score of 75.
#You could describe it as a group of higher-income customers with relatively high spending scores, assuming those values fit your dataset.
#Without profiling, you might know that a customer belongs to Cluster 1, but you wouldn't necessarily know what makes Cluster 1 different from the others.

#visualize the clusters/ two dimensions
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x=features[1], y=features[2],
                hue="kmeans_cluster", palette="tab10") #creates a scatter plot, with each point representing an observation.
plt.title("K-means customer segments")
plt.tight_layout()
plt.show()

# Fit fuzzy c-means
fcm = FCM(n_clusters=k, random_state=42) #creates the FCM model with the selected number of clusters
fcm.fit(X_scaled) #fits the model to the scaled data

U = fcm.u    # retreives the membership matrix saves as U
centres_scaled = fcm.centers #retreives the cluster centres in the scaled feature space



print("Membership matrix shape:", U.shape) #shows the number of rows and columns
print("First five membership vectors:\n", np.round(U[:5], 3))  #U[:5] selects the first five rows, shows u how mcuh each custoemr belongs to each cluster, ex cluster 0: 31%, cluster 1 , 10% and so on.
print("Row sums (should be about 1):", U.sum(axis=1)[:5].round(4))

#Give each customer one main group
df["fcm_cluster"] = U.argmax(axis=1)  # provisional hard label only
df["fcm_max_membership"] = U.max(axis=1)
sorted_U = np.sort(U, axis=1)
df["fcm_membership_gap"] = sorted_U[:, -1] - sorted_U[:, -2]

#U.argmax(axis=1)	Finds the cluster with the highest membership for each observation.
#U.max(axis=1)	Gets each observation's highest membership value.
#np.sort(U, axis=1)	Sorts each row's membership values from lowest to highest.
#sorted_U[:, -1]	Selects the highest membership in each row.
#sorted_U[:, -2]	Selects the second-highest membership.
#fcm_membership_gap	Calculates the difference between the top two memberships.

#argmax gives FCM a single label for each customer by selecting the strongest membership.
# That's why your notes call this a provisional hard label: the original FCM result is fuzzy, but this line converts it into a single cluster label for convenient comparison and plotting.


#this code was used to find the three observations with the lowest maximum memberships
print(df.nsmallest(3, "fcm_max_membership"))
#the difference between a customer's #1 top cluster score and their #2 runner-up cluster score).
#Customer 123 has a gap of only 0.015. That means their top choice and second choice are practically identical

#seeing how the k means and fuzzy group the customers differently
print(pd.crosstab(df["kmeans_cluster"], df["fcm_cluster"]))
#What this means: K-Means strictly dumped all 47 customers into Group 4. But when Fuzzy C-Means looked at their overlapping boundaries, it realized that 44 of those customers actually have a stronger affinity or overlap with FCM Cluster 1

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


#optional hierarchal clustering

from sklearn.cluster import AgglomerativeClustering

# <--- Use the same 'chosen_k' you used for K-means
hier = AgglomerativeClustering(n_clusters=k, linkage="ward")
df["hierarchical_cluster"] = hier.fit_predict(X_scaled)

# 1. Profile hierarchical clusters in original units
print("\n--- HIERARCHICAL CLUSTERING PROFILES ---")
print(df.groupby("hierarchical_cluster")[features].mean().round(1)) #avergae for age, annual income and speniding score
print(df["hierarchical_cluster"].value_counts().sort_index())

# 2. Visualize Hierarchical Clusters
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x=features[1], y=features[2], hue="hierarchical_cluster", palette="tab10")
plt.title("Hierarchical Clustering (Ward Linkage)")
plt.show()