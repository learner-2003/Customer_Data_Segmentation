import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("customers_1000.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    x='AnnualIncome',
    y='SpendingScore',
    data=df
)
plt.title("Income vs Spending Score")
plt.show()

X = df[['Age', 'AnnualIncome', 'SpendingScore']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []

for i in range(1, 11):
    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )
    model.fit(X_scaled)
    wcss.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nCluster Summary")

cluster_summary = df.groupby('Cluster').agg({
    'Age': 'mean',
    'AnnualIncome': 'mean',
    'SpendingScore': 'mean',
    'CustomerID': 'count'
}).rename(columns={
    'CustomerID': 'CustomerCount'
})

print(cluster_summary)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x='AnnualIncome',
    y='SpendingScore',
    hue='Cluster',
    palette='tab10',
    s=80
)

plt.title("Customer Segments")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.show()

plt.figure(figsize=(8, 5))

sns.countplot(
    x='Cluster',
    data=df
)

plt.title("Customers per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")

plt.show()

output_file = "segmented_customers.csv"

df.to_csv(output_file, index=False)

print(f"\nSegmented data saved to:\n{output_file}")