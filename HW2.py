import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram

# Opening our text file with the genome data
data_file = open("CMPSC497_HW2_geno.txt", "r")

matrix = []
next(data_file)
for line in data_file:
    file_data = line.strip().split('\t')[1:]
    matrix.append(list(map(float,file_data)))
matrix = np.array(matrix)

# Identify columns with only undefined values (all 9s)
column_mask = np.all(matrix == 9, axis=0)
matrix[:, column_mask] = 0  # Replace these columns with 0
matrix[matrix == 9] = np.nan
means = np.nanmean(matrix, axis=0)
matrix = np.where(np.isnan(matrix), means, matrix)

# Step 1: Replace undefined values (9) with the mean of each SNP position
matrix[matrix == 9] = np.nan  # Temporarily set undefined values to NaN for handling
means = np.nanmean(matrix, axis=0)
# Replace NaNs (original 9 values) with column means
matrix = np.where(np.isnan(matrix), means, matrix)

# Step 2: Center the data by subtracting the mean of each SNP
matrix_centered = matrix - means

# Step 3: Calculate the covariance matrix of the centered data
cov_matrix = np.cov(matrix_centered, rowvar=False)

# Step 4: Calculate the eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Step 5: Sort the eigenvalues and corresponding eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Step 6: Select the top 2 eigenvectors (for 2 principal components)
top_eigenvectors = eigenvectors[:, :2]

# Step 7: Project the data onto the top 2 principal components
pca_result = matrix_centered @ top_eigenvectors

#Optional: Plotting the results
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Manual PCA of Genotype Matrix')
plt.show()

# K means clustering

# Setting a range of different clusters to test
cluster_range = range(1,11)
inertia_values = []

# Testing k-means for 1-11 clusters and storing the inertia
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pca_result)
    inertia_values.append(kmeans.inertia_)

# Plot of intertia data with different number of clusters
plt.figure(figsize=(8,6))
plt.plot(cluster_range,inertia_values,marker='o')
plt.xlabel("# of Clusters")
plt.ylabel("Inertia")
plt.grid(True)

# # Highlight the knee point
knee_k = 5
# plt.axvline(x=knee_k, color='r',linestyle='--',label=f"optimal k = {knee_k}")
# plt.legend()
# plt.show()

kmeans = KMeans(n_clusters=knee_k,random_state=42)
population_lbl = kmeans.fit_predict(pca_result)

avg_hetzyg_arr =[]

for i in range(knee_k):
    pop_ind = np.where(population_lbl==i)[0]
    pop_data = matrix[pop_ind]

    avg_hetzyg_data = np.mean(pop_data == 1, axis=0).mean()
    avg_hetzyg_arr.append(avg_hetzyg_data)

avg_hetzyg_arr = np.array(avg_hetzyg_arr)

dist_matrix = np.abs(avg_hetzyg_arr[:, None]-avg_hetzyg_arr[None, :])

plt.figure(figsize=(8,6))
sns.heatmap(dist_matrix,annot=True,cmap="YlGnBu",xticklabels=[f'Pop {i+1}' for i in range(knee_k)],yticklabels=[f'Pop {i+1}' for i in range(knee_k)])
plt.title("Population Distance Matrix")
plt.xlabel("Population")
plt.ylabel("Population")
plt.show()

linked = linkage(dist_matrix, 'ward')

plt.figure(figsize=(10,7))
dendrogram(linked, labels=[f'Pop {i+1}' for i in range(knee_k)], leaf_rotation=90)
plt.title("Dendrogram of Populations")
plt.xlabel("Population")
plt.ylabel("Distance")
plt.show()


