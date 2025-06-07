import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from astropy.table import Table
import pandas as pd
import glob

directory = "/Users/emilydu/Code/Code_lingustic/Data/f0_computed_table"
file_paths = glob.glob(f"{directory}/*.xlsx")

tables = {}

for file_path in file_paths:
    file_name = file_path.split("/")[-1].replace(".xlsx", "")
    df = pd.read_excel(file_path)
    tables[file_name] = Table.from_pandas(df)

data = []
true_labels = []
label_mapping = {label: idx for idx, label in enumerate(tables.keys())}

for label, table in tables.items():
    selected_data = np.array([
        table['duration'], 
        table['onset_F0'], 
        table['offset_F0'], 
        table['mean_F0'], 
        table['delta_F0']
    ]).T

    data.append(selected_data)
    true_labels.extend([label_mapping[label]] * len(selected_data))

data = np.vstack(data)
true_labels = np.array(true_labels)

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=None)
cluster_labels = kmeans.fit_predict(data)

wrong_indices = true_labels != cluster_labels

df_clustered = pd.DataFrame(data, columns=['duration', 'onset_F0', 'offset_F0', 'mean_F0', 'delta_F0'])
df_clustered['true_label'] = true_labels
df_clustered['predicted_cluster'] = cluster_labels
df_clustered['misclassified'] = wrong_indices

df_clustered.to_csv("/Users/emilydu/Code/Code_lingustic/Data/clustered_results.csv", index=False)


plt.figure(figsize=(8, 6))
scatter = plt.scatter(data[:, 0], data[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7, label="Correctly Clustered")
plt.scatter(data[wrong_indices, 0], data[wrong_indices, 1], 
            color='red', marker='x', s=20, label="Misclassified")

plt.colorbar(label="Cluster ID")
plt.xlabel("Duration")
plt.ylabel("Onset F0")
plt.title("KMeans Clustering (First Two Dimensions) with Misclassified Points")
plt.legend()
plt.show()





 
plot_pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
param_names = ['Duration', 'Onset F0', 'Offset F0', 'Mean F0', 'Delta F0']

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20, 20))
axes = axes.flatten()

for i, (idx1, idx2) in enumerate(plot_pairs):
    ax = axes[i]

    
    scatter = ax.scatter(data[:, idx1], data[:, idx2], c=cluster_labels, cmap='viridis', alpha=0.7, label="Correctly Clustered")
    ax.scatter(data[wrong_indices, idx1], data[wrong_indices, idx2], color='red', marker='x', s=20, label="Misclassified")
    ax.tick_params(labelsize = 20, direction = 'in', width =  1.2, length = 10)
    ax.set_xlabel(param_names[idx1], fontsize = 23)
    ax.set_ylabel(param_names[idx2], fontsize = 23)
    if i == 0:
        ax.legend(fontsize = 20, loc = 'upper right')
# Adjust layout
plt.tight_layout()
plt.savefig('/Users/emilydu/Code/Code_lingustic/Data/data/kmeans/graphs/kmeans_parameter.pdf', format = 'pdf')
plt.show()







