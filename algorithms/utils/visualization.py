import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator

def plot_cluster_heatmap(X, labels, num_clusters):
    cluster_means = []

    for cluster_id in range(num_clusters):
        members = X[np.array(labels) == cluster_id]
        if len(members) == 0:
            mean_vector = np.zeros(X.shape[1])
        else:
            mean_vector = members.mean(axis=0)
        cluster_means.append(mean_vector)

    heatmap_data = np.array(cluster_means)

    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap="YlGnBu", cbar=True, xticklabels=False, yticklabels=[f"Cluster {i}" for i in range(num_clusters)])
    plt.xlabel("Feature Index")
    plt.ylabel("Cluster")
    plt.title("Cluster-wise Feature Activation Heatmap")
    plt.savefig("src/algorithms/utils/cluster_activation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_dict_curves(dict_list, title, ylabel, results_folder, target):
    num_curves = len(dict_list)
    colors = cm.viridis([i / num_curves for i in range(num_curves)])  # gleichmäßig verteilte Farben

    for i, d in enumerate(dict_list):
        keys = sorted(d.keys())
        values = [d[k] for k in keys]
        label = f"Insel {i+1}"
        plt.plot(keys, values, label=label, color=colors[i])

    png_path = f"{results_folder}/{title.replace(' ', '_').lower()}_{target}.png"

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.legend()
    # plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

    return png_path
