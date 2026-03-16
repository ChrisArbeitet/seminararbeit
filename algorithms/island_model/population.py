from deap import tools
import random
import numpy as np
from sklearn.cluster import KMeans
from algorithms.utils.visualization import plot_cluster_heatmap

def create_individual(max_features, individual_length):
    """Create a new individual for the population."""
    num_selected_features = random.randint(max_features-4, max_features)

    # Generate a genome with num_selected_features 1's and (number_columns - num_selected_features) 0's
    genome = ([1] * num_selected_features) + ([0] * (individual_length - num_selected_features))

    # Shuffle the genome randomly
    np.random.shuffle(genome)

    return genome
    
def generate_random_population(population_size, max_features, individual_length):
    return [create_individual(max_features, individual_length) for _ in range(population_size)]

def generate_random_forest_population(population_size, max_features, individual_length):
    pass

population_generators = {
    "random": generate_random_population,
    "random_forest": generate_random_forest_population,
}

def generate_initial_population(strategy: str, population_size, max_features: int, individual_length: int):
    try:
        return population_generators[strategy](population_size, max_features, individual_length)
    except KeyError:
        raise ValueError(f"Unknown population generation strategy: {strategy}")

def random_split(population, num_islands: int, create_activation_heatmap=False):
    """Split the population randomly into num_islands parts."""
    random.shuffle(population)
    splits = [population[i::num_islands] for i in range(num_islands)]

    # Create labels for the heatmap if requested
    if create_activation_heatmap:
        labels = [None] * len(population)
        idx = 0
        for island_id, group in enumerate(splits):
            for _ in group:
                labels[idx] = island_id
                idx += 1

        plot_cluster_heatmap(np.array(population), labels, num_islands)

    return splits

def kmeans_cluster_split(population, num_islands: int, create_activation_heatmap=False):
    """Split the population into clusters using KMeans."""
    print(f"Clustering population into {num_islands} islands using KMeans...\n")

    if len(population) < num_islands:
        raise ValueError("Population size must be greater than or equal to the number of islands for clustering.")

    # Feature-Matrix: population ist schon Liste von Listen aus 0/1
    X = np.array(population)

    # KMeans Clustering
    kmeans = KMeans(n_clusters=num_islands, random_state=42)
    labels = kmeans.fit_predict(X)

    # Cluster zusammensetzen
    clusters = [[] for _ in range(num_islands)]
    for ind, label in zip(population, labels):
        clusters[label].append(ind)

    if create_activation_heatmap: plot_cluster_heatmap(X, labels, num_islands)

    return clusters

population_splitters = {
    "random": random_split,
    "kmeans_cluster": kmeans_cluster_split,
}

def split_population(strategy: str, population, num_islands: int):
    try:
        return population_splitters[strategy](population, num_islands)
    except KeyError:
        raise ValueError(f"Unknown population generation strategy: {strategy}")