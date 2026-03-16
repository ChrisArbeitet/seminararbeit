from abc import ABC, abstractmethod
from itertools import combinations

class BaseIsland(ABC):
    def __init__(self, island_id, evaluator, island_rng, max_features, log_queue, dataset):
        self.island_id = island_id
        self.evaluator = evaluator
        self.island_rng = island_rng
        self.max_features = max_features
        self.log_queue = log_queue
        self.dataset = dataset
        self.status = None
        self.population = None

    @abstractmethod
    def set_population(self, population):
        pass

    @abstractmethod
    def inject_individuals(self, migrants):
        pass

    @abstractmethod
    def run_optimization(self, generations):
        pass

    def remove_individual(self, individual):
        """
        Removes an individual from the population.
        """
        if individual in self.population:
            self.population.remove(individual)

    def get_best_individuals(self, fraction):
        """
        Returns the top fraction of individuals based on fitness.
        """
        if not self.population:
            return []

        num_individuals = max(1, int(len(self.population) * fraction))
        sorted_pop = sorted(self.population, key=lambda ind: ind.fitness.values)
        return sorted_pop[:num_individuals]
    
    def get_best_x_individuals(self, x):
        """
        Returns the best x individuals in the population.
        """
        if not self.population:
            return []
        sorted_pop = sorted(self.population, key=lambda ind: ind.fitness.values)
        return sorted_pop[:x]

    def get_best_individual(self):
        """
        Returns the best individual in the population.
        """
        if not self.population:
            return None
        return min(self.population, key=lambda ind: ind.fitness.values)

    def hamming_distance(self, ind1, ind2):
        return sum(a != b for a, b in zip(ind1, ind2))
    
    def population_diversity(self):
        """
        Computes the average Hamming distance between all pairs of individuals in the population.
        """
        if len(self.population) < 2:
            return 0.0  # Nur ein Individuum → keine Diversität
        
        pairwise_distances = [
            self.hamming_distance(ind1, ind2)
            for ind1, ind2 in combinations(self.population, 2)
        ]
        
        return sum(pairwise_distances) / len(pairwise_distances)
    
    def normalized_diversity(self):
        """
        Computes the normalized diversity of the population.
        """
        length = len(self.population[0])
        if length == 0:
            return 0.0
        diversity = self.population_diversity()
        return diversity / length
    
    def average_distance_to_population(self, individual):
        """
        Computes the average Hamming distance of an individual to the entire population.
        """
        distances = [self.hamming_distance(individual, ind) for ind in self.population]
        return sum(distances) / len(distances)