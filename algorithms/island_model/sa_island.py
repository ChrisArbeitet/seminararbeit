import numpy as np
from algorithms.island_model.base import BaseIsland
from deap import creator

class SimulatedAnnealing(BaseIsland):
    def __init__(self, id, evaluator, island_rng, max_features, log_queue, dataset):
        super().__init__(id, evaluator, island_rng, max_features, log_queue, dataset)
        self.improved_individuals = []
        self.sa_log = []

    def set_population(self, population):
        self.population = [creator.Individual(ind) for ind in population]

    def inject_individuals(self, migrants):
        for migrant in migrants:
            ind = creator.Individual(migrant)
            ind.fitness.values = self.toolbox.evaluate(ind)
            self.population.append(ind)

    def set_sa_log(self, iteration, fitness_before, fitness_after):
        self.sa_log.append({
            "iteration": iteration,
            "fitness_before": fitness_before,
            "fitness_after": fitness_after
        })

    def run_optimization(self, stop_event, ga_iteration):
        while not stop_event.is_set():
            for ind in self.population:
                if not ind.fitness.valid:
                    ind.fitness.values = self.evaluator.evaluate(ind)

            best_ind = min(self.population, key=lambda ind: ind.fitness.values[0])
            self.population.remove(best_ind)
            best_solution, fitness = self.optimize(best_ind, max_iter=500, initial_temp=1, cooling_rate=0.99, min_temp=1e-3)
            self.improved_individuals.append(best_solution)
            self.set_sa_log(ga_iteration, best_ind.fitness.values[0], fitness)
            print(f"SA improved fitness from {best_ind.fitness.values[0]} to {fitness}")

    def optimize(self, ind, max_iter, initial_temp, cooling_rate, min_temp):
        """
        Runs the simulated annealing optimization for a specified number of generations.
        """
        ind = np.array(ind)

        if ind.sum() > self.max_features:
            selected = np.where(ind == 1)[0]
            to_remove = np.random.choice(selected, size=ind.sum() - self.max_features, replace=False)
            ind[to_remove] = 0
        elif ind.sum() < self.max_features:
            unselected = np.where(ind == 0)[0]
            to_add = np.random.choice(unselected, size=self.max_features - ind.sum(), replace=False)
            ind[to_add] = 1

        current_solution = ind.copy()
        current_fitness = self.evaluator.evaluate(current_solution.tolist())[0]
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        T = initial_temp

        for iteration in range(max_iter):
            selected = np.where(current_solution == 1)[0]
            unselected = np.where(current_solution == 0)[0]

            remove_id = np.random.choice(selected)
            add_id = np.random.choice(unselected)

            neighbor = current_solution.copy()
            neighbor[remove_id] = 0
            neighbor[add_id] = 1

            neighbor_fitness = self.evaluator.evaluate(neighbor.tolist())[0]
            delta_fitness = current_fitness - neighbor_fitness

            if delta_fitness > 0 or np.random.rand() < np.exp(delta_fitness / T):
                current_solution = neighbor
                current_fitness = neighbor_fitness

                if current_fitness < best_fitness:
                    best_solution = current_solution.copy()
                    best_fitness = current_fitness

            T *= cooling_rate
            if T < min_temp:
                break

        return best_solution.tolist(), best_fitness
