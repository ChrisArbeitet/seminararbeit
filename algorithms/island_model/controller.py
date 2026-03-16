import threading
import config
import random

class Controller:
    def __init__(self, ga_islands, ga_migration_strategy, sa_island=None):
        self.sa_island = sa_island
        self.sa_stop_event = threading.Event()
        self.sa_thread = None
        self.ga_islands = ga_islands
        self.ga_migration_strategy = ga_migration_strategy

    def run_optimization(self):
        for iteration in range(config.iterations):
            threads = [
                threading.Thread(target=island.run_optimization, args=(config.migration_interval,)) for island in self.ga_islands
                ]
            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

            self.sa_stop_event.set()
            if self.sa_thread is not None and self.sa_thread.is_alive():
                self.sa_thread.join()

            if iteration < config.iterations - 1:
                self.ga_migration_strategy.migrate(self.ga_islands)

                if config.population_reset: 
                    self.check_for_ga_restart()

                if self.sa_island is not None:
                    improved_individuals = self.sa_island.improved_individuals
                    self.assign_individuals_to_islands(improved_individuals)
                    self.sa_island.improved_individuals = []
                    self.start_simulated_annealing_island(iteration)

    def check_for_ga_restart(self, diversity_threshold=0.02):
        for island in self.ga_islands:
            diversity = island.normalized_diversity()
            if diversity < diversity_threshold:
                island.restart_population()
                print(f"Island {island.island_id} restarted due to low diversity: {diversity:.4f}")

    def assign_individuals_to_islands(self, individuals):
        for ind in individuals:
            chosen_island = random.choice(self.ga_islands)
            chosen_island.add_and_replace_individual(ind)

    def start_simulated_annealing_island(self, iteration):
        self.sa_stop_event.clear()
        best_individuals = []
        for island in self.ga_islands:
            best_individuals.extend(island.get_best_x_individuals(10))

        self.sa_island.set_population(best_individuals)
        self.sa_thread = threading.Thread(target=self.sa_island.run_optimization, args=(self.sa_stop_event, iteration))
        self.sa_thread.start()