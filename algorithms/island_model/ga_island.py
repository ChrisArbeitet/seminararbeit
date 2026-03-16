from algorithms.island_model.base import BaseIsland
from deap import creator, base, tools, algorithms
import algorithms.island_model.population as population
import numpy as np
import config
import csv
import os

class GAIsland(BaseIsland):
    def __init__(self, id, evaluator, island_rng, max_features, log_queue, dataset, rule_type, cxpb, mutpb):
        super().__init__(id, evaluator, island_rng, max_features, log_queue, dataset)

        # --- NEU: Feature-Namen und Log-Setup ---
        self.run_id = getattr(config, "run_id", "run_1")  # optional
        self.feature_names = list(self.dataset.df_preprocessed.columns)

        self.feature_log_path = "logs/individuals_log.csv"
        os.makedirs(os.path.dirname(self.feature_log_path), exist_ok=True)

        if not os.path.exists(self.feature_log_path):
            with open(self.feature_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["run_id", "island_id", "generation", "individual_index"] + self.feature_names
                writer.writerow(header)

        self.rule_type = rule_type
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.halloffame = tools.HallOfFame(50, similar=np.array_equal)
        self.toolbox = self.create_toolbox()
        self.stats = self.init_statistics()
        self.current_generation = 0
        self.population_size = {}
        self.population_diversity_per_gen = {}
        self.best_fitness_per_gen = {}


    def create_toolbox(self):
        toolbox = base.Toolbox()
        toolbox.register("evaluate", self.evaluator.evaluate)
        toolbox.register("select", tools.selTournament, tournsize=2, rng=self.island_rng)

        if self.rule_type == "simple":
            # Register simple operators for corssover and mutation
            toolbox.register("mate", tools.crossover.cxTwoPoint)
            toolbox.register("mutate", tools.mutation.basic_mutation, indpb=0.7, max_feature=self.max_features,
                        rng=self.island_rng)

        elif self.rule_type == "with_rules":
            crossover_points = self.dataset.get_crossover_points()

            toolbox.register("mutate", tools.mutation.mutation_with_groupRules, gen_dict=self.dataset.gen_dict,
                            group_dict=self.dataset.group_dict, max_features=self.max_features, exploitation=False,
                            dontDeactivate = [], prob_multiplier=1.5, halloffame=self.halloffame,
                            rng=self.island_rng)
            # Registriert Crossover-Funktion für crossover mit Gruppen
            toolbox.register("mate", tools.crossover.adapted_grouping_crossover, group_points=crossover_points, rng=self.island_rng)

        return toolbox

    def init_statistics(self):
        # Initialize statistics to compute stats of each generation
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        return stats

    def set_population(self, population):
        # Update counter about how often each feature is selected
        self.update_gen_dict_counter(population, 0) # 0 because this is the initial population

        # Convert each individual to the DEAP Individual type
        self.population = [creator.Individual(ind) for ind in population]

    def restart_population(self):
        current_population_size = len(self.population)

        new_population = population.generate_initial_population(
            strategy=config.start_population_strategy,
            population_size=current_population_size-len(self.halloffame),
            max_features=self.max_features,
            individual_length=self.dataset.df_preprocessed.shape[1]
        )
        self.set_population(new_population)
        self.population.extend(self.halloffame)

    def inject_individuals(self, migrants):
        # Integrate migrants into the local population
        self.status = "Received migrants"
        for migrant in migrants:
            self.add_individual(migrant)

    def add_individual(self, migrant):
        ind = creator.Individual(migrant)
        ind.fitness.values = self.toolbox.evaluate(ind)
        self.population.append(ind)
        self.halloffame.update(self.population)

    def add_and_replace_individual(self, migrant):
        """Add a migrant to the population and replace the worst individual."""
        self.add_individual(migrant)
        # Replace the worst individual in the population with the migrant
        worst_ind = max(self.population, key=lambda ind: ind.fitness.values)
        self.remove_individual(worst_ind)

    def update_gen_dict_counter(self, population, x):
        """
        Update the counter for how often each feature is selected in the population.
        x = 0 for initial population, 1 for offspring population.
        """
        for ind in population:
            for index, gene in enumerate(ind):
                if gene == 1:
                    self.dataset.gen_dict[index].memory[self.dataset.target][x] += 1

    def run_optimization(self, generations):
        # Run the genetic algorithm for a specified number of generations
        self.status = "Running optimization"

        for gen in range(generations):
            self.current_generation += 1

            offspring = algorithms.varOr(
                self.population,
                self.toolbox,
                lambda_=len(self.population),
                cxpb=self.cxpb,
                mutpb=self.mutpb,
                rng=self.island_rng
            )

            self.update_gen_dict_counter(offspring, 1)

            offspring.extend(self.population)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            offspring = self.toolbox.select(
                offspring,
                k=len(self.population) - len(self.halloffame)
            )
            offspring.extend(self.halloffame)
            self.halloffame.update(offspring)
            self.population[:] = offspring
            record = self.stats.compile(self.population)

            with open(self.feature_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                for idx, ind in enumerate(self.population):
                    writer.writerow(
                        [self.run_id, self.island_id, self.current_generation, idx] +
                        list(ind)
                    )

            if gen == generations - 1:
                self.status = "Done. Waiting for next step"

            population_size = len(self.population)
            population_diversity = self.normalized_diversity()
            best_fitness = self.halloffame[0].fitness.values[0]
            self.population_size[self.current_generation] = population_size
            self.population_diversity_per_gen[self.current_generation] = population_diversity
            self.best_fitness_per_gen[self.current_generation] = best_fitness

            self.log_queue.put({
                "island_id": self.island_id,
                "generation": self.current_generation,
                "population_size": population_size,
                "best_fitness": record["min"],
                "avg_fitness": record["avg"],
                "normalized_diversity": population_diversity,
                "status": self.status,
                "population": [list(map(int, ind)) for ind in self.population],
            })

        self.log_queue.put({"island_id": self.island_id, "done": True})


class DuarteIsland(GAIsland):
    """
    This class implements the Duarte's Dynamic Migration Strategy.
    It extends the GAIsland class and overrides the inject_individuals method.
    """
    def __init__(self, id, evaluator, island_rng, max_features, log_queue, dataset, rule_type, cxpb, mutpb):
        super().__init__(id, evaluator, island_rng, max_features, log_queue, dataset, rule_type, cxpb, mutpb)
        self.immigrants_from = {}

    # def __init__(self, id, evaluator, stop_event, island_rng, max_features, log_queue):
    #    super().__init__(id, evaluator, stop_event, island_rng, max_features, log_queue)
    #    self.immigrants_from = {}

    def get_native_population(self):
        """
        Native population refers to individuals that are not migrants.
        This method is especially relevant for Duartes Dynamic Migration Strategy.
        """
        return [ind for ind in self.population if getattr(ind, 'is_native', True)]

    def get_immigrants_from(self, source):
        return self.immigrants_from.get(source.island_id, [])

    def inject_individuals(self, migrants, source, M):
        # Integrate migrants into the local population
        for migrant in migrants:
            self.add_migrant(self, migrant, source, M)

    def add_migrant(self, migrant, source, M):  # <- Parameter korrigieren
        self.status = "Received migrant"
        ind = creator.Individual(migrant)
        ind.fitness.values = self.toolbox.evaluate(ind)
        ind.is_native = False
        ind.migration_counter = M
        self.population.append(ind)
        self.immigrants_from.setdefault(source.island_id, []).append(ind)

    def update_migration_counters(self):
        # Immigrants verlieren nach M Runden Status
        for source_id, immigrants in list(self.immigrants_from.items()):
            to_native = []
            for ind in immigrants:
                ind.migration_counter -= 1
                if ind.migration_counter <= 0:
                    ind.is_native = True
                    to_native.append(ind)
            for ind in to_native:
                immigrants.remove(ind)
            if not immigrants:
                del self.immigrants_from[source_id]

    def update_fitness_prev(self):
        # Save the previous fitness values for all individuals
        for ind in self.population:
            ind.fitness_prev = ind.fitness.values[0] if hasattr(ind, 'fitness') else None