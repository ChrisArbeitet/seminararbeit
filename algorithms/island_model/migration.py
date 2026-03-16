from abc import ABC, abstractmethod
import random

class MigrationStrategy(ABC):
    def __init__(self, topology):
        self.topology = topology

    @abstractmethod
    def migrate(self, islands):
        """
        Perform migration among the given islands.
        
        Parameters:
            - islands: List of islands participating in the migration.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
class ElitistMigration(MigrationStrategy):
    def __init__(self, topology, elite_fraction=0.1, replacement_strategy="SoftMigration"):
        super().__init__(topology)
        self.elite_fraction = elite_fraction
        # Replacement strategy can be "StrictMigration" or "SoftMigration"
        # Strict Migration: Migrants are removed from the source island and injected into the target island.
        # Soft Migration: Migrants are added to the target island and replace the worst individual.
        self.replacement_strategy = replacement_strategy 

    def migrate(self, islands):
        """
        Perform elitist migration among the given islands. 
        Currently injects the elite individuals from each island to all migration targets.
        """
        migrations = []
        for source in islands:
            migrants = source.get_best_individuals(self.elite_fraction)
            targets = self.topology.get_migration_targets(source, islands)

            random.shuffle(migrants)

            num_targets = len(targets)
            chunks = [ [] for _ in range(num_targets) ]
            for i, migrant in enumerate(migrants):
                chunks[i % num_targets].append(migrant)
         
            for chunk, target in zip(chunks, targets):
                for migrant in chunk:
                    migrations.append((source, target, migrant))

        for source, target, migrant in migrations:
            if self.replacement_strategy == "StrictMigration":
                source.remove_individual(migrant)
                target.add_individual(migrant)
            elif self.replacement_strategy == "SoftMigration":
                target.add_and_replace_individual(migrant)


class DiverseMigration(MigrationStrategy):
    def __init__(self, topology, migration_fraction=0.1):
        super().__init__(topology)
        self.migration_fraction = migration_fraction
    
    def migrate(self, islands):
        migrations = []
        for source in islands: 
            source_population = source.population
            total_migrants = max(1, int(len(source_population) * self.migration_fraction))

            targets = self.topology.get_migration_targets(source, islands)
            migrants_per_target = max(1, total_migrants // len(targets))
            for target in targets:
                distances = [
                    (ind, target.average_distance_to_population(ind))
                    for ind in source_population
                ]

                distances.sort(key=lambda x: x[1], reverse=True)

                migrants = [ind for ind, _ in distances[:migrants_per_target]]
                for migrant in migrants:
                    migrations.append((source, target, migrant))
                    source.remove_individual(migrant)

        for source, target, migrant in migrations:
            target.add_individual(migrant)

class DuarteDynamicMigration(MigrationStrategy):
    """
    This migration strategy is based on the Duarte et al. (2017) "A dynamic migration policy to the Island Model".
    The paper can be found here: https://ieeexplore.ieee.org/abstract/document/7969434
    
    This strategy chooses migration targets based on dynamically adjusted connection weights that reflect the attractiveness of 
    destination islands. Attractiveness is computed from improvements in native and immigrant populations, enabling adaptive, 
    probabilistic migration decisions.

    NOTE: In the paper, the chosen topology is a fully connected one. Although this strategy is more a topology rather than a 
    migration strategy, it is implemented here due to special rules while choosing the migrants. This wouldnt be compatible with 
    the existing migration strategies.
    """
    def __init__(self, topology, M=3, theta=0.1):
        super().__init__(topology)
        self.M = M # Number of migrations before individuals become native
        self.theta = theta
        self.attractiveness_prev = {}

    def calculate_attractiveness(self, source, targets):
        alpha = {}

        for target in targets:
            native_prev = [getattr(ind, 'fitness_prev', ind.fitness.values[0]) for ind in target.get_native_population()]
            native_curr = [ind.fitness.values[0] for ind in target.get_native_population()]
            Sp = len(native_curr)
            eta_pop = 0
            if Sp > 0:
                eta_pop = sum(f_prev - f_curr for f_prev, f_curr in zip(native_prev, native_curr)) / Sp

            immigrants = target.get_immigrants_from(source)
            Sm = len(immigrants)
            eta_mig = 0
            if Sm > 0:
                immigrants_prev = [getattr(ind, 'fitness_prev', ind.fitness.values[0]) for ind in immigrants]
                immigrants_curr = [ind.fitness.values[0] for ind in immigrants]
                eta_mig = sum(f_prev - f_curr for f_prev, f_curr in zip(immigrants_prev, immigrants_curr)) / Sm

            if target.island_id not in self.attractiveness_prev:
                self.attractiveness_prev[target.island_id] = 0

            # Special cases from the paper:
            if Sp == 0 and Sm > 0:
                alpha[target.island_id] = self.attractiveness_prev[target.island_id] + eta_mig
            elif Sm == 0 and Sp > 0:
                eta_mig = self.theta * eta_pop
                alpha[target.island_id] = self.attractiveness_prev[target.island_id] + eta_pop + eta_mig
            elif Sp == 0 and Sm == 0:
                alpha[target.island_id] = -1  # Invalid
            else:
                alpha[target.island_id] = self.attractiveness_prev[target.island_id] + eta_pop + eta_mig
        
        return alpha

    def calculate_connection_weights(self, attractiveness, targets):
        weights = {}
        delta = sum(attractiveness[target.island_id] for target in targets)

        for target in targets:
            alpha_j = attractiveness[target.island_id]
            weight = alpha_j / delta if delta > 0 else 0
            weights[target.island_id] = weight

        return weights

    def weighted_choice(self, weights, targets):
        total_weight = sum(weights[target.island_id] for target in targets)
        r = random.uniform(0, total_weight)
        cumulative_weight = 0.0
        for target in targets:
            cumulative_weight += weights[target.island_id]
            if r < cumulative_weight:
                return target
        return random.choice(targets)  # Fallback in case of rounding errors

    def migrate(self, islands):
        migrations = [] # List of (Source Island, Target Island, Individuals)

        for source in islands:
            # Get migration targets based on the topology
            targets = self.topology.get_migration_targets(source, islands)

            # Choose a random number of migrants here (only from native population) -> In Paper 10% of the population
            native_population = source.get_native_population()
            print(f"Länge der nativen Population: {len(native_population)}")
            num_migrants = max(0, int(len(native_population) * 0.1))
            print(f"Num Migrants: {num_migrants}")
            migrants = random.sample(native_population, num_migrants)

            # Calculate the attractiveness of each target island
            attractiveness = self.calculate_attractiveness(source, targets)
            weights = self.calculate_connection_weights(attractiveness, targets)

            for target in targets: 
                alpha = attractiveness[target.island_id]
                self.attractiveness_prev[target.island_id] = alpha

            for migrant in migrants:
                target = self.weighted_choice(weights, targets)
                migrations.append((source, target, migrant))

        print(f"Number of Migrations: {len(migrations)}")
        # Perform the actual migration
        for source, target, migrant in migrations:
            source.remove_individual(migrant)
            target.add_migrant(migrant, source, self.M)

        for island in islands:
            island.update_migration_counters()
            island.update_fitness_prev()