from abc import ABC, abstractmethod
import random

class TopologyStrategy(ABC):
    @abstractmethod
    def get_migration_targets(self, source_island, all_islands):
        """
        Get the migration targets for a given source island.

        Parameters:
            - source_island: The island from which individuals will migrate.
            - all_islands: List of all islands in the model.

        Returns:
            - List of target islands for migration.
        """
        pass
    
class RingTopology(TopologyStrategy):
    def get_migration_targets(self, source_island, all_islands):
        idx = all_islands.index(source_island)
        return [all_islands[(idx + 1) % len(all_islands)]]

class FullyConnectedTopology(TopologyStrategy):
    def get_migration_targets(self, source_island, all_islands):
        return [island for island in all_islands if island != source_island]
    
class StarTopology(TopologyStrategy):
    """
    Star topology where one central island receives migrants from all others and sends migrants to all others.
    This will lead to a central island that has n-1 times as many individuals as the other islands. 
    Therefore it is advised to implement a migration protocol where outer islands dont have to wait due to lower populationn sizes.
    """
    def __init__(self, central_island):
        self.central_island = central_island

    def get_migration_targets(self, source_island, all_islands):
        if source_island == self.central_island:
            return [island for island in all_islands if island != self.central_island]
        else:
            return [self.central_island]
        
class RandomTopology(TopologyStrategy):
    """Random topology where each island randomly selects up to k other islands as migration targets."""
    def __init__(self, k=2):
        self.k = k

    def get_migration_targets(self, source_island, all_islands):
        targets = all_islands.copy()
        targets.remove(source_island)
        return random.sample(targets, k=min(len(targets), self.k))  # Randomly select up to 2 targets
    
