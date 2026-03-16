from algorithms.island_model.ga_island import GAIsland, DuarteIsland
from algorithms.island_model.sa_island import SimulatedAnnealing
from algorithms.island_model.topology import RingTopology, FullyConnectedTopology, StarTopology, RandomTopology
from algorithms.island_model.migration import ElitistMigration, DuarteDynamicMigration, DiverseMigration
from algorithms.evaluation.fitness_evaluator import FitnessEvaluator
from algorithms.utils.terminal_logger import monitor_progress
from algorithms.island_model.results import process_results
import algorithms.island_model.population as population
from algorithms.island_model.controller import Controller
from deap import base, creator, tools, algorithms
import threading
import random
from queue import Queue
import config
import timeit

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

TOPOLOGY_MAP = {
    "ring": RingTopology,
    "fully_connected": FullyConnectedTopology,
    "star": StarTopology,
    "random": RandomTopology,
}

MIGRATION_STRATEGY_MAP = {
    "elitist": ElitistMigration,
    "duarte_dynamic": DuarteDynamicMigration,
    "diverse": DiverseMigration,
}

def initialize_migration_strategy():
    topology = TOPOLOGY_MAP[config.topology]()
    migration_strategy = MIGRATION_STRATEGY_MAP[config.migration_strategy](topology=topology)
    return migration_strategy

def initialize_ga_islands(num_islands, dataset, log_queue):
    islands = []
    for i in range(num_islands):
        island_rng = random.Random(random.randint(0, 100_000_000))

        evaluator = FitnessEvaluator(target=dataset.target, data_split=0.8, dataset=dataset.df_preprocessed, 
                                                max_features=dataset.max_features, penalty=True)
        
        if config.migration_strategy == "duarte_dynamic":
            island = DuarteIsland(id=i, evaluator=evaluator, island_rng=island_rng, max_features=dataset.max_features,
                          log_queue=log_queue, dataset=dataset, rule_type=config.rule_type, cxpb=config.cxpb, mutpb=config.mutpb)
        else:
            island = GAIsland(id=i, evaluator=evaluator, island_rng=island_rng, max_features=dataset.max_features,
                          log_queue=log_queue, dataset=dataset, rule_type=config.rule_type, cxpb=config.cxpb, mutpb=config.mutpb)
        islands.append(island)

    initial_population = population.generate_initial_population(
        strategy=config.start_population_strategy,
        population_size=config.pop_size,
        max_features=dataset.max_features,
        individual_length=dataset.df_preprocessed.shape[1]
    )

    initial_population_splits = population.split_population(
        strategy="random",
        population=initial_population,
        num_islands=num_islands,
    )

    for island, pop in zip(islands, initial_population_splits):
        island.set_population(pop)
    
    return islands

def initialize_sa_island(dataset, log_queue, id):
    evaluator = FitnessEvaluator(target=dataset.target, data_split=0.8, dataset=dataset.df_preprocessed, 
                                                max_features=dataset.max_features, penalty=True)
    island_rng = random.Random(random.randint(0, 100_000_000))
    sa_island = SimulatedAnnealing(id=id, evaluator=evaluator, island_rng=island_rng, max_features=dataset.max_features,
                          log_queue=log_queue, dataset=dataset)
    return sa_island

def start_logging_thread(num_islands, target):
    """Start a thread to log the progress of the optimization."""
    log_queue = Queue()
    logger_stop_event = threading.Event()
    logger_thread = threading.Thread(
        target=monitor_progress,
        args=(log_queue, num_islands, target, logger_stop_event)
    )
    logger_thread.start()
    return logger_thread, log_queue, logger_stop_event

def execute(num_islands, dataset, results_folder, results_file_path):
    logger_thread, logging_queue, logger_stop_event = start_logging_thread(num_islands, dataset.target)
    
    if config.sa_island_active:
        sa_island = initialize_sa_island(dataset, log_queue=logging_queue, id=num_islands)
    else:
        sa_island = None

    ga_islands = initialize_ga_islands(num_islands, dataset, log_queue=logging_queue)

    migration_strategy = initialize_migration_strategy()
    controller = Controller(ga_islands, migration_strategy, sa_island=sa_island)
    controller.run_optimization()
    logger_stop_event.set() # End logger thread
    logger_thread.join() # Wait for logger thread to finish
    process_results(ga_islands, results_folder, results_file_path, dataset, sa_island)