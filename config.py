from pathlib import Path

"""
    File provides an overview of all necessary file paths and hyperparameters required to configure
    and run the genetic algorithm.
"""

"""
    File paths
"""
# Base path for input files
data_base_path = "data"

# Directory of MFT-files -> All MFT-files should begin with 'MFT'
mft_dir = "MFT*.csv"

# Directory of Laboratory files
lab_dir = "LABORATORY.csv"

# File with expert evaluations of the single features
exp_knowledge = "Expert_knowledge.xlsx"

# Tags table to map evaluations to features
tags_table = "240722_TAGS_Tabelle.csv"

#project folder
BASE_DIR = Path(__file__).parent

EXCEL_PATH = BASE_DIR / "algorithms" / "Beste-Features.xlsx"
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data"


"""
    General Hyperparameters for the GA
"""
# Defines the group rules. Currently implemented two Versions, 'Version_A' & 'Version_B'
group_type = "Version_A"

# Type of generation of the start population, currently implemented 'random' and 'random_forest'
start_population_strategy = 'random'

# Number of individuals per population
pop_size: int = 300

# Population reset if diversity drops below threshold
population_reset: bool = False


"""
    Hyperparameters for the Island Model
"""
# Number of Islands
num_islands: int = 4

# Migration Interval
migration_interval: int = 15

# Number of Iterations: Total number of generations is then migration_interval * iterations
iterations: int = 30

# Migration size: Typically 10% of Population size
migration_size: int = 75

# Topology of the islands
topology = "fully_connected"  # fully_connected, ring, star, random

migration_strategy = "diverse"  # diverse, elitist, duarte_dynamic

sa_island_active: bool = False

rule_type = 'with_rules'

# cxpb/mutpb abhängig von rule_type:
#   rule_type='simple'     → cxpb=0.8, mutpb=0.1
#   rule_type='with_rules' → cxpb=0.3, mutpb=0.5
cxpb  = 0.3
mutpb = 0.5


"""
    Others
"""
# Material type number used to filter the dataset
material_type: list = [24007453]

# Number of runs the Algorithm is executed
validation_runs: int = 1

# Form of regression – als Liste: ['OLS'], ['PLS'] oder ['Ridge']
regression = ['Ridge']

# Target variable – als Liste: ['IB_AVG'], ['DENSITY_AVG'] oder ['MOR_AVG']
targets = ['MOR_AVG']