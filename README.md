# Particleboard Quality Prediction via Genetic Algorithm Feature Selection
Quality prediction of particleboards is currently only possible on a sampling
basis using destructive testing methods. This project addresses this limitation
by developing a genetic algorithm to reduce the relevant production parameters and predict quality
 with ML-modells.

## Motivation
The production process of particleboards involves over 300 measurable process
parameters. Manual selection of relevant features is therefore very cost and time expensive.
To automate this step, a Genetic Algorithm (GA) was developed that reduces the feature space
from 300+ parameters to a maximum of 20 — without significant loss of predictive power.

## Target Variables
The models predict three key quality indicators:
- **Modulus of Rupture (MOR)** – bending strength of the board
- **Density** – bulk density of the finished product
- **Internal Bond (IB)** – internal tensile strength perpendicular to the surface

## Data Basis
The dataset is compiled from two sources, linked via a unique data key:
- **Laboratory data** – destructive measurements of the three quality factors
- **Machine data** – process parameter measurements recorded during production

## Genetic Algorithm

A Genetic Algorithm (GA) is a population-based optimization method that iteratively 
improves candidate solutions. Each individual in the population represents one possible solution and is evaluated by a
fitness function. Over multiple generations, individuals with higher fitness are more likely
to survive and generate offspring, guiding the search toward better solutions.

To improve candidate solutions the GA uses three different operations:

- **Selection**: Selection determines which individuals are chosen to reproduce based
on their fitness. Individuals with higher fitness have a greater probability of being 
selected, which guides the population toward better solutions over successive generations.


- **Crossover**: Crossover combines the information of two parent individuals to create new offspring.
Its purpose is to transfer and recombine beneficial characteristics from different solutions, 
allowing the algorithm to explore promising regions of the search space more efficiently.


- **Mutation**: Mutation introduces random changes into an individual after reproduction. 
This helps maintain diversity in the population, prevents the search from becoming too uniform, 
and reduces the risk of premature convergence to suboptimal solutions.

To improve exploration and reduce the risk of premature convergence, the GA can be implemented 
as an island model, in which several subpopulations evolve independently and exchange individuals 
at predefined intervals. Additional diversity regulation helps maintain variation within and across 
populations, preventing the search from collapsing too early into similar solutions.

## Installation
- **Clone the repository:**
```bash
git clone https://github.com/ChrisArbeitet/seminararbeit.git
cd your-repository
```
- **Python version needs to be <= 13.3**
- **Relevant external data:**
  - 240722_TAGS_Tabelle.csv
  - Expert_knowledge.xlsx
  - group_rules_A.xlsx
  - group_rules_B.xlsx
  - group_Version_A.xlsx
  - group_Version_B.xlsx
  - LABORATORY.csv
  - MFT_VALUE_LAB_COOLINGLINE.csv
  - MFT_VALUE_LAB_FORMING_LINE.csv
  - MFT_VALUE_LAB_GLUEING.csv
  - MFT_VALUE_LAB_HEATING.csv
  - MFT_VALUE_LAB_PREPARATION.csv
  - MFT_VALUE_LAB_PRESS_COMMON.csv
- **Relevant packages are in requirements.txt and can be installed directly**

## Usage/Quick Start

The algorithm supports two execution modes: **single run** and **batch mode**.
The mode is selected via the `--mode` argument when starting the script.

### Single Run

In single run mode, the GA is executed once using all hyperparameters defined
in `config.py`. Only the random seed and an optional label for the results
folder can be passed via the command line.

```python
python main.py --mode single
```

### Batch Mode

In batch mode, the algorithm reads run configurations from an Excel file
and executes them sequentially.

```python
python main.py --mode batch
```
## Batch Configuration File

The batch mode reads run configurations from an Excel file (`Beste-Features.xlsx`),
sheet name `Tabelle1`. Each row defines one independent algorithm run.

### Required Columns

| Column | Allowed Values | Description |
|--------|---------------|-------------|
| `Qualityparameter` | `IB`, `Density`, `MOR` | Target quality variable to predict |
| `Seed` | any integer | Random seed for reproducibility |
| `Regression` | `OLS`, `PLS`, `Ridge` | Regression model for fitness evaluation |
| `Rules` | `No exp`, `No rules`, `With Rules`, `With rules` | Operator rule set |

### Example

Copy the following table directly into Excel (sheet name: `Tabelle1`):

| Qualityparameter | Seed | Regression | Rules |
|------------------|------|------------|-------|
| MOR | 42 | Ridge | With rules |
| IB | 42 | Ridge | With rules |
| Density | 42 | Ridge | With rules |
| MOR | 17 | OLS | No rules |
| IB | 17 | OLS | No rules |
| Density | 17 | OLS | No rules |

> **Note:** The sheet must be named exactly `Tabelle1`.
> The column headers must match exactly as shown above (case-sensitive) 
> and it must be saved in the algorithms folder .
## Hyperparameters
the current algorithm depends on many different Hyperparameters, which need to be initialized 
beforehand in config.py:

```python

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
```
- **Group Type** – defines which version of the production group rules is applied. Two configurations are implemented: 'Version_A' and 'Version_B'


- **Start Population** - choose between a random start population or a start population created with
a random forest initialization. Since the random forrest start population can not improve the fitness
due to the big input area, it is recommended to use the random initialization to improve performance


- **Population Size** - Defines how many candidate solutions exist simultaneously in each generation.
A larger population generally increases diversity and exploration, but also raises the
computational cost per generation.


- **Population Reset** - Determines whether the population is reset when diversity falls below
a predefined threshold. This mechanism helps prevent premature convergence by restoring
variation in the search process when the population becomes too similar.
```python
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
```
- **Number of Islands** – Defines how many subpopulations evolve independently in parallel.
A higher number of islands generally increases diversity and improves exploration of the search space.


- **Migration Interval** – Defines after how many generations individuals are exchanged between
islands. Regular migration enables information sharing between subpopulations while still
preserving independent search behavior.


- **Iterations** – Defines the number of migration cycles. The total number of generations is
calculated as `migration_interval * iterations`.


- **Migration Size** – Defines how many individuals are exchanged during each migration event.
This parameter controls the intensity of information transfer between islands.


- **Topology** – Defines the communication structure between islands, i.e. which islands are allowed
to exchange individuals. Available options are `fully_connected`, `ring`, `star`, and `random`.


- **Migration Strategy** – Defines which individuals are selected for migration. 
For example, migration can focus on the best-performing individuals, on diverse individuals, 
or on a dynamic strategy that adapts during the optimization process.


- **SA Island Active** – Activates an additional Simulated Annealing mechanism on island level. 
This can be used to improve local refinement of solutions and to support escaping local optima.


- **Rule Type** – Defines which operator rule set is applied. Depending on the selected rule type, 
different crossover and mutation probabilities are used.


- **Crossover Probability (`cxpb`)** – Defines the probability that two selected parent individuals
are recombined to generate offspring. Crossover is used to combine beneficial properties from 
different solutions.

- **Mutation Probability (`mutpb`)** – Defines the probability that an offspring is randomly 
modified after reproduction. Mutation helps preserve diversity and reduces the risk of 
premature convergence.

```python
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
```
- **Material Type** – Defines which material type is used to filter the dataset before the
optimization starts. This allows the algorithm to be applied only to a specific subset of the
available production data.


- **Validation Runs** – Defines how many times the algorithm is executed. Multiple runs can be
sed to evaluate the stability and robustness of the optimization results, since Genetic 
Algorithms may produce slightly different solutions in different runs.


- **Regression Model** – Defines which regression method is used to evaluate the fitness of a 
candidate solution. Currently, the implemented options are `OLS`, `PLS`, and `Ridge`.


- **Target Variable** – Defines which target variable is optimized and predicted by the model.
Currently available targets are `IB_AVG`, `DENSITY_AVG`, and `MOR_AVG`.

## Output

Each run produces one result folder per validation run
(e.g. `results/MOR_AVG_Seed42_Ridge_With_Rules/run1_pop75/`)
containing the following files.

---

### `results_run<N>_<date>.xlsx`

The main results file with three sections:

**Generation log** — one row per generation:

| Column | Description |
|---|---|
| `Gen` | Generation number |
| `Best Fitness` | Best fitness found so far (minimization, lower = better) |
| `Avg Diversity` | Average population diversity across all islands |

**Run settings** — configuration used for this run (target variable,
rule type, topology, migration strategy, population size, etc.).

**Best individual** — the final selected feature subset as a binary
vector (1 = selected, 0 = excluded), along with the regression
parameter, evaluation score and the total
number of selected features (`IND_SUM`).

---

### `best_fitness_over_generations_<TARGET>.png`

Plots the best fitness per island over all generations. The fitness
decreases as the GA minimizes the objective.

---

### `average_diversity_over_generations_<TARGET>.png`

Tracks the average population diversity per island over generations.
Diversity starts around 0.12 and gradually declines as the population
converges. Islands with rule-based mutation (`with_rules`) tend to
maintain higher diversity for longer, reducing the risk of premature
convergence.

---

### `population_size_over_generations_<TARGET>.png`

Shows the population size per island over time. With the default
configuration, population size remains **constant at 75 individuals
per island** throughout the entire run — no dynamic population
resizing is applied.

## Dashboard
## Dashboard

After the GA terminates, three log files are written to the results folder:

- `feature_freq_log.csv` — per-individual binary feature selection vectors,
  logged across all islands and generations
- `run_log.csv` — per-generation statistics: best fitness, average fitness,
  and normalized diversity per island
- `individuals_log.csv` — all evaluated individuals with their fitness scores

These files are consumed by `dashboard.py`, which launches an interactive
[Dash](https://dash.plotly.com/) application on `http://localhost:8051`.

### KPI Bar (top)

| Tile | Description |
|---|---|
| **Runtime** | Elapsed wall-clock time since the dashboard was started |
| **Generationen** | Current generation displayed |
| **#Lösungen** | Number of evaluated individuals out of total |
| **Score** | Best fitness achieved so far (minimization) |

### Charts

- **Best Fitness per Island** — decreasing fitness curves,
  one line per island, with interactive zoom and pan
- **Avg Fitness per Island** — average population fitness per island,
  showing convergence speed across islands
- **Diversity per Island** — normalized population diversity over time;
  declining diversity indicates increasing convergence
- **Feature Frequency (cumulative)** — bar chart of cumulative selection
  counts per feature, filterable by island via dropdown; group boundaries 
 are overlaid as colored background regions
- **Gene Selection Frequency** — heatmap of per-generation selection
  frequency for each feature index across all generations

### GA Parameters (right panel)

Displays the full hyperparameter configuration read directly from
`config.py` at startup: population size, number of islands, iterations,
migration interval and strategy, topology, crossover and mutation
probabilities, rule type, dataset size, and train/test split.

### Refresh

The dashboard polls all log files every **600 ms** by default
(`dcc.Interval`). This allows it to be used for **live monitoring**
during an active run as well as for post-hoc analysis after completion.

To launch:

```bash
python dashboard.py
```

