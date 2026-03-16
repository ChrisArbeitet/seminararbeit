from queue import Queue, Empty
from rich.live import Live
from rich.table import Table
import csv
import os
from datetime import datetime

LOG_PATH      = "logs/run_log.csv"
FREQ_LOG_PATH = "logs/feature_freq_log.csv"  # ✅ Neu


# ── RUN-LOG ──────────────────────────────────────────────────────────────────
def init_log_file():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "run_id",
                "target",
                "island_id",
                "generation",
                "population_size",
                "best_fitness",
                "avg_fitness",
                "normalized_diversity",
                "status",
            ])


def append_log_row(run_id, target, island_id, entry):
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            run_id,
            target,
            island_id,
            entry.get("generation"),
            entry.get("population_size"),
            entry.get("best_fitness"),
            entry.get("avg_fitness"),
            entry.get("normalized_diversity"),
            entry.get("status"),
        ])


# ── FEATURE-HÄUFIGKEITS-LOG ──────────────────────────────────────────────────
def init_feature_freq_log(num_features):
    """Einmalig vor GA-Start aufrufen mit Anzahl der Features."""
    os.makedirs(os.path.dirname(FREQ_LOG_PATH), exist_ok=True)
    if not os.path.exists(FREQ_LOG_PATH):
        with open(FREQ_LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp", "run_id", "island_id", "generation"] +
                [f"f{i}" for i in range(num_features)]
            )


def append_feature_freq_row(run_id, island_id, generation, population):
    """
    Schreibt eine Zeile pro (island_id, generation) mit der Selektionshäufigkeit
    jedes Features über alle Individuen der Population.
    population: Liste von Individuen, jedes Individuum = binäre Liste [0,1,0,...]
    """
    if not population:
        return
    num_features = len(population[0])
    counts = [sum(int(ind[i]) for ind in population) for i in range(num_features)]
    with open(FREQ_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [datetime.utcnow().isoformat(), run_id, island_id, generation] + counts
        )


# ── RICH TABLE ───────────────────────────────────────────────────────────────
def create_table(status, num_islands, target):
    table = Table(title=f"Status of Islands for Target: {target}")
    table.add_column("Island",               justify="right")
    table.add_column("Generation",           justify="right")
    table.add_column("Population Size",      justify="right")
    table.add_column("Best Fitness",         justify="right")
    table.add_column("Average Fitness",      justify="right")
    table.add_column("Normalized Diversity", justify="right")
    table.add_column("Status",               justify="right")

    for island_id in range(num_islands):
        if island_id in status:
            gen                  = status[island_id]["generation"]
            pop_size             = status[island_id]["population_size"]
            fitness              = f"{status[island_id]['best_fitness']:.4f}"
            avg_fitness          = f"{status[island_id]['avg_fitness']:.4f}"
            normalized_diversity = f"{status[island_id]['normalized_diversity']:.6f}"
            status_str           = status[island_id]["status"]
        else:
            gen = pop_size = fitness = avg_fitness = normalized_diversity = status_str = "-"

        table.add_row(
            str(island_id), str(gen), str(pop_size),
            fitness, avg_fitness, normalized_diversity, status_str
        )
    return table


# ── MONITOR ──────────────────────────────────────────────────────────────────
def monitor_progress(log_queue, num_islands, target, stop_event, run_id="run_1"):
    status            = {}
    freq_log_init     = False   # ✅ Wird beim ersten population-Eintrag initialisiert
    init_log_file()

    with Live(create_table(status, num_islands, target), refresh_per_second=1) as live:
        while not stop_event.is_set():
            try:
                msg = log_queue.get(timeout=0.05)

                island_id = msg["island_id"]
                if msg.get("done"):
                    continue

                if island_id not in status:
                    status[island_id] = {
                        "generation":           None,
                        "population_size":      None,
                        "best_fitness":         None,
                        "avg_fitness":          None,
                        "normalized_diversity": None,
                        "status":               None,
                    }

                for key in msg:
                    if key != "island_id":
                        status[island_id][key] = msg[key]

                # Bestehend: Run-Log schreiben
                if status[island_id]["generation"] is not None:
                    append_log_row(run_id, target, island_id, status[island_id])

                    # ✅ Neu: Feature-Häufigkeit schreiben wenn population mitgeschickt
                    population = msg.get("population")
                    if population is not None:
                        if not freq_log_init:
                            init_feature_freq_log(len(population[0]))
                            freq_log_init = True
                        append_feature_freq_row(
                            run_id,
                            island_id,
                            status[island_id]["generation"],
                            population
                        )

                live.update(create_table(status, num_islands, target))

            except Empty:
                continue
