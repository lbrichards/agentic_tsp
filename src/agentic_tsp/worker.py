# In src/agentic_tsp/worker.py
from .core.schemas import MacroCall, WorkerResult
from .core.algorithms import GeneticAlgorithmTSP

def execute_worker_task(call: MacroCall) -> WorkerResult:
    """
    Receives a MacroCall, executes the specified algorithm,
    and returns a WorkerResult.
    """
    # Logic to select algorithm based on call.algorithm will go here.
    # For now, we only have one.
    ga = GeneticAlgorithmTSP(
        cities_count=len(call.distance_matrix),
        seed=call.seed,
        **call.parameters
    )
    best_tour, tour_length = ga.solve(call.distance_matrix)

    result = WorkerResult(
        task_id=call.task_id,
        best_tour=best_tour,
        tour_length=tour_length
    )
    return result