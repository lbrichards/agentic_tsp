# In src/agentic_tsp/orchestrator.py
import asyncio
from typing import List, Tuple
from .core.schemas import MacroCall, WorkerResult
from .worker import execute_worker_task
from .llm_worker import llm_improve_tour

async def run_orchestration(distance_matrix: List[List[float]], num_workers: int) -> WorkerResult:
    """
    Main orchestration logic. Creates and dispatches tasks to workers,
    then aggregates the results to find the best one.
    """
    # Create macro calls for each worker, potentially with different seeds
    import os
    generations = int(os.getenv('TSP_GA_GENERATIONS', '1500'))
    population_size = int(os.getenv('TSP_GA_POPULATION', '100'))
    mutation_rate = float(os.getenv('TSP_GA_MUTATION', '0.05'))
    
    calls = [
        MacroCall(
            algorithm="GA_TSP",
            distance_matrix=distance_matrix,
            parameters={
                "generations": generations,
                "population_size": population_size,
                "mutation_rate": mutation_rate
            },
            seed=i
        ) for i in range(num_workers)
    ]

    # In a real-world scenario, you might run these in parallel processes
    # For now, we run them concurrently using asyncio
    tasks = [asyncio.to_thread(execute_worker_task, call) for call in calls]
    results: List[WorkerResult] = await asyncio.gather(*tasks)

    # Find the best result among all workers
    best_result = min(results, key=lambda r: r.tour_length)
    return best_result

async def run_hybrid_orchestration(distance_matrix: List[List[float]], cities: List[Tuple[float, float]], num_workers: int):
    """
    Runs a two-stage orchestration:
    1. Classical GA workers find a good candidate solution.
    2. An LLM worker attempts to refine the best solution from stage 1.
    """
    # --- Stage 1: Classical GA ---
    print("\n--- Stage 1: Running Classical Genetic Algorithm Workers ---")
    classical_best_result = await run_orchestration(distance_matrix, num_workers)
    
    # --- Stage 2: LLM Refinement ---
    import os
    if os.getenv('TSP_NO_LLM', '0') == '1':
        print("\n--- Stage 2: LLM Refinement (SKIPPED) ---")
        llm_improved_tour = classical_best_result.best_tour
    else:
        print("\n--- Stage 2: Handing off best tour to LLM for refinement ---")
        candidate_tour = classical_best_result.best_tour
        
        # This is a synchronous call, but we wrap it for consistency
        llm_improved_tour = await asyncio.to_thread(llm_improve_tour, candidate_tour, cities)
    
    return classical_best_result, llm_improved_tour