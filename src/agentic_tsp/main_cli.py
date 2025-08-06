# In src/agentic_tsp/main_cli.py
import asyncio
import time
import numpy as np
import argparse
from .core.data import get_berlin52_data, calculate_distance_matrix
from .orchestrator import run_hybrid_orchestration

def calculate_tour_length(tour: list, distance_matrix: np.ndarray) -> float:
    """Helper to calculate total distance of a tour."""
    if not tour: return float('inf')
    return sum(distance_matrix[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))

def main():
    """
    Main function with CLI arguments for configurable runs.
    """
    parser = argparse.ArgumentParser(description='Agentic TSP Solver with configurable parameters')
    parser.add_argument('--workers', type=int, default=10, 
                        help='Number of GA workers per run (default: 10)')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of benchmark runs (default: 5)')
    parser.add_argument('--llm-model', type=str, default='gpt-4o',
                        choices=['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini'],
                        help='LLM model to use for refinement (default: gpt-4o)')
    parser.add_argument('--generations', type=int, default=1500,
                        help='Number of generations for GA (default: 1500)')
    parser.add_argument('--population', type=int, default=100,
                        help='Population size for GA (default: 100)')
    parser.add_argument('--mutation-rate', type=float, default=0.05,
                        help='Initial mutation rate for GA (default: 0.05)')
    parser.add_argument('--no-llm', action='store_true',
                        help='Skip LLM refinement stage')
    
    args = parser.parse_args()
    
    print("--- Agentic TSP Benchmark ---")
    print(f"Configuration:")
    print(f"  Workers: {args.workers}")
    print(f"  Runs: {args.runs}")
    print(f"  LLM Model: {'None (skipped)' if args.no_llm else args.llm_model}")
    print(f"  GA Generations: {args.generations}")
    print(f"  GA Population: {args.population}")
    print(f"  GA Mutation Rate: {args.mutation_rate}")
    print()
    
    # Store configuration for workers
    import os
    os.environ['TSP_LLM_MODEL'] = args.llm_model
    os.environ['TSP_GA_GENERATIONS'] = str(args.generations)
    os.environ['TSP_GA_POPULATION'] = str(args.population)
    os.environ['TSP_GA_MUTATION'] = str(args.mutation_rate)
    os.environ['TSP_NO_LLM'] = '1' if args.no_llm else '0'
    
    cities, optimal_length = get_berlin52_data()
    distance_matrix = calculate_distance_matrix(cities)
    dist_matrix_np = np.array(distance_matrix)
    
    print(f"Loaded 'berlin52' benchmark with {len(cities)} cities.")
    print(f"Known optimal tour length: {optimal_length:.2f}")
    print(f"Running {args.runs} benchmark iterations with {args.workers} workers each...")
    
    all_classical_lengths = []
    all_llm_lengths = []
    total_start_time = time.time()

    for i in range(args.runs):
        print(f"\n--- Starting Benchmark Run {i+1}/{args.runs} ---")
        
        # Run the full hybrid orchestration
        classical_result, llm_tour = asyncio.run(
            run_hybrid_orchestration(distance_matrix, cities, args.workers)
        )
        
        # Store the results for this run
        classical_length = classical_result.tour_length
        llm_length = calculate_tour_length(llm_tour, dist_matrix_np)
        
        all_classical_lengths.append(classical_length)
        all_llm_lengths.append(llm_length)
        
        if args.no_llm:
            print(f"Run {i+1} Result: Classical={classical_length:.2f} (LLM skipped)")
        else:
            print(f"Run {i+1} Result: Classical={classical_length:.2f}, LLM={llm_length:.2f}")

    total_end_time = time.time()
    
    # --- Final Benchmark Report ---
    print("\n\n--- Benchmark Report ---")
    
    # Calculate statistics
    avg_classical = np.mean(all_classical_lengths)
    std_classical = np.std(all_classical_lengths)
    best_classical = np.min(all_classical_lengths)
    
    print("\n--- Performance Statistics (lower is better) ---")
    
    if not args.no_llm:
        avg_llm = np.mean(all_llm_lengths)
        std_llm = np.std(all_llm_lengths)
        best_llm = np.min(all_llm_lengths)
        
        print(f"                          | Classical GA       | LLM-Refined ({args.llm_model}) |")
        print(f"--------------------------|--------------------|---------------------------|")
        print(f"Average Tour Length       | {avg_classical:<18.2f} | {avg_llm:<25.2f} |")
        print(f"Best Tour Length          | {best_classical:<18.2f} | {best_llm:<25.2f} |")
        print(f"Std Dev of Length         | {std_classical:<18.2f} | {std_llm:<25.2f} |")
        
        avg_error = ((avg_llm - optimal_length) / optimal_length) * 100
        print(f"\nAverage LLM-Refined result is {avg_error:.2f}% away from optimal.")
    else:
        print(f"Classical GA Results:")
        print(f"  Average Tour Length: {avg_classical:.2f}")
        print(f"  Best Tour Length: {best_classical:.2f}")
        print(f"  Std Dev of Length: {std_classical:.2f}")
        
        avg_error = ((avg_classical - optimal_length) / optimal_length) * 100
        print(f"\nAverage result is {avg_error:.2f}% away from optimal.")
    
    avg_total_time = (total_end_time - total_start_time) / args.runs
    print(f"\n--- Cost Statistics ---")
    print(f"Average Computation Time per Run: {avg_total_time:.2f} seconds.")

if __name__ == "__main__":
    main()