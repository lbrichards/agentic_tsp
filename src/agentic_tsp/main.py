# In src/agentic_tsp/main.py
import asyncio
import time
import numpy as np
from .core.data import get_berlin52_data, calculate_distance_matrix
from .orchestrator import run_hybrid_orchestration

def calculate_tour_length(tour: list, distance_matrix: np.ndarray) -> float:
    """Helper to calculate total distance of a tour."""
    if not tour: return float('inf')
    return sum(distance_matrix[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))

def main():
    """
    Main function to run the Agentic TSP system as a multi-run benchmark.
    """
    print("--- Agentic TSP Benchmark ---")
    print("Establishing a benchmark for token-based LLM collaboration.")
    
    # --- Configuration ---
    NUM_BENCHMARK_RUNS = 5
    NUM_WORKERS_PER_RUN = 10
    
    cities, optimal_length = get_berlin52_data()
    distance_matrix = calculate_distance_matrix(cities)
    dist_matrix_np = np.array(distance_matrix)
    
    print(f"\nLoaded 'berlin52' benchmark with {len(cities)} cities.")
    print(f"Known optimal tour length: {optimal_length:.2f}")
    print(f"Running {NUM_BENCHMARK_RUNS} full benchmark iterations...")
    
    all_classical_lengths = []
    all_llm_lengths = []
    total_start_time = time.time()

    for i in range(NUM_BENCHMARK_RUNS):
        print(f"\n--- Starting Benchmark Run {i+1}/{NUM_BENCHMARK_RUNS} ---")
        
        # Run the full hybrid orchestration
        classical_result, llm_tour = asyncio.run(run_hybrid_orchestration(distance_matrix, cities, NUM_WORKERS_PER_RUN))
        
        # Store the results for this run
        classical_length = classical_result.tour_length
        llm_length = calculate_tour_length(llm_tour, dist_matrix_np)
        
        all_classical_lengths.append(classical_length)
        all_llm_lengths.append(llm_length)
        
        print(f"Run {i+1} Result: Classical={classical_length:.2f}, LLM={llm_length:.2f}")

    total_end_time = time.time()
    
    # --- Final Benchmark Report ---
    print("\n\n--- Token-Based LLM Collaboration Benchmark Report ---")
    
    # Calculate statistics
    avg_classical = np.mean(all_classical_lengths)
    std_classical = np.std(all_classical_lengths)
    best_classical = np.min(all_classical_lengths)
    
    avg_llm = np.mean(all_llm_lengths)
    std_llm = np.std(all_llm_lengths)
    best_llm = np.min(all_llm_lengths)
    
    avg_total_time = (total_end_time - total_start_time) / NUM_BENCHMARK_RUNS

    print("\n--- Performance Statistics (lower is better) ---")
    print(f"                          | Classical GA       | LLM-Refined      |")
    print(f"--------------------------|--------------------|------------------|")
    print(f"Average Tour Length       | {avg_classical:<18.2f} | {avg_llm:<16.2f} |")
    print(f"Best Tour Length          | {best_classical:<18.2f} | {best_llm:<16.2f} |")
    print(f"Std Dev of Length         | {std_classical:<18.2f} | {std_llm:<16.2f} |")
    
    avg_error = ((avg_llm - optimal_length) / optimal_length) * 100
    print(f"\nAverage LLM-Refined result is {avg_error:.2f}% away from the optimal solution.")
    
    print("\n--- Cost Statistics ---")
    print(f"Average Computation Time per Run: {avg_total_time:.2f} seconds.")

    print("\nThis report concludes the benchmark for the token-based, natural language I/O channel.")

if __name__ == "__main__":
    main()