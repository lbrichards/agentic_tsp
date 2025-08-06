# In src/agentic_tsp/core/algorithms.py

import random
import numpy as np
from typing import List, Tuple

class GeneticAlgorithmTSP:
    """Genetic Algorithm implementation for TSP."""

    def __init__(self, cities_count: int, population_size: int = 100, generations: int = 1000, mutation_rate: float = 0.01, seed: int = None):
        """Initialize GA parameters."""
        self.cities_count = cities_count
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

    def _calculate_tour_length(self, tour: List[int], distance_matrix: np.ndarray) -> float:
        """Calculates the total distance of a single tour."""
        total_distance = 0
        for i in range(len(tour)):
            # Get the distance between the current city and the next city
            # The modulo operator (%) handles the wrap-around from the last city to the first
            total_distance += distance_matrix[tour[i], tour[(i + 1) % len(tour)]]
        return total_distance

    def solve(self, distance_matrix: List[List[float]]) -> Tuple[List[int], float]:
        """Solves the TSP using a genetic algorithm."""
        # Convert list of lists to a numpy array for efficient calculations
        dist_matrix_np = np.array(distance_matrix)
        num_cities = self.cities_count
        
        # Near the top of the solve method
        elite_size = 1  # We will carry over the single best individual

        # 1. Initialize Population
        # Create a list of `self.population_size` random tours.
        # A tour is a shuffled list of city indices, e.g., [0, 1, 2, ..., num_cities-1]
        population = [random.sample(range(num_cities), num_cities) for _ in range(self.population_size)]

        best_tour = None
        best_tour_length = float('inf')

        for generation in range(self.generations):
            # 2. Fitness Evaluation
            # Calculate the length of each tour in the population.
            # Store them as a list of (tour, length) tuples.
            fitness_scores = [(tour, self._calculate_tour_length(tour, dist_matrix_np)) for tour in population]

            # 3. Selection (Tournament Selection is a good choice)
            # Create a new population by selecting the fittest individuals.
            parents = []
            for _ in range(self.population_size):
                # Select a few random individuals from the population for the "tournament"
                tournament_size = 5
                contenders = random.sample(fitness_scores, tournament_size)
                # The winner of the tournament is the one with the shortest tour length
                winner = min(contenders, key=lambda x: x[1])
                parents.append(winner[0]) # Add the winning tour to the list of parents

            # 4. Crossover (Ordered Crossover - OX1 is a good choice for TSP)
            offspring_population = []
            for i in range(0, self.population_size, 2):
                parent1 = parents[i]
                # Ensure we have a second parent, handle odd population size
                parent2 = parents[i+1] if i+1 < self.population_size else parents[0]

                # --- Start of Crossover Logic ---
                start, end = sorted(random.sample(range(num_cities), 2))
                child1 = [None] * num_cities
                # Copy the segment from parent1 to child1
                child1[start:end+1] = parent1[start:end+1]
                # Fill the rest of child1 with genes from parent2
                p2_genes = [gene for gene in parent2 if gene not in child1]
                child1_idx = (end + 1) % num_cities
                for gene in p2_genes:
                    if child1[child1_idx] is None:
                        child1[child1_idx] = gene
                        child1_idx = (child1_idx + 1) % num_cities
                # --- End of Crossover Logic ---
                offspring_population.append(child1)
            
            # Just before the mutation loop
            # --- Start of Adaptive Mutation Logic ---
            explore_phase_generations = self.generations * 0.7
            if generation < explore_phase_generations:
                current_mutation_rate = self.mutation_rate  # High rate for exploration
            else:
                current_mutation_rate = self.mutation_rate / 5.0  # Lower rate for exploitation
            # --- End of Adaptive Mutation Logic ---
            
            # 5. Mutation (Swap Mutation)
            for tour in offspring_population:
                # Use the new adaptive rate here
                if random.random() < current_mutation_rate:
                    # Select two distinct random indices to swap
                    idx1, idx2 = random.sample(range(num_cities), 2)
                    tour[idx1], tour[idx2] = tour[idx2], tour[idx1]

            # 6. Replacement with Elitism
            # First, sort the current population by fitness (tour length)
            sorted_population = sorted(fitness_scores, key=lambda x: x[1])
            
            # The new population starts with the "elite" individuals from the old generation
            new_population = [tour for tour, length in sorted_population[:elite_size]]
            
            # Fill the rest of the new population with the best offspring
            # The number of offspring to add is the total population size minus the elite size
            num_offspring_to_add = self.population_size - elite_size
            new_population.extend(offspring_population[:num_offspring_to_add])
            
            population = new_population

            # Keep track of the best tour found so far
            current_best_tour, current_best_length = min(fitness_scores, key=lambda x: x[1])
            if current_best_length < best_tour_length:
                best_tour = current_best_tour
                best_tour_length = current_best_length
                # Only print every 100 generations or significant improvements (>5% better)
                if generation % 100 == 0 or (best_tour_length > 0 and (1 - current_best_length/best_tour_length) > 0.05):
                    print(f"Worker {self.seed}: Generation {generation}: New best: {best_tour_length:.2f}")

        return best_tour, best_tour_length