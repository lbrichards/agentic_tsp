# Agentic TSP Solver
![alt text](https://img.shields.io/badge/python-3.10+-blue.svg)
![alt text](https://img.shields.io/badge/License-MIT-yellow.svg)

This repository provides a framework for exploring a key question in modern AI: **Can a Large Language Model (LLM) effectively collaborate with classical algorithms to solve a complex optimization problem?**

We use the well-known Traveling Salesperson Problem (TSP) as a testbed. The system implements a configurable, hybrid, two-stage agentic architecture to find solutions for the berlin52 benchmark problem. This framework is designed to produce replicable benchmarks, demonstrating that the success of algorithm-AI collaboration is highly dependent on the LLM's capability and the scale of the computational effort.

## Core Concept: A Hybrid Agentic Architecture

The solver orchestrates a collaboration between two different types of agents in a two-stage process:

### Stage 1: Classical Optimization (The Workhorses)
A configurable pool of independent Genetic Algorithm (GA) workers runs concurrently. Each worker evolves a population of solutions, applying selection, crossover, and mutation. This stage performs the heavy computational lifting to find a strong candidate solution.

### Stage 2: LLM Refinement (The Heuristic Expert)
The single best tour from Stage 1 is handed off to an LLM agent. The LLM is prompted with the tour and context on TSP heuristics. Its task is to analyze the tour and propose a refinement based on this high-level guidance.

This architecture allows us to measure the value added by different LLMs when they attempt to improve upon the work of a powerful, specialized algorithm.

## Getting Started

### 1. Installation

First, clone the repository and set up a Python virtual environment.

```bash
git clone git@github.com:lbrichards/agentic_tsp.git
cd agentic_tsp
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. API Key Configuration

The system requires an OpenAI API key. A helper script is provided to set this up securely. Run the following command and you will be prompted to paste your API key:

```bash
python src/setup_env.py
```

This creates a `.env` file that is ignored by Git and should never be committed.

## Usage: Running Experiments via CLI

The project includes a command-line interface (`main_cli.py`) for flexible experimentation.

### Help Menu

To see all available options, run:

```bash
python -m src.agentic_tsp.main_cli --help
```

### Common Examples

```bash
# High-performance run with 40 workers and the default gpt-4o model
python -m src.agentic_tsp.main_cli --workers 40 --runs 1

# Quick test run
python -m src.agentic_tsp.main_cli --runs 1

# Compare a different LLM (e.g., gpt-3.5-turbo)
python -m src.agentic_tsp.main_cli --llm-model gpt-3.5-turbo

# Run the classical algorithm only, skipping the LLM refinement
python -m src.agentic_tsp.main_cli --no-llm

# A fully custom configuration
python -m src.agentic_tsp.main_cli --workers 20 --runs 10 --generations 2000
```

## Empirical Results: A Tale of Two Benchmarks

Our experiments reveal that the value of LLM collaboration is not absolute. It is a function of both the LLM's inherent capability and the quality of the problem state it is given.

### Benchmark A: The Baseline with a Standard LLM

This benchmark used a moderate number of classical workers and a standard, cost-effective LLM.

**Configuration:** 10 GA Workers, gpt-3.5-turbo model.

**Result:** The LLM agent did not reliably improve upon the already strong baseline provided by the classical algorithms.

|                    | Classical GA | LLM-Refined (gpt-3.5-turbo) |
|--------------------|-------------|------------------------------|
| Average Tour Length | 8026.85    | 8041.81                      |
| Best Tour Length   | 7904.52     | 7904.52                      |

### Benchmark B: High-Performance Collaboration

This benchmark scaled up the classical compute and used a state-of-the-art LLM.

**Configuration:** 40 GA Workers, gpt-4o model.

**Result:** A significant and measurable improvement was achieved. The more capable LLM was able to identify a sophisticated refinement that the classical algorithms had missed, improving the tour by 94.65 units.

| Result                   | Distance    | from Optimal |
|--------------------------|-------------|--------------|
| Best Classical GA Tour   | 7788.74     | 3.3%        |
| LLM-Refined Tour (gpt-4o)| 7694.09     | 2.0%        |

## Key Takeaway

A token-based, natural-language collaboration between algorithms and AI agents can be effective, but success is not guaranteed. Its value is highly dependent on:

1. **The Reasoning Power of the LLM:** More capable models like gpt-4o can find valuable improvements where less advanced models cannot.

2. **The Quality of the Input:** Providing the LLM with a highly optimized starting point (achieved here by scaling to 40 workers) allows it to focus on complex, final-stage refinements.

This framework provides a robust tool for further research into these dynamic collaborations.

## License

This project is licensed under the Apache License 2.0.