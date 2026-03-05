#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
from astar import Grid, AStar, compare_algorithms
from markov import MarkovChain
from simulation import MarkovSimulation
from experiments import (
    experiment_1_compare_algorithms,
    experiment_2_vary_epsilon,
    experiment_3_compare_heuristics,
    experiment_4_weighted_astar
)


def demo_simple():
    """Démonstration simple du fonctionnement"""
    
    print("\n" + "="*70)
    print("DÉMONSTRATION SIMPLE")
    print("="*70)
    
    # Créer une grille simple
    grid = Grid(5, 5, obstacles=[(1, 1), (2, 2), (3, 3)])
    start = (0, 0)
    goal = (4, 4)
    
    print(f"\nGrille: {grid.width}x{grid.height}")
    print(f"Obstacles: {grid.obstacles}")
    print(f"Départ: {start}, But: {goal}")
    
    # 1. Planification avec A*
    print("\n--- Étape 1: Planification avec A* ---")
    astar = AStar(grid)
    result = astar.search(start, goal, algorithm='astar')
    
    if result['success']:
        print(f"Chemin trouvé: {result['path']}")
        print(f"Coût: {result['cost']}")
        print(f"Nœuds développés: {result['nodes_expanded']}")
        
        # Extraire la politique
        policy = astar.extract_policy(result['path'])
        print(f"\nPolitique extraite ({len(policy)} états)")
        
        # 2. Construction de la chaîne de Markov
        print("\n--- Étape 2: Construction de la chaîne de Markov ---")
        
        # États d'échec (obstacles)
        fail_states = grid.obstacles
        
        # Ajouter GOAL et FAIL comme états absorbants
        absorbing = {goal: 'GOAL', (-1, -1): 'FAIL'}
        
        # Tous les états
        all_states = set(result['path']) | set(fail_states) | {(-1, -1)}
        mc = MarkovChain(all_states, absorbing)
        
        # Construire avec incertitude ε=0.2
        epsilon = 0.2
        mc.build_from_policy(policy, grid, epsilon, fail_states)
        
        print(f"Matrice P construite (taille: {mc.n}x{mc.n})")
        print(f"Stochastique: {mc.is_stochastic()}")
        
        # 3. Analyse
        print("\n--- Étape 3: Analyse ---")
        mc.print_analysis(start)
        
        # 4. Simulation
        print("\n--- Étape 4: Simulation Monte-Carlo ---")
        sim = MarkovSimulation(mc)
        stats = sim.run_simulations(start, n_simulations=500, max_steps=20)
        sim.analyze_simulations(stats, 500)
        
    else:
        print("Aucun chemin trouvé!")


def main():
    parser = argparse.ArgumentParser(description='Mini-projet IA: A* + Markov')
    parser.add_argument('--experiment', type=int, choices=[0, 1, 2, 3, 4], default=0,
                       help='Numéro de l\'expérience (0: démo, 1-4: expériences)')
    parser.add_argument('--epsilon', type=float, default=0.2,
                       help='Taux d\'incertitude pour la démo')
    
    args = parser.parse_args()
    
    print("="*70)
    print("MINI-PROJET : PLANIFICATION ROBUSTE SUR GRILLE")
    print("A* + Chaînes de Markov")
    print("="*70)
    print(f"Date: 2026")
    print("="*70)
    
    if args.experiment == 0:
        demo_simple()
    
    elif args.experiment == 1:
        print("\nExécution de l'expérience 1: Comparaison des algorithmes\n")
        experiment_1_compare_algorithms()
    
    elif args.experiment == 2:
        print("\nExécution de l'expérience 2: Impact de l'incertitude ε\n")
        experiment_2_vary_epsilon()
    
    elif args.experiment == 3:
        print("\nExécution de l'expérience 3: Comparaison des heuristiques\n")
        experiment_3_compare_heuristics()
    
    elif args.experiment == 4:
        print("\nExécution de l'expérience 4: Weighted A*\n")
        experiment_4_weighted_astar()
    
    print("\n" + "="*70)
    print("FIN DE L'EXÉCUTION")
    print("="*70)


if __name__ == "__main__":
    main()