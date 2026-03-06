# -*- coding: utf-8 -*-
"""
========================================================
main.py  --  Point d'entree principal du projet
========================================================
"""

import argparse
import sys

# Importations nettoyees (suppression des espaces insecables)
from Grid import Grid
from astar import AStar, compare_algorithms
from markov import MarkovChain
from simulation import MarkovSimulation

# Mise a jour des imports selon vos figures
from experiments import (
    figure_2 as experiment_1_compare_algorithms, # UCS/Greedy/A* est en Figure 2
    figure_3 as experiment_2_vary_epsilon,       # Impact incertitude est en Figure 3
    figure_4 as experiment_3_compare_heuristics, # h=0 vs Manhattan est en Figure 4
    figure_5 as experiment_4_weighted_astar      # Weighted A* est en Figure 5
)

BANNER = """
+==========================================================+
|   MINI-PROJET -- PLANIFICATION ROBUSTE SUR GRILLE        |
|   A* + Chaines de Markov                                 |
|   Bases de l'Intelligence Artificielle | 2025-2026       |
+==========================================================+
"""

# ----------------------------------------------------------------
# DEMONSTRATION COMPLETE
# ----------------------------------------------------------------
def demo():
    sep = "=" * 60
    print(f"\n{sep}")
    print("DEMONSTRATION COMPLETE -- GRILLE FACILE 5x5")
    print(sep)

    # Configuration par defaut
    grid  = Grid(5, 5, obstacles=[(2, 2)])
    start = (0, 0)
    goal  = (4, 4)
    eps   = 0.2

    # -- Phase 2 : Planification A*
    print(f"\nPhase 2 -- Planification avec A*")
    a      = AStar(grid, heuristic='manhattan')
    result = a.search(start, goal, algorithm='astar')

    if not result['success']:
        print("  ERREUR : aucun chemin trouve.")
        return

    print(f"  Chemin optimal : {result['path']}")

    # -- Phase 3 & 4 : Markov
    print(f"\nPhase 3 & 4 -- Chaine de Markov")
    policy     = a.extract_policy(result['path'])
    # Definition des etats absorbants pour l'analyse
    absorbing  = {goal: 'GOAL', (-1,-1): 'FAIL'}
    all_states = set(result['path']) | {(-1,-1)}

    mc = MarkovChain(all_states, absorbing)
    mc.build_from_policy(policy, grid, epsilon=eps)
    
    # Analyse de periodicite et d'absorption
    if hasattr(mc, 'absorption_analysis'):
        analysis = mc.absorption_analysis()
        print("  Analyse d'absorption terminee.")

    # -- Phase 5 : Simulation Monte-Carlo
    print(f"\nPhase 5 -- Simulation Monte-Carlo")
    n_sims = 1000
    sim   = MarkovSimulation(mc)
    stats = sim.run_simulations(start, n_simulations=n_sims)
    
    # Correction du calcul du taux de succes pour n_simulations
    success_rate = (stats['goal_count'] / n_sims) * 100
    print(f"  Succes : {success_rate}%")

# ----------------------------------------------------------------
# POINT D'ENTREE CLI
# ----------------------------------------------------------------
def main():
    print(BANNER)

    parser = argparse.ArgumentParser(description="Mini-projet IA")
    parser.add_argument('--exp', default='0', help="0: Demo, 1-4: Exp specifiques, all: Tout")
    args = parser.parse_args()

    dispatch = {
        '0':   demo,
        '1':   experiment_1_compare_algorithms,
        '2':   experiment_2_vary_epsilon,
        '3':   experiment_3_compare_heuristics,
        '4':   experiment_4_weighted_astar,
    }

    if args.exp == 'all':
        for key in ['1', '2', '3', '4']:
            print(f"\nLancement de l'Experience {key}...")
            dispatch[key]()
    elif args.exp in dispatch:
        dispatch[args.exp]()
    else:
        print(f"Option inconnue : {args.exp}")

    print("\n" + "="*60)
    print("FIN DE L'EXECUTION")
    print("="*60)

if __name__ == "__main__":
    main()