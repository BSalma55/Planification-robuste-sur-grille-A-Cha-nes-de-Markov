# -*- coding: utf-8 -*-


import argparse
import sys

from Grid       import Grid
from astar      import AStar                          # FIX 1
from markov     import MarkovChain
from simulation import MarkovSimulation

from experiments import (
    figure_1,                                          # FIX 2
    figure_2 as experiment_1_compare_algorithms,
    figure_3 as experiment_2_vary_epsilon,
    figure_4 as experiment_3_compare_heuristics,
    figure_5 as experiment_4_weighted_astar,
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

    grid  = Grid(5, 5, obstacles=[(2, 2)])
    start = (0, 0)
    goal  = (4, 4)
    eps   = 0.2

    # Phase 2 : Planification A*
    print("\nPhase 2 -- Planification avec A*")
    a      = AStar(grid, heuristic='manhattan')
    result = a.search(start, goal, algorithm='astar')

    if not result['success']:
        print("  ERREUR : aucun chemin trouve.")
        return

    print(f"  Chemin optimal  : {result['path']}")
    print(f"  Cout            : {result['cost']}")
    print(f"  Noeuds explores : {result['nodes_expanded']}")

    # Phase 3 & 4 : Chaine de Markov
    print(f"\nPhase 3 & 4 -- Chaine de Markov (epsilon={eps})")
    policy     = a.extract_policy(result['path'])
    absorbing  = {goal: 'GOAL', (-1, -1): 'FAIL'}
    all_states = set(result['path']) | {(-1, -1)}

    mc = MarkovChain(all_states, absorbing)
    mc.build_from_policy(policy, grid, epsilon=eps)

    # FIX 3 : affichage P(GOAL) et E[T] theoriques
    analysis = mc.absorption_analysis()
    if analysis:
        i0 = mc.state_to_idx.get(start)
        if i0 is not None and i0 in analysis['trans_indices']:
            loc = analysis['trans_indices'].index(i0)
            for j, ai in enumerate(analysis['abs_indices']):
                s   = mc.idx_to_state[ai]
                typ = mc.absorbing_states.get(s, '?')
                print(f"  P(-> {s} [{typ}]) theorique = "
                      f"{analysis['B'][loc, j]:.4f}")
            print(f"  E[T] theorique  = {analysis['t'][loc]:.2f} etapes")
    else:
        print("  Analyse d'absorption non disponible.")

    # Phase 5 : Simulation Monte-Carlo
    print("\nPhase 5 -- Simulation Monte-Carlo")
    n_sims = 1000
    sim    = MarkovSimulation(mc)
    stats  = sim.run_simulations(start, n_simulations=n_sims)

    # FIX 4 : protection division par zero
    success_rate = (stats['goal_count'] / n_sims * 100) if n_sims > 0 else 0.0
    print(f"  Succes empirique : {success_rate:.1f}%")

    if stats['times_to_goal']:
        t_emp = sum(stats['times_to_goal']) / len(stats['times_to_goal'])
        print(f"  E[T] empirique   : {t_emp:.2f} etapes")


# ----------------------------------------------------------------
# POINT D'ENTREE CLI
# ----------------------------------------------------------------
def main():
    print(BANNER)

    parser = argparse.ArgumentParser(description="Mini-projet IA")
    parser.add_argument(
        '--exp', default='0',
        help=("0: Demo | 1: Figure2 UCS/Greedy/A* | "
              "2: Figure3 epsilon | 3: Figure4 heuristiques | "
              "4: Figure5 Weighted A* | all: Tout"))
    args = parser.parse_args()

    dispatch = {
        '0': demo,
        '1': experiment_1_compare_algorithms,
        '2': experiment_2_vary_epsilon,
        '3': experiment_3_compare_heuristics,
        '4': experiment_4_weighted_astar,
    }

    if args.exp == 'all':
        print("\nLancement de la Figure 1...")
        figure_1()                                     # FIX 2
        for key in ['1', '2', '3', '4']:
            print(f"\nLancement de l'Experience {key}...")
            dispatch[key]()
    elif args.exp in dispatch:
        dispatch[args.exp]()
    else:
        print(f"Option inconnue : {args.exp}")
        print("Options valides : 0, 1, 2, 3, 4, all")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("FIN DE L'EXECUTION")
    print("=" * 60)


if __name__ == "__main__":
    main()