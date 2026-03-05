#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================
main.py  --  Point d'entree principal du projet
========================================================
Mini-projet : Planification robuste sur grille
              A* + Chaines de Markov
Annee       : 2025-2026

Usage
-----
  python main.py               # demonstration complete
  python main.py --exp 1       # E.1 : comparaison algorithmes + sauvegarde photo
  python main.py --exp 2       # E.2 : variation de epsilon + sauvegarde photo
  python main.py --exp 3       # E.3 : comparaison heuristiques + sauvegarde photo
  python main.py --exp 4       # E.4 : Weighted A* (option) + sauvegarde photo
  python main.py --exp all     # toutes les experiences + sauvegarde photos
  python main.py --exp 0       # demonstration complete (sans sauvegarde)

CDC ?5 (Livrables) :
  ? Code Python structure en modules
  ? A* + variantes, matrice P stochastique
  ? Calculs Markov (P^n, pi(n)), absorption, simulation Monte-Carlo
"""

import argparse
import sys

from astar       import Grid, AStar, compare_algorithms
from markov      import MarkovChain
from simulation  import MarkovSimulation
from experiments import (
    experiment_1_compare_algorithms,
    experiment_2_vary_epsilon,
    experiment_3_compare_heuristics,
    experiment_4_weighted_astar,
)

BANNER = """
+==========================================================+
|   MINI-PROJET -- PLANIFICATION ROBUSTE SUR GRILLE         |
|   A* + Chaines de Markov                                 |
|   Bases de l'Intelligence Artificielle | 2025-2026       |
+==========================================================+
"""


# ----------------------------------------------------------------
# DEMONSTRATION COMPLETE  (toutes les phases du CDC)
# ----------------------------------------------------------------
def demo():
    """
    Execute toutes les phases du CDC sur la grille Facile 5x5 :
      Phase 2 : Planification avec A*
      Phase 3 : Construction de P
      Phase 4 : Analyse Markov (classes, absorption, periodicite)
      Phase 5 : Simulation Monte-Carlo + comparaison theorie
    """
    sep = "=" * 60

    print(f"\n{sep}")
    print("DEMONSTRATION COMPLETE -- GRILLE FACILE 5x5")
    print(sep)

    # -- Phase 1 : Grille (CDC ?4.1) ------------------------------
    grid  = Grid(5, 5, obstacles=[(2, 2)])
    start = (0, 0)
    goal  = (4, 4)
    eps   = 0.2

    print(f"\nPhase 1 -- Grille {grid.width}x{grid.height} | "
          f"obstacles={grid.obstacles} | start={start} | goal={goal} | epsilon={eps}")

    # -- Phase 2 : Planification A* (CDC ?4.2) --------------------
    print(f"\n{sep}\nPhase 2 -- Planification avec A*\n{sep}")
    a      = AStar(grid, heuristic='manhattan')
    result = a.search(start, goal, algorithm='astar')

    if not result['success']:
        print("  ERREUR : aucun chemin trouve.")
        return

    print(f"  Chemin optimal : {result['path']}")
    print(f"  Cout           : {result['cost']}")
    print(f"  Noeuds developpes : {result['nodes_expanded']}")
    print(f"  Taille max OPEN  : {result['max_open_size']}")
    print(f"  Temps            : {result['time_sec']*1000:.3f} ms")

    # Comparaison des 3 algorithmes
    compare_algorithms(grid, start, goal, verbose=True)

    # -- Phase 3 : Chaine de Markov (CDC ?4.3) --------------------
    print(f"\n{sep}\nPhase 3 -- Construction de la chaine de Markov (epsilon={eps})\n{sep}")

    policy     = a.extract_policy(result['path'])
    absorbing  = {goal: 'GOAL', (-1,-1): 'FAIL'}
    all_states = set(result['path']) | {(-1,-1)}

    mc = MarkovChain(all_states, absorbing)
    mc.build_from_policy(policy, grid, epsilon=eps)

    print(f"  Taille de P        : {mc.n} x {mc.n}")
    print(f"  P stochastique     : {mc.is_stochastic()}")

    # pi(n) = pi(0).P^n
    print("\n  Evolution P(GOAL) = pi(n)[GOAL] :")
    evol     = mc.get_distribution_evolution(start, max_steps=20)
    goal_idx = mc.state_to_idx[goal]
    for n in [1, 5, 10, 15, 20]:
        print(f"    n={n:>2} -> P(GOAL) = {evol[n-1][goal_idx]:.6f}")

    # -- Phase 4 : Analyse Markov (CDC ?4.4) ----------------------
    print(f"\n{sep}\nPhase 4 -- Analyse Markov\n{sep}")
    mc.print_analysis(initial_state=start, with_periodicity=True)

    # -- Phase 5 : Simulation Monte-Carlo (CDC ?4.5) --------------
    print(f"\n{sep}\nPhase 5 -- Simulation Monte-Carlo (N=1000)\n{sep}")
    sim   = MarkovSimulation(mc)
    stats = sim.run_simulations(start, n_simulations=1000, max_steps=100)
    sim.analyze_simulations(stats, verbose=True)
    sim.compare_with_theory(stats, max_steps=20, verbose=True)


# ----------------------------------------------------------------
# FONCTION POUR AFFICHER LES FICHIERS GENERES
# ----------------------------------------------------------------
def afficher_fichiers_generes():
    """Affiche la liste des fichiers PNG générés."""
    print("\n" + "="*60)
    print("📸 FICHIERS GENERES :")
    print("="*60)
    fichiers = [
        "experiment_1_comparaison_algorithmes.png",
        "experiment_2_vary_epsilon.png",
        "experiment_3_compare_heuristics.png",
        "experiment_4_weighted_astar.png"
    ]
    for f in fichiers:
        print(f"  - {f}")
    print("\n📁 Localisation :", __file__.replace('main.py', ''))


# ----------------------------------------------------------------
# POINT D'ENTREE CLI (VERSION MODIFIEE AVEC SAUVEGARDE PHOTOS)
# ----------------------------------------------------------------
def main():
    print(BANNER)

    parser = argparse.ArgumentParser(
        description="Mini-projet IA : A* + Chaines de Markov",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--exp',
        default='0',
        help=(
            "Experience a lancer :\n"
            "  0   -- Demonstration complete (defaut, sans sauvegarde photo)\n"
            "  1   -- E.1 : UCS / Greedy / A* sur 3 grilles (AVEC sauvegarde photo)\n"
            "  2   -- E.2 : Impact de l'incertitude epsilon (AVEC sauvegarde photo)\n"
            "  3   -- E.3 : Comparaison heuristiques (AVEC sauvegarde photo)\n"
            "  4   -- E.4 : Weighted A* (option) (AVEC sauvegarde photo)\n"
            "  all -- Toutes les experiences (AVEC sauvegarde photos)"
        ),
    )
    args = parser.parse_args()

    print(f"\n🔧 Option choisie : --exp {args.exp}\n")

    # Version modifiée avec save_fig=True pour générer les photos
    if args.exp == 'all':
        print("📸 Génération de toutes les photos...")
        experiment_1_compare_algorithms(save_fig=True)
        experiment_2_vary_epsilon(save_fig=True)
        experiment_3_compare_heuristics(save_fig=True)
        experiment_4_weighted_astar(save_fig=True)
        afficher_fichiers_generes()
        
    elif args.exp == '1':
        print("📸 Génération de la photo E.1...")
        experiment_1_compare_algorithms(save_fig=True)
        print("\n✅ Photo générée : experiment_1_comparaison_algorithmes.png")
        
    elif args.exp == '2':
        print("📸 Génération de la photo E.2...")
        experiment_2_vary_epsilon(save_fig=True)
        print("\n✅ Photo générée : experiment_2_vary_epsilon.png")
        
    elif args.exp == '3':
        print("📸 Génération de la photo E.3...")
        experiment_3_compare_heuristics(save_fig=True)
        print("\n✅ Photo générée : experiment_3_compare_heuristics.png")
        
    elif args.exp == '4':
        print("📸 Génération de la photo E.4 (option)...")
        experiment_4_weighted_astar(save_fig=True)
        print("\n✅ Photo générée : experiment_4_weighted_astar.png")
        
    elif args.exp == '0':
        print("📋 Démonstration complète (sans sauvegarde photo)...")
        demo()
        
    else:
        print(f"❌ Option inconnue : --exp {args.exp}")
        parser.print_help()
        sys.exit(1)

    print("\n" + "="*60)
    print("🏁 FIN DE L'EXECUTION")
    print("="*60)


if __name__ == "__main__":
    main()