# -*- coding: utf-8 -*-
"""
========================================================
astar.py  --  PHASE 2 : Planification deterministe
========================================================
CDC ?4.2 :
  P2.1  A* sur grille  (OPEN = heap, CLOSED = set)
  P2.2  Comparer A*, UCS, Greedy
  P2.3  Mesurer cout, noeuds developpes, taille OPEN, temps

CDC ?3.1 :
  Espace d'etats = grille 2D, cout uniforme c(n,n') = 1
  Heuristique Manhattan h((x,y)) = |x-xg| + |y-yg|
"""

import heapq
import time
import numpy as np


# ----------------------------------------------------------------
# GRILLE 2D  (CDC ?3.1 -- P1.1 / P1.2)
# ----------------------------------------------------------------
class Grid:
    """
    Grille 2D a 4-connexite.

    Chaque cellule libre est un etat n dans S (CDC ?3.1).
    Cout uniforme c(n, n') = 1 pour tout deplacement (CDC P1.2).
    """

    def __init__(self, width, height, obstacles=None):
        """
        Parametres
        ----------
        width, height : dimensions de la grille
        obstacles     : liste de tuples (x, y) bloques
        """
        self.width     = width
        self.height    = height
        self.obstacles = set(obstacles) if obstacles else set()

    def is_free(self, x, y):
        """True si la cellule (x, y) est dans la grille et libre."""
        return (0 <= x < self.width
                and 0 <= y < self.height
                and (x, y) not in self.obstacles)

    def neighbors(self, x, y):
        """
        Voisins accessibles en 4-connexite (N, S, E, O).
        CDC ?3.1 : transitions du graphe de recherche.
        """
        return [(nx, ny)
                for nx, ny in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
                if self.is_free(nx, ny)]

    # -- Heuristiques ---------------------------------------------

    def manhattan(self, pos1, pos2):
        """
        h(n) = |x - xg| + |y - yg|    (CDC ?3.1)

        ? Admissible  : h(n) <= h*(n)  car chaque pas coute 1
        ? Consistante : h(n) <= 1 + h(voisin)
          -> garantit l'optimalite de A* sans reexpansion
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def zero(self, pos1, pos2):
        """
        h(n) = 0   (CDC E.3 -- heuristique nulle)

        ? Admissible (trivial)
        ? A* avec h=0 equivaut exactement a UCS
        ? Moins informative -> plus de noeuds developpes
        """
        return 0

    def euclidean(self, pos1, pos2):
        """h(n) = distance euclidienne -- admissible sur grille unitaire."""
        return float(np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2))


# ----------------------------------------------------------------
# A* ET VARIANTES  (CDC ?4.2 -- P2.1 / P2.2 / P2.3)
# ----------------------------------------------------------------
class AStar:
    """
    Recherche heuristique unifiee.

    Formule generale :  f(n) = wg . g(n) + wh . h(n)

    ?--------------?-----?-----?----------------------------?
    | Algorithme   |  wg |  wh | Propriete                  |
    ?--------------?-----?-----?----------------------------?
    | UCS          | 1.0 | 0.0 | Optimal                    |
    | Greedy       | 0.0 | 1.0 | Rapide, non garanti opt.   |
    | A*           | 1.0 | 1.0 | Optimal si h admissible    |
    | Weighted A*  | 1.0 | w>1 | w-optimal  (CDC E.4)       |
    ?--------------?-----?-----?----------------------------?

    OPEN  = tas binaire min-heap  (insertion O(log n))
    CLOSED = ensemble Python       (appartenance O(1))
    """

    _HEURISTICS = {
        'manhattan': lambda g: g.manhattan,
        'zero':      lambda g: g.zero,
        'euclidean': lambda g: g.euclidean,
    }

    def __init__(self, grid, heuristic='manhattan'):
        self.grid = grid
        fn = self._HEURISTICS.get(heuristic)
        if fn is None:
            raise ValueError(f"Heuristique inconnue '{heuristic}'. "
                             f"Choix : {list(self._HEURISTICS)}")
        self.h = fn(grid)

    # -- Recherche ------------------------------------------------

    def search(self, start, goal, algorithm='astar', weight=1.0):
        """
        Recherche du chemin de start a goal.

        Parametres
        ----------
        start     : (x, y) -- etat initial s?
        goal      : (x, y) -- etat but g
        algorithm : 'astar' | 'ucs' | 'greedy'
        weight    : w >= 1 pour Weighted A* (ignore sinon)

        Retourne  dict
        --------------
        path           : [(x0,y0),...,(xn,yn)] ou None
        cost           : g(goal)  (cout optimal)
        nodes_expanded : noeuds extraits de OPEN (taille de CLOSED)
        max_open_size  : pic memoire de OPEN
        time_sec       : duree d'execution
        success        : bool
        """
        # -- Poids selon l'algorithme (CDC P2.2)
        if   algorithm == 'ucs':    wg, wh = 1.0, 0.0
        elif algorithm == 'greedy': wg, wh = 0.0, 1.0
        else:                       wg, wh = 1.0, float(weight)   # astar / weighted

        t0 = time.perf_counter()

        # -- Initialisation OPEN / CLOSED
        counter = 0                                 # brise les egalites de f
        open_heap   = [(0.0, counter, start)]       # (f, tie, etat)
        in_open     = {start}                       # appartenance OPEN  O(1)
        closed      = set()                         # CLOSED
        came_from   = {}                            # arbre de recherche
        g_score     = {start: 0}                    # g(n)

        nodes_expanded = 0
        max_open_size  = 0

        while open_heap:
            max_open_size = max(max_open_size, len(open_heap))

            _, _, current = heapq.heappop(open_heap)
            in_open.discard(current)

            # But atteint
            if current == goal:
                return {
                    'path':           self._reconstruct(came_from, current),
                    'cost':           g_score[current],
                    'nodes_expanded': nodes_expanded,
                    'max_open_size':  max_open_size,
                    'time_sec':       time.perf_counter() - t0,
                    'success':        True,
                }

            # Entree obsolete dans le heap (lazy deletion)
            if current in closed:
                continue
            closed.add(current)
            nodes_expanded += 1

            for nbr in self.grid.neighbors(*current):
                if nbr in closed:
                    continue
                g_new = g_score[current] + 1           # cout uniforme = 1
                if g_new < g_score.get(nbr, float('inf')):
                    came_from[nbr] = current
                    g_score[nbr]   = g_new
                    f_new = wg * g_new + wh * self.h(nbr, goal)
                    counter += 1
                    heapq.heappush(open_heap, (f_new, counter, nbr))
                    in_open.add(nbr)

        # Echec : aucun chemin
        return {
            'path':           None,
            'cost':           float('inf'),
            'nodes_expanded': nodes_expanded,
            'max_open_size':  max_open_size,
            'time_sec':       time.perf_counter() - t0,
            'success':        False,
        }

    def _reconstruct(self, came_from, node):
        """Remonte came_from pour reconstituer le chemin optimal."""
        path = [node]
        while node in came_from:
            node = came_from[node]
            path.append(node)
        path.reverse()
        return path

    # -- Politique induite (CDC ?4.3 -- P3) -----------------------

    def extract_policy(self, path):
        """
        Construit pi : etat -> action depuis le chemin A*.

        CDC ?4.3 : politique induite pour construire la matrice P Markov.
        L'etat but recoit l'action (0,0) -> il devient GOAL absorbant.

        Retourne  {(x, y): (dx, dy)}
        """
        policy = {}
        for i in range(len(path) - 1):
            policy[path[i]] = (path[i+1][0] - path[i][0],
                               path[i+1][1] - path[i][1])
        if path:
            policy[path[-1]] = (0, 0)       # etat but : self-loop
        return policy


# ----------------------------------------------------------------
# COMPARAISON ALGORITHMES  (CDC E.1)
# ----------------------------------------------------------------
def compare_algorithms(grid, start, goal, verbose=True):
    """
    Compare UCS, Greedy et A* sur la meme grille.
    CDC E.1 : cout, noeuds developpes, taille OPEN, temps.

    Retourne  {algo: resultat}
    """
    searcher = AStar(grid, heuristic='manhattan')
    results  = {}

    if verbose:
        print("\n" + "="*60)
        print("E.1 -- COMPARAISON  UCS / GREEDY / A*")
        print("="*60)
        print(f"Grille {grid.width}x{grid.height} | "
              f"obstacles={len(grid.obstacles)} | "
              f"start={start} | goal={goal}")

    for algo in ('ucs', 'greedy', 'astar'):
        r = searcher.search(start, goal, algorithm=algo)
        results[algo] = r
        if verbose:
            print(f"\n  {algo.upper()} :")
            print(f"    Chemin trouve      : {'Oui' if r['success'] else 'Non'}")
            if r['success']:
                print(f"    Cout total         : {r['cost']}")
                print(f"    Longueur chemin    : {len(r['path'])}")
            print(f"    Noeuds developpes   : {r['nodes_expanded']}")
            print(f"    Taille max OPEN    : {r['max_open_size']}")
            print(f"    Temps              : {r['time_sec']*1000:.3f} ms")

    return results


# ----------------------------------------------------------------
if __name__ == "__main__":
    print("=== Test astar.py ===")
    g = Grid(5, 5, obstacles=[(2, 2)])
    compare_algorithms(g, (0, 0), (4, 4))

    # Weighted A* (CDC E.4)
    print("\n--- Weighted A* (CDC E.4) ---")
    a = AStar(g)
    for w in [1.0, 1.5, 2.0, 3.0]:
        r = a.search((0,0), (4,4), algorithm='astar', weight=w)
        print(f"  w={w} | cout={r['cost']} | noeuds={r['nodes_expanded']}")
