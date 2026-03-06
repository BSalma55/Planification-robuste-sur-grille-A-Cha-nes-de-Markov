# -*- coding: utf-8 -*-
"""
astar.py -- Implémentation de A* et variantes
Mini-projet : Planification robuste sur grille
A* + Chaînes de Markov (2025-2026)

Algorithmes :
- UCS      : f(n) = g(n)
- Greedy   : f(n) = h(n)
- A*       : f(n) = g(n) + h(n)
- Weighted : f(n) = g(n) + w·h(n), w > 1
"""

import heapq
import time


class AStar:
    """
    Implémentation unifiée de A* et variantes.
    OPEN  : tas binaire min-heap
    CLOSED: ensemble O(1)
    Lazy deletion : doublons gérés à l'extraction
    """

    def __init__(self, grid, heuristic='manhattan'):
        self.grid = grid

        heuristic_map = {
            'manhattan': grid.manhattan_distance,
            'euclidean': grid.euclidean_distance,
            'zero':      grid.zero_heuristic,
        }

        if heuristic not in heuristic_map:
            raise ValueError(f"Heuristique inconnue : {heuristic}")

        self.heuristic = heuristic_map[heuristic]

    def search(self, start, goal, algorithm='astar', weight=1.0):
        """
        Recherche du chemin de start à goal.

        Retourne dict :
            path, cost, nodes_expanded, max_open_size,
            time_sec, success, visited
        """
        if algorithm == 'ucs':
            wg, wh = 1.0, 0.0
        elif algorithm == 'greedy':
            wg, wh = 0.0, 1.0
        else:
            wg, wh = 1.0, float(weight)

        t0 = time.perf_counter()

        counter   = 0
        open_heap = [(0.0, counter, start)]
        closed    = set()
        came_from = {}
        g_score   = {start: 0}

        nodes_expanded = 0
        max_open_size  = 0
        visited        = []

        while open_heap:
            max_open_size = max(max_open_size, len(open_heap))
            _, _, current = heapq.heappop(open_heap)

            if current == goal:
                return {
                    'path':           self._reconstruct_path(came_from, current),
                    'cost':           g_score[current],
                    'nodes_expanded': nodes_expanded,
                    'max_open_size':  max_open_size,
                    'time_sec':       time.perf_counter() - t0,
                    'success':        True,
                    'visited':        visited,
                }

            # Lazy deletion
            if current in closed:
                continue
            closed.add(current)
            visited.append(current)
            nodes_expanded += 1

            for neighbor in self.grid.neighbors(*current):
                if neighbor in closed:
                    continue
                g_new = g_score[current] + 1
                if g_new < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor]   = g_new
                    f_new = wg * g_new + wh * self.heuristic(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_heap, (f_new, counter, neighbor))

        return {
            'path':           None,
            'cost':           float('inf'),
            'nodes_expanded': nodes_expanded,
            'max_open_size':  max_open_size,
            'time_sec':       time.perf_counter() - t0,
            'success':        False,
            'visited':        visited,
        }

    def _reconstruct_path(self, came_from, node):
        path = [node]
        while node in came_from:
            node = came_from[node]
            path.append(node)
        path.reverse()
        return path

    def extract_policy(self, path):
        policy = {}
        for i in range(len(path) - 1):
            policy[path[i]] = (path[i+1][0] - path[i][0],
                               path[i+1][1] - path[i][1])
        if path:
            policy[path[-1]] = (0, 0)
        return policy


def compare_algorithms(grid, start, goal, verbose=True):
    """Compare UCS, Greedy et A* sur la même grille."""
    searcher = AStar(grid, heuristic='manhattan')
    results  = {}

    if verbose:
        print("\n" + "=" * 70)
        print("COMPARAISON UCS / GREEDY / A*")
        print("=" * 70)
        print(f"Grille {grid.width}x{grid.height} | "
              f"Obstacles : {len(grid.obstacles)} | "
              f"Start : {start} | Goal : {goal}")
        print("-" * 70)

    for algo in ['ucs', 'greedy', 'astar']:
        result        = searcher.search(start, goal, algorithm=algo)
        results[algo] = result
        if verbose:
            print(f"\n{algo.upper():12} | ", end="")
            if result['success']:
                print(f"Cout: {result['cost']:2} | "
                      f"Noeuds: {result['nodes_expanded']:3} | "
                      f"OPEN: {result['max_open_size']:2} | "
                      f"Temps: {result['time_sec']*1000:6.2f} ms")
            else:
                print("ECHEC")

    return results


if __name__ == "__main__":
    from Grid import Grid
    grid    = Grid(5, 5, obstacles=[(2, 2)])
    results = compare_algorithms(grid, (0, 0), (4, 4))
    if results['astar']['success']:
        print(f"\nChemin A* : {results['astar']['path']}")