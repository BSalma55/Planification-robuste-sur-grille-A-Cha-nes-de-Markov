# -*- coding: utf-8 -*-
"""
astar.py -- Implémentation de A* et variantes
Mini-projet : Planification robuste sur grille
A* + Chaînes de Markov (2025-2026)

Algorithmes implémentés :
- UCS (Uniform Cost Search) : f(n) = g(n)
- Greedy Best-First : f(n) = h(n)
- A* : f(n) = g(n) + h(n)
- Weighted A* : f(n) = g(n) + w·h(n), w > 1
"""

import heapq
import time


class AStar:
    """
    Implémentation unifiée de A* et variantes.
    
    Structure de données :
    - OPEN : tas binaire (min-heap) pour f(n) croissant
    - CLOSED : ensemble pour O(1) lookup
    """
    
    def __init__(self, grid, heuristic='manhattan'):
        """
        Paramètres
        ----------
        grid : Grid
            Grille de recherche
        heuristic : str
            'manhattan', 'euclidean', 'zero'
        """
        self.grid = grid
        
        # Map heuristiques
        heuristic_map = {
            'manhattan': grid.manhattan_distance,
            'euclidean': grid.euclidean_distance,
            'zero': grid.zero_heuristic
        }
        
        if heuristic not in heuristic_map:
            raise ValueError(f"Heuristique inconnue : {heuristic}")
        
        self.heuristic = heuristic_map[heuristic]
    
    def search(self, start, goal, algorithm='astar', weight=1.0):
        """
        Recherche du chemin optimal de start à goal.
        
        Paramètres
        ----------
        start : tuple
            Position initiale (x, y)
        goal : tuple
            Position but (x, y)
        algorithm : str
            'astar' | 'ucs' | 'greedy'
        weight : float
            Poids pour Weighted A* (w ≥ 1)
        
        Retourne
        --------
        dict
            {
                'path': [(x0,y0),...,(xn,yn)] ou None,
                'cost': int,
                'nodes_expanded': int,
                'max_open_size': int,
                'time_sec': float,
                'success': bool
            }
        """
        # Poids selon l'algorithme
        if algorithm == 'ucs':
            weight_g, weight_h = 1.0, 0.0
        elif algorithm == 'greedy':
            weight_g, weight_h = 0.0, 1.0
        else:  # 'astar' ou weighted
            weight_g, weight_h = 1.0, float(weight)
        
        t0 = time.perf_counter()
        
        # Initialisation OPEN/CLOSED
        counter = 0
        open_heap = [(0.0, counter, start)]  # (f, tie-breaker, state)
        in_open = {start}
        closed = set()
        
        came_from = {}
        g_score = {start: 0}
        
        nodes_expanded = 0
        max_open_size = 0
        
        # Recherche
        while open_heap:
            max_open_size = max(max_open_size, len(open_heap))
            
            # Extraire le nœud avec f minimal
            _, _, current = heapq.heappop(open_heap)
            in_open.discard(current)
            
            # But atteint
            if current == goal:
                return {
                    'path': self._reconstruct_path(came_from, current),
                    'cost': g_score[current],
                    'nodes_expanded': nodes_expanded,
                    'max_open_size': max_open_size,
                    'time_sec': time.perf_counter() - t0,
                    'success': True
                }
            
            # Lazy deletion : ignorer les entrées obsolètes
            if current in closed:
                continue
            closed.add(current)
            nodes_expanded += 1
            
            # Explorer les voisins
            for neighbor in self.grid.neighbors(*current):
                if neighbor in closed:
                    continue
                
                # Coût uniforme = 1
                g_new = g_score[current] + 1
                
                # Meilleur chemin vers ce voisin ?
                if g_new < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = g_new
                    
                    # Calculer f
                    h_new = self.heuristic(neighbor, goal)
                    f_new = weight_g * g_new + weight_h * h_new
                    
                    # Ajouter à OPEN
                    if neighbor not in in_open:
                        counter += 1
                        heapq.heappush(open_heap, (f_new, counter, neighbor))
                        in_open.add(neighbor)
        
        # Aucun chemin trouvé
        return {
            'path': None,
            'cost': float('inf'),
            'nodes_expanded': nodes_expanded,
            'max_open_size': max_open_size,
            'time_sec': time.perf_counter() - t0,
            'success': False
        }
    
    def _reconstruct_path(self, came_from, node):
        """Remonte l'arbre de recherche pour obtenir le chemin."""
        path = [node]
        while node in came_from:
            node = came_from[node]
            path.append(node)
        path.reverse()
        return path
    
    def extract_policy(self, path):
        """
        Extrait une politique (actions recommandées) à partir du chemin.
        """
        policy = {}
        for i in range(len(path) - 1):
            current = path[i]
            next_state = path[i + 1]
            action = (next_state[0] - current[0], next_state[1] - current[1])
            policy[current] = action
        
        # État but : rester sur place (self-loop)
        if path:
            policy[path[-1]] = (0, 0)
        
        return policy


def compare_algorithms(grid, start, goal, verbose=True):
    """
    Compare UCS, Greedy et A* sur la même grille.
  
    """
    searcher = AStar(grid, heuristic='manhattan')
    results = {}
    
    if verbose:
        print("\n" + "="*70)
        print("COMPARAISON UCS / GREEDY / A*")
        print("="*70)
        print(f"Grille {grid.width}×{grid.height} | "
              f"Obstacles : {len(grid.obstacles)} | "
              f"Start : {start} | Goal : {goal}")
        print("-"*70)
    
    for algo in ['ucs', 'greedy', 'astar']:
        result = searcher.search(start, goal, algorithm=algo)
        results[algo] = result
        
        if verbose:
            print(f"\n{algo.upper():12} | ", end="")
            if result['success']:
                print(f"Coût: {result['cost']:2} | "
                      f"Nœuds: {result['nodes_expanded']:3} | "
                      f"OPEN: {result['max_open_size']:2} | "
                      f"Temps: {result['time_sec']*1000:6.2f} ms")
            else:
                print("ÉCHEC")
    
    return results


# Test basique
if __name__ == "__main__":
    from grid import Grid
    
    # Créer une grille
    grid = Grid(5, 5, obstacles=[(2, 2)])
    
    # Comparer les algorithmes
    results = compare_algorithms(grid, (0, 0), (4, 4))
    
    # Afficher le chemin trouvé par A*
    if results['astar']['success']:
        print(f"\nChemin A* : {results['astar']['path']}")