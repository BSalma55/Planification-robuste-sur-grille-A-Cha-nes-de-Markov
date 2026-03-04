import heapq
import numpy as np
from collections import defaultdict
import time

class Grid:
    """Grille 2D avec obstacles"""
    
    def __init__(self, width, height, obstacles=None):
        self.width = width
        self.height = height
        self.obstacles = set(obstacles) if obstacles else set()
        
    def is_free(self, x, y):
        """Vérifie si une cellule est libre"""
        return (0 <= x < self.width and 0 <= y < self.height 
                and (x, y) not in self.obstacles)
    
    def neighbors(self, x, y):
        """Retourne les voisins accessibles (4-connectivité)"""
        candidates = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        return [(nx, ny) for nx, ny in candidates if self.is_free(nx, ny)]
    
    def manhattan_distance(self, pos1, pos2):
        """Heuristique Manhattan"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def euclidean_distance(self, pos1, pos2):
        """Heuristique euclidienne"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def zero_heuristic(self, pos1, pos2):
        """Heuristique nulle (pour UCS)"""
        return 0


class AStar:
    """Implémentation de A* et variantes"""
    
    def __init__(self, grid, heuristic='manhattan'):
        self.grid = grid
        self.heuristic_map = {
            'manhattan': grid.manhattan_distance,
            'euclidean': grid.euclidean_distance,
            'zero': grid.zero_heuristic
        }
        self.heuristic = self.heuristic_map.get(heuristic, grid.manhattan_distance)
        
    def search(self, start, goal, weight=1.0, algorithm='astar'):
        """
        Recherche du chemin optimal
        algorithm: 'astar', 'ucs', 'greedy'
        weight: facteur de pondération pour A* (weighted A*)
        """
        # Adaptation des poids selon l'algorithme
        if algorithm == 'ucs':
            weight_g = 1.0
            weight_h = 0.0
        elif algorithm == 'greedy':
            weight_g = 0.0
            weight_h = 1.0
        else:  # A* standard ou pondéré
            weight_g = 1.0
            weight_h = weight
        
        # OPEN: file de priorité (f, compteur, état)
        # Compteur pour éviter les comparaisons entre états non comparables
        counter = 0
        open_set = []
        heapq.heappush(open_set, (0, counter, start))
        
        # Structures de données
        came_from = {}
        g_score = {start: 0}
        f_score = {start: weight_g * 0 + weight_h * self.heuristic(start, goal)}
        
        open_set_lookup = {start}
        closed_set = set()
        
        # Métriques
        nodes_expanded = 0
        max_open_size = 0
        
        while open_set:
            # Mise à jour de la taille max de OPEN
            max_open_size = max(max_open_size, len(open_set))
            
            # Extraire le nœud avec le plus petit f
            current_f, _, current = heapq.heappop(open_set)
            open_set_lookup.remove(current)
            
            # Vérifier si on a atteint le but
            if current == goal:
                path = self.reconstruct_path(came_from, current)
                return {
                    'path': path,
                    'cost': g_score[current],
                    'nodes_expanded': nodes_expanded,
                    'max_open_size': max_open_size,
                    'success': True
                }
            
            # Marquer comme visité
            closed_set.add(current)
            nodes_expanded += 1
            
            # Explorer les voisins
            for neighbor in self.grid.neighbors(*current):
                if neighbor in closed_set:
                    continue
                
                # Coût du mouvement (uniforme = 1)
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # Meilleur chemin trouvé
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = weight_g * tentative_g + weight_h * self.heuristic(neighbor, goal)
                    
                    if neighbor not in open_set_lookup:
                        counter += 1
                        heapq.heappush(open_set, (f, counter, neighbor))
                        open_set_lookup.add(neighbor)
                    else:
                        # Mise à jour de la priorité (supprimer et réinsérer)
                        # Dans une implémentation plus efficace, on utiliserait decrease-key
                        pass
        
        # Aucun chemin trouvé
        return {
            'path': None,
            'cost': float('inf'),
            'nodes_expanded': nodes_expanded,
            'max_open_size': max_open_size,
            'success': False
        }
    
    def reconstruct_path(self, came_from, current):
        """Reconstruit le chemin du début à la fin"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def extract_policy(self, path):
        """
        Extrait une politique (actions recommandées) à partir du chemin
        Retourne un dictionnaire {état: action}
        """
        policy = {}
        for i in range(len(path) - 1):
            current = path[i]
            next_state = path[i + 1]
            # Déterminer l'action (dx, dy)
            action = (next_state[0] - current[0], next_state[1] - current[1])
            policy[current] = action
        # État final : rester sur place
        if path:
            policy[path[-1]] = (0, 0)
        return policy


def compare_algorithms(grid, start, goal):
    """Compare les différents algorithmes de recherche"""
    astar = AStar(grid)
    algorithms = ['astar', 'ucs', 'greedy']
    results = {}
    
    print("\n" + "="*60)
    print("COMPARAISON DES ALGORITHMES DE RECHERCHE")
    print("="*60)
    
    for algo in algorithms:
        start_time = time.time()
        result = astar.search(start, goal, algorithm=algo)
        elapsed = time.time() - start_time
        
        results[algo] = {
            **result,
            'time': elapsed
        }
        
        print(f"\n{algo.upper()} :")
        print(f"  Chemin trouvé: {'Oui' if result['success'] else 'Non'}")
        if result['success']:
            print(f"  Longueur du chemin: {len(result['path'])}")
            print(f"  Coût total: {result['cost']}")
        print(f"  Nœuds développés: {result['nodes_expanded']}")
        print(f"  Taille max OPEN: {result['max_open_size']}")
        print(f"  Temps d'exécution: {elapsed:.4f} s")
    
    return results


# Test simple
if __name__ == "__main__":
    # Créer une grille simple
    grid = Grid(5, 5, obstacles=[(2, 2), (2, 3)])
    start = (0, 0)
    goal = (4, 4)
    
    # Comparer les algorithmes
    results = compare_algorithms(grid, start, goal)
    
    # Afficher le chemin trouvé par A*
    astar = AStar(grid)
    result = astar.search(start, goal, algorithm='astar')
    if result['success']:
        print(f"\nChemin A*: {result['path']}")