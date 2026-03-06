# -*- coding: utf-8 -*-
"""
grid.py -- Modélisation de la grille 2D
Mini-projet : Planification robuste sur grille
A* + Chaînes de Markov (2025-2026)
"""

import numpy as np


class Grid:
    """
    Représentation d'une grille 2D avec obstacles.
    
    Chaque cellule libre est un état dans l'espace de recherche.
    Connectivité : 4-voisins (haut, bas, gauche, droite).
    Coût uniforme : c(n, n') = 1 pour tout déplacement.
    """
    
    # ============================================================
    # INITIALISATION
    # ============================================================
    
    def __init__(self, width, height, obstacles=None):
        """
        Paramètres
        ----------
        width : int
            Largeur de la grille
        height : int
            Hauteur de la grille
        obstacles : list of tuple
            Liste des positions (x, y) bloquées
        """
        self.width = width
        self.height = height
        self.obstacles = set(obstacles) if obstacles else set()
    
    # ============================================================
    # GESTION DES CELLULES
    # ============================================================
    
    def is_free(self, x, y):
        """
        Vérifie si une cellule est libre.
        
        Paramètres
        ----------
        x, y : int
            Coordonnées
        
        Retourne
        --------
        bool
            True si la cellule est dans la grille et libre
        """
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                (x, y) not in self.obstacles)
    
    def neighbors(self, x, y):
        """
        Retourne les voisins accessibles en 4-connectivité.
        
        Paramètres
        ----------
        x, y : int
            Position courante
        
        Retourne
        --------
        list of tuple
            Liste des voisins libres [(x', y'), ...]
        """
        candidates = [
            (x + 1, y),  # droite
            (x - 1, y),  # gauche
            (x, y + 1),  # bas
            (x, y - 1)   # haut
        ]
        return [(nx, ny) for nx, ny in candidates if self.is_free(nx, ny)]
    
    # ============================================================
    # HEURISTIQUES
    # ============================================================
    
    def manhattan_distance(self, pos1, pos2):
        """
        Heuristique Manhattan (distance L1).
        
        h(n) = |x - x_g| + |y - y_g|
        
        Propriétés :
        - Admissible : h(n) ≤ h*(n) pour grilles 4-voisins
        - Cohérente : h(n) ≤ 1 + h(n') pour tout voisin n'
        
        Paramètres
        ----------
        pos1, pos2 : tuple
            Positions (x, y)
        
        Retourne
        --------
        float
            Distance Manhattan
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def euclidean_distance(self, pos1, pos2):
        """
        Heuristique Euclidienne (distance L2).
        
        h(n) = sqrt((x - x_g)² + (y - y_g)²)
        
        Propriétés :
        - Admissible : oui
        - Cohérente : non garantie sur grilles 4-voisins
        
        Paramètres
        ----------
        pos1, pos2 : tuple
            Positions (x, y)
        
        Retourne
        --------
        float
            Distance Euclidienne
        """
        return float(np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2))
    
    def zero_heuristic(self, pos1, pos2):
        """
        Heuristique nulle (h = 0).
        
        Propriétés :
        - Admissible : trivial
        - Effet : A* dégénère en UCS
        
        Paramètres
        ----------
        pos1, pos2 : tuple
            (ignorées)
        
        Retourne
        --------
        float
            0.0
        """
        return 0.0


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    # Créer une grille 5x5 avec un obstacle central
    grid = Grid(5, 5, obstacles=[(2, 2)])
    
    print("Grille 5x5 avec obstacle en (2, 2)")
    print(f"Voisins de (0, 0) : {grid.neighbors(0, 0)}")
    print(f"Voisins de (2, 1) : {grid.neighbors(2, 1)}")
    print(f"Distance Manhattan (0,0)->(4,4) : {grid.manhattan_distance((0, 0), (4, 4))}")