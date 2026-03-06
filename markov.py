# -*- coding: utf-8 -*-
"""
markov.py -- Chaine de Markov pour modeliser l'incertitude
Mini-projet : Planification robuste sur grille
A* + Chaines de Markov (2025-2026)
"""

import numpy as np
import networkx as nx
import sys


class MarkovChain:
    """
    Chaine de Markov a temps discret sur un espace d'etats fini.
    
    Matrice de transition : P = (p_ij) stochastique par lignes
    Evolution : p^(n) = p^(0) · P^n (Chapman-Kolmogorov)
    
    Periodicite : Une classe C a pour periode d = pgcd des longueurs
                  de tous les retours a un etat (methode puissances).
    """
    
    # ============================================================
    # INITIALISATION
    # ============================================================
    
    def __init__(self, states, absorbing_states=None):
        """
        Parametres
        ----------
        states : list of tuple
            Ensemble d'etats (ex : positions sur grille)
        absorbing_states : dict
            {etat: 'GOAL' | 'FAIL'} — etats absorbants
        """
        self.states = list(states)
        self.n = len(self.states)
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for i, s in enumerate(self.states)}
        
        # Matrice de transition (initialisee a 0)
        self.P = np.zeros((self.n, self.n))
        
        # Etats absorbants
        self.absorbing_states = absorbing_states if absorbing_states else {}
        self.absorbing_indices = {
            self.state_to_idx[s] 
            for s in self.absorbing_states 
            if s in self.state_to_idx
        }
    
    # ============================================================
    # CONSTRUCTION DE LA MATRICE P
    # ============================================================
    
    def build_from_policy(self, policy, grid, epsilon, fail_states=None):
        """
        Construit P a partir d'une politique A*.
        
        Modele d'incertitude avec parametre e :
        - p(s -> s_voulu) = 1 - e : action prevue reussit
        - p(s -> lateral) = e/2 : deviation laterale gauche/droite
        - Self-loop si collision/hors-grille
        
        Parametres
        ----------
        policy : dict
            {etat: (dx, dy)} — action recommandee
        grid : Grid
            Grille pour tester les destinations
        epsilon : float
            Taux d'incertitude dans [0, 1]
        fail_states : set
            Etats d'echec (obstacles) [optionnel]
        """
        valid = set(self.states)
        
        for state in self.states:
            i = self.state_to_idx[state]
            
            # Etat absorbant : boucle sur lui-meme
            if state in self.absorbing_states:
                self.P[i, i] = 1.0
                continue
            
            # Action recommandee
            action = policy.get(state, (0, 0))
            dx, dy = action
            
            # Destination principale
            dest_main = (state[0] + dx, state[1] + dy)
            if not grid.is_free(*dest_main) or dest_main not in valid:
                dest_main = state  # Self-loop si bloque
            
            self._add_transition(i, dest_main, 1.0 - epsilon)
            
            # Deviations laterales
            if action != (0, 0):
                # Voisins perpendiculaires a l'action
                if dx != 0:
                    laterals = [(0, 1), (0, -1)]  # haut/bas
                else:
                    laterals = [(1, 0), (-1, 0)]  # droite/gauche
                
                for ldx, ldy in laterals:
                    dest_lat = (state[0] + ldx, state[1] + ldy)
                    if not grid.is_free(*dest_lat) or dest_lat not in valid:
                        dest_lat = state
                    self._add_transition(i, dest_lat, epsilon / 2.0)
        
        # Normaliser pour garantir stochasticite
        self._normalize()
    
    def _add_transition(self, i, dest, prob):
        """Ajoute une transition avec probabilite."""
        j = self.state_to_idx.get(dest)
        if j is not None:
            self.P[i, j] += prob
    
    def _normalize(self):
        """Normalise les lignes de P pour qu'elles somment a 1."""
        row_sums = self.P.sum(axis=1, keepdims=True)
        # Eviter la division par zero
        row_sums[row_sums == 0] = 1.0
        self.P /= row_sums
    
    def is_stochastic(self, tol=1e-9):
        """Verifie si P est stochastique (lignes = 1)."""
        return bool(np.allclose(self.P.sum(axis=1), 1.0, atol=tol))
    
    # ============================================================
    # CALCUL DES DISTRIBUTIONS
    # ============================================================
    
    def get_Pn(self, n):
        """Calcule P^n."""
        return np.linalg.matrix_power(self.P, n)
    
    def get_distribution(self, initial_state, n):
        """
        Calcule p^(n) = p^(0) · P^n.
        
        Parametres
        ----------
        initial_state : tuple
            Etat initial
        n : int
            Nombre d'etapes
        
        Retourne
        --------
        np.array
            Distribution p^(n)
        """
        if initial_state not in self.state_to_idx:
            raise ValueError(f"Etat inconnu : {initial_state}")
        
        i0 = self.state_to_idx[initial_state]
        pi0 = np.zeros(self.n)
        pi0[i0] = 1.0
        
        return pi0 @ self.get_Pn(n)
    
    def get_distribution_evolution(self, initial_state, max_steps):
        """
        Retourne [p^(1), p^(2), ..., p^(max_steps)].
        
        Utile pour tracer P(GOAL) au fil du temps.
        """
        if initial_state not in self.state_to_idx:
            raise ValueError(f"Etat inconnu : {initial_state}")
        
        i0 = self.state_to_idx[initial_state]
        pi = np.zeros(self.n)
        pi[i0] = 1.0
        
        evol = []
        for _ in range(max_steps):
            pi = pi @ self.P
            evol.append(pi.copy())
        
        return evol
    
    # ============================================================
    # ANALYSE DES CLASSES
    # ============================================================
    
    def analyze_classes(self):
        """
        Identifie les classes de communication via composantes
        fortement connexes du graphe des transitions.
        
        Retourne
        --------
        list of dict
            Classes avec type (absorbant/transitoire)
        """
        G = nx.DiGraph()
        G.add_nodes_from(range(self.n))
        
        for i in range(self.n):
            for j in range(self.n):
                if self.P[i, j] > 1e-12:
                    G.add_edge(i, j)
        
        classes = []
        for comp in nx.strongly_connected_components(G):
            # Verifier si la classe est fermee (aucune sortie)
            is_closed = True
            for i in comp:
                for j in range(self.n):
                    if self.P[i, j] > 1e-12 and j not in comp:
                        is_closed = False
                        break
                if not is_closed:
                    break
            
            # Determiner le type
            if is_closed and len(comp) == 1:
                idx = next(iter(comp))
                if np.isclose(self.P[idx, idx], 1.0):
                    ctype = 'absorbant'
                else:
                    ctype = 'transitoire'
            elif is_closed:
                ctype = 'absorbant'
            else:
                ctype = 'transitoire'
            
            classes.append({
                'etats': [self.idx_to_state[i] for i in comp],
                'indices': list(comp),
                'type': ctype
            })
        
        return classes
    
    # ============================================================
    # ANALYSE DE PERIODICITE
    # ============================================================
    
    def _periode_par_puissances(self, etat_idx, max_iter=100):
        """
        Calcule la periode d'un etat en cherchant le plus petit k > 0
        tel que P^k[i,i] > 0.
        
        Parametres
        ----------
        etat_idx : int
            Indice de l'etat
        max_iter : int
            Nombre maximum d'iterations
        
        Retourne
        --------
        int
            Periode (0 si non trouvee, 1 si aperiodique)
        """
        Pk = self.P.copy()
        
        for k in range(1, max_iter + 1):
            if Pk[etat_idx, etat_idx] > 1e-10:
                return k
            Pk = Pk @ self.P
        
        return 0  # Non trouve
    
    def analyze_periodicity(self, method='puissances', verbose=False, max_iter=100):
        """
        Analyse la periodicite de la chaine.
        
        Parametres
        ----------
        method : str
            Methode de calcul ('puissances')
        verbose : bool
            Afficher les resultats
        max_iter : int
            Nombre maximum d'iterations (pour methode puissances)
        
        Retourne
        --------
        list of dict
            Periodicite par classe
        """
        classes = self.analyze_classes()
        results = []
        
        for cls in classes:
            if cls['type'] == 'absorbant':
                # Un etat absorbant a une periode de 1 (self-loop)
                periode = 1
                est_periodique = False
                methode_utilisee = 'absorbant'
            else:
                # Prendre le premier etat de la classe
                i = cls['indices'][0]
                
                periode = self._periode_par_puissances(i, max_iter)
                methode_utilisee = f'puissances (max_iter={max_iter})'
                
                est_periodique = periode > 1
                if periode == 0:
                    periode = 'indeterminee'
                    est_periodique = None
            
            results.append({
                'classe': cls,
                'etats': cls['etats'],
                'type': cls['type'],
                'periode': periode,
                'est_periodique': est_periodique,
                'methode': methode_utilisee
            })
        
        if verbose:
            self._afficher_periodicite(results, method)
        
        return results
    
    def _afficher_periodicite(self, results, method):
        """Affiche les resultats de l'analyse de periodicite."""
        print("\n" + "="*60)
        print("ANALYSE DE PERIODICITE")
        print("="*60)
        print(f"Methode utilisee : {method}")
        
        for i, r in enumerate(results):
            etats_str = str(r['etats'][:3])
            if len(r['etats']) > 3:
                etats_str = etats_str[:-1] + ", ...]"
            
            if r['est_periodique'] is None:
                etat_perio = "indeterminee"
            elif r['est_periodique']:
                etat_perio = "periodique"
            else:
                etat_perio = "aperiodique"
            
            print(f"\nClasse {i+1} [{r['type']}] :")
            print(f"  Etats : {etats_str}")
            print(f"  Periode : {r['periode']}")
            print(f"  Type : {etat_perio}")
    
    # ============================================================
    # VISUALISATION GRAPHIQUE
    # ============================================================
    
    def plot_periodicity(self, method='puissances', save_fig=True, show=True, figsize=(14, 10)):
        """
        Genere des graphiques de visualisation de la periodicite.
        
        Parametres
        ----------
        method : str
            Methode de calcul ('puissances')
        save_fig : bool
            Sauvegarder la figure
        show : bool
            Afficher la figure
        figsize : tuple
            Taille de la figure (largeur, hauteur)
        
        Retourne
        --------
        matplotlib.figure.Figure
            La figure generee
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch
        except ImportError:
            print("matplotlib non installe. Installation : pip install matplotlib")
            return None
        
        # Analyser la periodicite
        results = self.analyze_periodicity(method=method, verbose=False)
        
        # Preparer les donnees
        classes, periodes_num, periodes_texte, types, couleurs, tailles = self._preparer_donnees_periodicite(results)
        
        # Creer la figure
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.3, wspace=0.3)
        
        # Titre principal
        fig.suptitle(f"Analyse de Periodicite - Methode: {method.upper()}", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Graphique 1 : Diagramme en barres des periodes
        self._plot_barres_periodes(fig, gs[0, 0], classes, periodes_num, periodes_texte, couleurs)
        
        # Graphique 2 : Distribution des types
        self._plot_distribution_types(fig, gs[0, 1], results, types)
        
        # Graphique 3 : Taille des classes
        self._plot_taille_classes(fig, gs[1, :], classes, tailles, periodes_texte, couleurs)
        
        # Ajustement et sauvegarde
        plt.subplots_adjust(left=0.1, right=0.9, top=0.88, bottom=0.12, hspace=0.4, wspace=0.3)
        
        if save_fig:
            plt.savefig(f'periodicite_{method}.png', dpi=150, bbox_inches='tight')
            print(f"Figure sauvegardee : periodicite_{method}.png")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def _preparer_donnees_periodicite(self, results):
        """Prepare les donnees pour les graphiques de periodicite."""
        classes = []
        periodes_numeriques = []
        periodes_texte = []
        types = []
        couleurs_barres = []
        tailles_classes = []
        
        for i, r in enumerate(results):
            classes.append(f"Classe {i+1}")
            tailles_classes.append(len(r['etats']))
            
            # Periode (numerique ou texte)
            if isinstance(r['periode'], (int, float)):
                periodes_numeriques.append(float(r['periode']))
                periodes_texte.append(str(r['periode']))
            else:
                periodes_numeriques.append(0)
                periodes_texte.append(r['periode'])
            
            types.append(r['type'])
            
            # Couleurs selon le type et la periodicite
            if r['type'] == 'absorbant':
                couleurs_barres.append('gold')
            elif r['est_periodique'] is True:
                couleurs_barres.append('red')
            elif r['est_periodique'] is False:
                couleurs_barres.append('lightgreen')
            else:
                couleurs_barres.append('lightgray')
        
        return classes, periodes_numeriques, periodes_texte, types, couleurs_barres, tailles_classes
    
    def _plot_barres_periodes(self, fig, subplot, classes, periodes_num, periodes_texte, couleurs):
        """Cree le graphique en barres des periodes."""
        from matplotlib.patches import Patch
        
        ax = fig.add_subplot(subplot)
        x = np.arange(len(classes))
        bars = ax.bar(x, periodes_num, color=couleurs, edgecolor='black', alpha=0.8, width=0.6)
        
        # Ajouter les valeurs
        for i, (bar, texte) in enumerate(zip(bars, periodes_texte)):
            hauteur = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, hauteur + 0.1,
                    texte, ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel("Classes de communication", fontsize=11)
        ax.set_ylabel("Periode", fontsize=11)
        ax.set_title("Periode par classe", fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, max(3, max(periodes_num) + 0.5))
        ax.grid(True, alpha=0.3, axis='y')
        
        # Legende
        legend_elements = [
            Patch(facecolor='gold', edgecolor='black', label='Absorbant (periode 1)'),
            Patch(facecolor='lightgreen', edgecolor='black', label='Aperiodique'),
            Patch(facecolor='red', edgecolor='black', label='Periodique'),
            Patch(facecolor='lightgray', edgecolor='black', label='Indetermine')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    def _plot_distribution_types(self, fig, subplot, results, types):
        """Cree le camembert de distribution des types."""
        ax = fig.add_subplot(subplot)
        
        # Compter les types
        types_count = {
            'absorbant': sum(1 for t in types if t == 'absorbant'),
            'aperiodique': sum(1 for r in results if r['type'] == 'transitoire' and r['est_periodique'] is False),
            'periodique': sum(1 for r in results if r['est_periodique'] is True),
            'indetermine': sum(1 for r in results if r['est_periodique'] is None)
        }
        
        # Filtrer les comptes nuls
        labels = []
        sizes = []
        colors_pie = []
        
        mapping = [
            ('absorbant', 'Absorbant', 'gold'),
            ('aperiodique', 'Aperiodique', 'lightgreen'),
            ('periodique', 'Periodique', 'red'),
            ('indetermine', 'Indetermine', 'lightgray')
        ]
        
        for key, label, color in mapping:
            if types_count[key] > 0:
                labels.append(f"{label}\n({types_count[key]})")
                sizes.append(types_count[key])
                colors_pie.append(color)
        
        if sizes:
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie,
                                              autopct='%1.1f%%', startangle=90,
                                              textprops={'fontsize': 10, 'fontweight': 'bold'})
            for autotext in autotexts:
                autotext.set_color('black')
        else:
            ax.text(0.5, 0.5, 'Aucune donnee', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title("Distribution des types de classes", fontsize=12, fontweight='bold')
    
    def _plot_taille_classes(self, fig, subplot, classes, tailles, periodes, couleurs):
        """Cree le graphique des tailles de classes."""
        ax = fig.add_subplot(subplot)
        
        # Trier par taille
        classes_triees = sorted(zip(classes, tailles, periodes), key=lambda x: x[1], reverse=True)
        
        classes_noms = [c[0] for c in classes_triees]
        tailles_triees = [c[1] for c in classes_triees]
        periodes_triees = [c[2] for c in classes_triees]
        
        # Reordonner les couleurs
        couleurs_triees = []
        for nom in classes_noms:
            idx = classes.index(nom)
            couleurs_triees.append(couleurs[idx])
        
        bars = ax.bar(range(len(classes_triees)), tailles_triees, 
                      color=couleurs_triees, edgecolor='black', alpha=0.8)
        
        # Ajouter les valeurs
        for i, (bar, taille, periode) in enumerate(zip(bars, tailles_triees, periodes_triees)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{taille} (p={periode})", ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel("Classes de communication (triees par taille)", fontsize=11)
        ax.set_ylabel("Nombre d'etats", fontsize=11)
        ax.set_title("Taille des classes", fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(classes_triees)))
        ax.set_xticklabels(classes_noms, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    # ============================================================
    # ANALYSE D'ABSORPTION
    # ============================================================
    
    def absorption_analysis(self):
        """
        Analyse d'absorption : N, B, t.
        
        Decomposition : P = [I 0; R Q]
        - N = (I - Q)^(-1) : matrice fondamentale
        - B = N·R : probabilites d'absorption
        - t = N·1 : temps moyen
        
        Retourne
        --------
        dict ou None
            {'B': B, 't': t, 'N': N, 'trans_indices': [...], 'abs_indices': [...]}
        """
        if not self.absorbing_indices:
            return None
        
        trans_idx = [i for i in range(self.n) if i not in self.absorbing_indices]
        abs_idx = sorted(self.absorbing_indices)
        
        if not trans_idx:
            return None
        
        Q = self.P[np.ix_(trans_idx, trans_idx)]
        R = self.P[np.ix_(trans_idx, abs_idx)]
        
        try:
            N = np.linalg.inv(np.eye(len(trans_idx)) - Q)
            B = N @ R
            t = N @ np.ones(len(trans_idx))
            
            return {
                'B': B,
                't': t,
                'N': N,
                'trans_indices': trans_idx,
                'abs_indices': abs_idx
            }
        except np.linalg.LinAlgError:
            return None
    
    # ============================================================
    # AFFICHAGE
    # ============================================================
    
    def print_analysis(self, initial_state=None, with_periodicity=False, periodicity_method='puissances'):
        """
        Affiche une analyse complete de la chaine.
        
        Parametres
        ----------
        initial_state : tuple
            Etat initial pour l'analyse d'absorption
        with_periodicity : bool
            Inclure l'analyse de periodicite
        periodicity_method : str
            Methode pour le calcul de periodicite
        """
        print("\n" + "="*70)
        print("ANALYSE DE LA CHAINE DE MARKOV")
        print("="*70)
        
        print(f"Dimension : {self.n} etats")
        print(f"Stochastique : {self.is_stochastic()}")
        
        # Afficher les etats
        self._afficher_etats()
        
        # Classes
        self._afficher_classes()
        
        # Periodicite
        if with_periodicity:
            self.analyze_periodicity(method=periodicity_method, verbose=True)
        
        # Absorption
        self._afficher_absorption(initial_state)
    
    def _afficher_etats(self):
        """Affiche la liste des etats avec leur type."""
        print("\nEtats :")
        for i, state in enumerate(self.states):
            typ = self.absorbing_states.get(state, 'transitoire')
            print(f"  {i}: {state} [{typ}]")
    
    def _afficher_classes(self):
        """Affiche les classes de communication."""
        classes = self.analyze_classes()
        print(f"\nClasses de communication : {len(classes)}")
        for k, cls in enumerate(classes):
            etats_str = str(cls['etats'][:3])
            if len(cls['etats']) > 3:
                etats_str = etats_str[:-1] + ", ...]"
            print(f"  Classe {k+1} [{cls['type']}] : {etats_str}")
    
    def _afficher_absorption(self, initial_state):
        """Affiche l'analyse d'absorption."""
        ab = self.absorption_analysis()
        if ab and initial_state and initial_state in self.state_to_idx:
            i0 = self.state_to_idx[initial_state]
            if i0 in ab['trans_indices']:
                loc = ab['trans_indices'].index(i0)
                print(f"\nAbsorption depuis {initial_state} :")
                for j, ai in enumerate(ab['abs_indices']):
                    s = self.idx_to_state[ai]
                    typ = self.absorbing_states.get(s, '?')
                    prob = ab['B'][loc, j]
                    print(f"  P(-> {s} [{typ}]) = {prob:.6f}")
                print(f"  E[T] = {ab['t'][loc]:.4f} etapes")
                
                # Afficher la matrice fondamentale (extrait)
                if ab['N'].shape[0] <= 5:
                    print(f"\nMatrice fondamentale N :")
                    print(ab['N'])
                else:
                    print(f"\nMatrice fondamentale N (extrait 3x3) :")
                    print(ab['N'][:3, :3])


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TEST DE MarkovChain AVEC ANALYSE DE PERIODICITE")
    print("="*70)
    
    try:
        from astar import Grid, AStar
        print("[OK] Module astar importe avec succes")
    except ImportError as e:
        print(f"[Attention] Module astar non trouve: {e}")
        print("Creation d'une grille simple pour le test...")
        
        # Definition simple de Grid pour le test
        class Grid:
            def __init__(self, width, height, obstacles=None):
                self.width = width
                self.height = height
                self.obstacles = set(obstacles) if obstacles else set()
            
            def is_free(self, x, y):
                return (0 <= x < self.width and 
                        0 <= y < self.height and 
                        (x, y) not in self.obstacles)
        
        class AStar:
            def __init__(self, grid):
                self.grid = grid
            
            def search(self, start, goal):
                # Chemin simplifie pour le test
                return {
                    'success': True,
                    'path': [(0,0), (0,1), (0,2), (0,3), (0,4), 
                            (1,4), (2,4), (3,4), (4,4)],
                    'cost': 8
                }
            
            def extract_policy(self, path):
                policy = {}
                for i in range(len(path)-1):
                    dx = path[i+1][0] - path[i][0]
                    dy = path[i+1][1] - path[i][1]
                    policy[path[i]] = (dx, dy)
                policy[path[-1]] = (0, 0)
                return policy
    
    # Creer une grille et planifier
    grid = Grid(5, 5, obstacles=[(2, 2)])
    astar = AStar(grid)
    result = astar.search((0, 0), (4, 4))
    
    if result['success']:
        policy = astar.extract_policy(result['path'])
        
        # Construire la chaine Markov
        absorbing = {(4, 4): 'GOAL', (-1, -1): 'FAIL'}
        all_states = set(result['path']) | {(-1, -1)}
        
        mc = MarkovChain(all_states, absorbing)
        mc.build_from_policy(policy, grid, epsilon=0.2)
        
        print(f"\n[OK] Chaine construite avec {mc.n} etats")
        print(f"[OK] P stochastique : {mc.is_stochastic()}")
        
        # Afficher la matrice P
        print("\nMatrice P (extrait des 5 premiers etats) :")
        print(mc.P[:5, :5])
        
        # Test de periodicite
        print("\n" + "="*60)
        print("TEST METHODE : Par puissances de P")
        print("="*60)
        mc.analyze_periodicity(method='puissances', verbose=True, max_iter=50)
        
        # Generer les graphiques
        print("\n" + "="*60)
        print("GENERATION DES GRAPHIQUES DE PERIODICITE")
        print("="*60)
        
        print("\nGeneration avec methode 'puissances'...")
        mc.plot_periodicity(method='puissances', save_fig=True, show=True)
        
        # Analyse complete
        print("\n" + "="*60)
        print("ANALYSE COMPLETE")
        print("="*60)
        mc.print_analysis((0, 0), with_periodicity=True, periodicity_method='puissances')
        
        print("\n[OK] Test reussi !")
    else:
        print("[Erreur] Echec de la planification")