import numpy as np
import networkx as nx
from collections import defaultdict

class MarkovChain:
    """Chaîne de Markov à temps discret pour modéliser l'incertitude"""
    
    def __init__(self, states, absorbing_states=None):
        """
        states: liste des états (tuples (x,y))
        absorbing_states: états absorbants (ex: GOAL, FAIL)
        """
        self.states = list(states)
        self.state_to_idx = {state: i for i, state in enumerate(self.states)}
        self.idx_to_state = {i: state for i, state in enumerate(self.states)}
        self.n = len(states)
        
        # Matrice de transition (initialisée à zéro)
        self.P = np.zeros((self.n, self.n))
        
        # États absorbants
        self.absorbing_states = absorbing_states if absorbing_states else {}
        self.absorbing_indices = {self.state_to_idx[s] for s in absorbing_states.keys() if s in self.state_to_idx}
        
    def add_transition(self, from_state, to_state, probability):
        """Ajoute une transition avec sa probabilité"""
        if from_state not in self.state_to_idx or to_state not in self.state_to_idx:
            # Ignorer les transitions vers des états non valides
            return
        
        i = self.state_to_idx[from_state]
        j = self.state_to_idx[to_state]
        self.P[i, j] += probability
        
    def normalize_rows(self):
        """Normalise les lignes pour que la matrice soit stochastique"""
        row_sums = self.P.sum(axis=1)
        # Éviter la division par zéro
        row_sums[row_sums == 0] = 1
        self.P = self.P / row_sums[:, np.newaxis]
        
    def is_stochastic(self):
        """Vérifie si la matrice est stochastique (lignes = 1)"""
        row_sums = self.P.sum(axis=1)
        return np.allclose(row_sums, 1.0)
    
    def build_from_policy(self, policy, grid, epsilon, fail_states=None):
        """
        Construit la matrice P à partir d'une politique
        policy: {state: action} action = (dx, dy)
        grid: objet Grid pour connaître les voisins
        epsilon: probabilité de déviation
        fail_states: états qui mènent à l'échec (obstacles)
        """
        # Ajouter GOAL et FAIL comme états absorbants si nécessaire
        goal_state = None
        fail_state = None
        
        # Identifier l'état but (celui avec action (0,0))
        for state, action in policy.items():
            if action == (0, 0):
                goal_state = state
                self.absorbing_states[goal_state] = 'GOAL'
                break
        
        if goal_state not in self.state_to_idx:
            raise ValueError("L'état but doit être dans la liste des états")
        
        # Ajouter FAIL si des états d'échec sont spécifiés
        if fail_states:
            fail_state = (-1, -1)  # État spécial pour FAIL
            while fail_state in self.state_to_idx:
                fail_state = (fail_state[0] - 1, fail_state[1] - 1)
            
            # Vérifier si FAIL n'est pas déjà dans les états
            if fail_state not in self.state_to_idx:
                self.states.append(fail_state)
                self.state_to_idx[fail_state] = self.n
                self.idx_to_state[self.n] = fail_state
                self.n += 1
                self.absorbing_states[fail_state] = 'FAIL'
                self.absorbing_indices.add(self.n - 1)
                
                # Redimensionner P
                new_P = np.zeros((self.n, self.n))
                new_P[:self.n-1, :self.n-1] = self.P
                self.P = new_P
        
        # Créer un ensemble de tous les états valides (ceux dans self.states)
        valid_states = set(self.states)
        
        # Remplir la matrice
        for state in self.states:
            if state in self.absorbing_states:
                # État absorbant
                i = self.state_to_idx[state]
                self.P[i, i] = 1.0
                continue
            
            if state == fail_state:
                continue
            
            i = self.state_to_idx[state]
            
            # Action recommandée par la politique
            if state in policy:
                action = policy[state]
                intended_next = (state[0] + action[0], state[1] + action[1])
            else:
                # Si pas de politique pour cet état (hors chemin), rester sur place
                intended_next = state
                action = (0, 0)
            
            # Vérifier si l'action est valide et si l'état destination existe
            if not grid.is_free(*intended_next) or intended_next not in valid_states:
                intended_next = state  # Reste sur place si collision ou état inexistant
            
            # Distribution des probabilités
            # - Aller vers intended_next avec probabilité 1-epsilon
            # - Déviation vers les voisins latéraux avec probabilité epsilon/2
            
            # Ajouter la transition principale
            if intended_next in valid_states:
                self.add_transition(state, intended_next, 1 - epsilon)
            else:
                # Si la destination n'est pas valide, rester sur place
                self.add_transition(state, state, 1 - epsilon)
            
            # Ajouter les déviations
            if action != (0, 0):
                # Déviations latérales
                lateral_moves = []
                if action[0] != 0:  # Mouvement horizontal
                    lateral_moves = [(0, 1), (0, -1)]
                elif action[1] != 0:  # Mouvement vertical
                    lateral_moves = [(1, 0), (-1, 0)]
                
                for lat_action in lateral_moves:
                    lat_next = (state[0] + lat_action[0], state[1] + lat_action[1])
                    
                    # Vérifier si la destination latérale est valide et existe
                    if not grid.is_free(*lat_next) or lat_next not in valid_states:
                        lat_next = state  # Reste sur place si collision ou état inexistant
                    
                    if lat_next in valid_states:
                        self.add_transition(state, lat_next, epsilon / 2)
                    else:
                        self.add_transition(state, state, epsilon / 2)
            
            # Vérifier les états d'échec
            if fail_states and state in fail_states and fail_state in valid_states:
                self.add_transition(state, fail_state, 1.0)
        
        # Normaliser
        self.normalize_rows()
        
    def get_Pn(self, n):
        """Calcule P^n"""
        return np.linalg.matrix_power(self.P, n)
    
    def get_distribution(self, initial_state, n):
        """Calcule π(n) = π(0) * P^n"""
        if initial_state not in self.state_to_idx:
            raise ValueError(f"L'état {initial_state} n'existe pas dans la chaîne")
        
        i0 = self.state_to_idx[initial_state]
        pi0 = np.zeros(self.n)
        pi0[i0] = 1.0
        
        Pn = self.get_Pn(n)
        return pi0 @ Pn
    
    def get_distribution_evolution(self, initial_state, max_steps):
        """Calcule l'évolution de la distribution sur max_steps"""
        if initial_state not in self.state_to_idx:
            raise ValueError(f"L'état {initial_state} n'existe pas dans la chaîne")
        
        distributions = []
        pi = np.zeros(self.n)
        pi[self.state_to_idx[initial_state]] = 1.0
        
        for _ in range(max_steps):
            distributions.append(pi.copy())
            pi = pi @ self.P
        
        return distributions
    
    def analyze_classes(self):
        """Identifie les classes de communication de la chaîne"""
        # Construire le graphe orienté
        G = nx.DiGraph()
        G.add_nodes_from(range(self.n))
        
        for i in range(self.n):
            for j in range(self.n):
                if self.P[i, j] > 0:
                    G.add_edge(i, j)
        
        # Composantes fortement connexes
        try:
            cfc = list(nx.strongly_connected_components(G))
        except:
            # Fallback si networkx échoue
            cfc = [{i} for i in range(self.n)]
        
        # Classifier les états
        classes = []
        for comp in cfc:
            # Vérifier si la composante est fermée
            is_closed = True
            for i in comp:
                for j in range(self.n):
                    if self.P[i, j] > 0 and j not in comp:
                        is_closed = False
                        break
                if not is_closed:
                    break
            
            class_type = 'absorbante' if is_closed else 'transitoire'
            
            # Vérifier si c'est un état absorbant
            if len(comp) == 1:
                i = next(iter(comp))
                if self.P[i, i] == 1.0:
                    class_type = 'absorbant'
            
            classes.append({
                'etats': [self.idx_to_state[idx] for idx in comp],
                'indices': list(comp),
                'type': class_type
            })
        
        return classes
    
    def absorption_analysis(self):
        """
        Analyse d'absorption pour les chaînes avec états absorbants
        Retourne les probabilités d'absorption et les temps moyens
        """
        if not self.absorbing_indices:
            return None, None
        
        # Indices des états transitoires et absorbants
        trans_indices = [i for i in range(self.n) if i not in self.absorbing_indices]
        abs_indices = list(self.absorbing_indices)
        
        if not trans_indices:
            return None, None
        
        # Matrice fondamentale N = (I - Q)^(-1)
        Q = self.P[np.ix_(trans_indices, trans_indices)]
        I = np.eye(len(trans_indices))
        
        try:
            N = np.linalg.inv(I - Q)
            
            # Matrice R : transitions des transitoires vers absorbants
            R = self.P[np.ix_(trans_indices, abs_indices)]
            
            # Probabilités d'absorption B = N * R
            B = N @ R
            
            # Temps moyen avant absorption t = N * 1
            t = N @ np.ones(len(trans_indices))
            
            return {
                'B': B,
                't': t,
                'trans_indices': trans_indices,
                'abs_indices': abs_indices
            }
        except np.linalg.LinAlgError:
            # Matrice non inversible
            return None, None
    
    def analyze_periodicity(self):
        """
        Analyse la périodicité des classes (optionnel)
        Retourne un dictionnaire avec les périodes de chaque classe
        """
        print("\n" + "="*60)
        print("ANALYSE DE PERIODICITE (OPTIONNEL)")
        print("="*60)
        
        classes = self.analyze_classes()
        periodicity_results = []
        
        for i, cls in enumerate(classes):
            if cls['type'] == 'absorbant':
                print(f"  Classe {i+1} (absorbant): periode = 1 (triviale)")
                periodicity_results.append({
                    'classe': i+1,
                    'etats': cls['etats'],
                    'type': cls['type'],
                    'periode': 1,
                    'est_periodique': False
                })
                continue
            
            # Prendre le premier état de la classe pour analyser la période
            etat_idx = cls['indices'][0]
            
            # Calculer la période (méthode simplifiée)
            periode = None
            for k in range(1, min(self.n * 2, 10)):  # Limiter pour éviter trop de calculs
                Pk = self.get_Pn(k)
                if Pk[etat_idx, etat_idx] > 1e-10:  # Seuil de tolérance
                    periode = k
                    break
            
            if periode is None:
                periode = 0  # Non déterminé
            
            est_periodique = periode > 1
            
            if periode == 1:
                print(f"  Classe {i+1} ({cls['type']}): periode = 1 (Aperiodique)")
            elif periode > 1:
                print(f"  Classe {i+1} ({cls['type']}): periode = {periode} (Periodique)")
            else:
                print(f"  Classe {i+1} ({cls['type']}): periode non determinee")
            
            periodicity_results.append({
                'classe': i+1,
                'etats': cls['etats'],
                'type': cls['type'],
                'periode': periode,
                'est_periodique': est_periodique
            })
        
        # Analyse approfondie pour les classes périodiques
        classes_periodiques = [r for r in periodicity_results if r['est_periodique']]
        if classes_periodiques:
            print(f"\n  -> {len(classes_periodiques)} classe(s) periodique(s) detectee(s)")
            for r in classes_periodiques:
                print(f"    - Classe {r['classe']}: periode {r['periode']}")
        else:
            print("\n  -> Aucune classe periodique detectee (chaine aperiodique)")
        
        return periodicity_results
    
    def print_analysis(self, initial_state=None, with_periodicity=False):
        """Affiche une analyse complète de la chaîne"""
        print("\n" + "="*60)
        print("ANALYSE DE LA CHAINE DE MARKOV")
        print("="*60)
        
        # Vérification stochastique
        print(f"\nMatrice stochastique: {self.is_stochastic()}")
        
        # Analyse des classes
        classes = self.analyze_classes()
        print(f"\nNombre de classes: {len(classes)}")
        for i, cls in enumerate(classes):
            print(f"  Classe {i+1} ({cls['type']}): {cls['etats']}")
        
        # Analyse d'absorption
        absorption = self.absorption_analysis()
        if absorption:
            print("\nANALYSE D'ABSORPTION:")
            
            # Pour l'état initial spécifié
            if initial_state and initial_state in self.state_to_idx:
                i0 = self.state_to_idx[initial_state]
                if i0 in absorption['trans_indices']:
                    idx_in_trans = absorption['trans_indices'].index(i0)
                    
                    print(f"\nDepuis l'etat {initial_state}:")
                    
                    # Probabilités d'absorption
                    for j, abs_idx in enumerate(absorption['abs_indices']):
                        prob = absorption['B'][idx_in_trans, j]
                        abs_state = self.idx_to_state[abs_idx]
                        abs_type = self.absorbing_states.get(abs_state, 'inconnu')
                        print(f"  Probabilite d'atteindre {abs_state} ({abs_type}): {prob:.4f}")
                    
                    # Temps moyen
                    t_mean = absorption['t'][idx_in_trans]
                    print(f"  Temps moyen avant absorption: {t_mean:.2f} etapes")
                else:
                    print(f"\nL'etat {initial_state} est deja absorbant")
            else:
                print("\nAucun etat initial specifie ou etat non trouve")
        else:
            print("\nPas d'analyse d'absorption disponible")
        
        # Analyse de périodicité (optionnelle)
        if with_periodicity:
            self.analyze_periodicity()


# Test
if __name__ == "__main__":
    # Créer une chaîne simple
    states = [(0,0), (0,1), (1,0), (1,1)]
    absorbing = {(1,1): 'GOAL'}
    mc = MarkovChain(states, absorbing)
    
    # Ajouter des transitions
    mc.add_transition((0,0), (0,1), 0.8)
    mc.add_transition((0,0), (1,0), 0.2)
    mc.add_transition((0,1), (1,1), 0.9)
    mc.add_transition((0,1), (0,0), 0.1)
    mc.add_transition((1,0), (1,1), 0.7)
    mc.add_transition((1,0), (0,0), 0.3)
    mc.add_transition((1,1), (1,1), 1.0)
    
    mc.normalize_rows()
    
    print("Matrice P:")
    print(mc.P)
    print("\nSomme des lignes:", mc.P.sum(axis=1))
    
    print("\nP^2:")
    print(mc.get_Pn(2))
    
    print("\nDistribution apres 3 etapes depuis (0,0):")
    print(mc.get_distribution((0,0), 3))
    
    # Analyse complète avec périodicité
    mc.print_analysis((0,0), with_periodicity=True)