<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
========================================================
markov.py  --  PHASES 3 & 4 : Chaine de Markov
========================================================
CDC ?4.3 :
  P3.1  Construire P sur les etats accessibles (grille + GOAL/FAIL)
  P3.2  Verifier que P est stochastique
  P3.3  Calculer pi(n) = pi(0) . P^n

CDC ?4.4 :
  P4.1  Graphe oriente des transitions (arc i->j si pij > 0)
  P4.2  Classes de communication, etats transitoires/persistants
  P4.3  Probabilites d'absorption + temps moyen (N, B, t)
  P4.4  Periodicite / aperiodicite  [option]

CDC ?3.2 / ?3.3 :
  Modele d'incertitude epsilon, etats absorbants GOAL et FAIL
  pi(n) = pi(0) . P^n  (Chapman-Kolmogorov)
  N = (I-Q)-1,  B = N.R,  t = N.1
"""

import numpy as np
import networkx as nx


# ----------------------------------------------------------------
# CHAINE DE MARKOV  (CDC ?3.2, ?3.3, ?4.3, ?4.4)
# ----------------------------------------------------------------
class MarkovChain:
    """
    Chaine de Markov a temps discret {Xn}n>=0 sur S.

    Matrice de transition P = (pij) stochastique par lignes :
        P(Xn+1 = j | Xn = i) = pij     (CDC ?3.2)

    Evolution (Chapman-Kolmogorov) :
        pi(n) = pi(0) . P^n               (CDC ?3.2)

    Etats absorbants (CDC ?3.3) :
        GOAL : pGOAL,GOAL = 1  -> cible atteinte
        FAIL : pFAIL,FAIL = 1  -> echec irreversible

    Decomposition canonique :
        P = [ I  0 ]   Q : transitoire -> transitoire
            [ R  Q ]   R : transitoire -> absorbant

    Matrice fondamentale  N = (I ? Q)-1
    Probas d'absorption   B = N . R
    Temps moyen absorpt.  t = N . 1      (CDC ?4.3 / P4.3)
    """

    def __init__(self, states, absorbing_states=None):
        """
        Parametres
        ----------
        states           : ensemble/liste d'etats  (tuples + speciaux)
        absorbing_states : dict {etat: 'GOAL' | 'FAIL'}
        """
        self.states           = list(states)
        self.n                = len(self.states)
        self.state_to_idx     = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state     = {i: s for i, s in enumerate(self.states)}
        self.P                = np.zeros((self.n, self.n))
        self.absorbing_states = dict(absorbing_states) if absorbing_states else {}
        self.absorbing_indices = {
            self.state_to_idx[s]
            for s in self.absorbing_states
            if s in self.state_to_idx
        }

    # -- Construction de P  (CDC P3.1) ----------------------------

    def build_from_policy(self, policy, grid, epsilon, fail_states=None):
        """
        Construit la matrice de transition P depuis la politique A*.

        Modele d'incertitude (CDC ?3.2) :
            P(s -> s_voulue)  = 1 ? epsilon   action recommandee par pi(s)
            P(s -> s_lat1)    = epsilon/2     deviation laterale gauche
            P(s -> s_lat2)    = epsilon/2     deviation laterale droite
            -> self-loop si la destination est hors grille ou obstacle

        Parametres
        ----------
        policy      : {(x,y): (dx,dy)}  sortie de AStar.extract_policy()
        grid        : objet Grid
        epsilon     : epsilon dans [0,1]  taux d'incertitude  (CDC P1.3)
        fail_states : set d'obstacles (ignore ici, FAIL gere par self-loop)
        """
        valid = set(self.states)

        for state in self.states:
            i = self.state_to_idx[state]

            # -- Etat absorbant : boucle sur lui-meme -------------
            if state in self.absorbing_states:
                self.P[i, i] = 1.0
                continue

            # -- Action recommandee -------------------------------
            action = policy.get(state, (0, 0))
            dx, dy = action

            # Destination principale
            dest_main = (state[0] + dx, state[1] + dy)
            if not grid.is_free(*dest_main) or dest_main not in valid:
                dest_main = state          # self-loop si bloque

            self._add(i, dest_main, 1.0 - epsilon)

            # -- Deviations laterales (CDC ?3.2) ------------------
            if action != (0, 0):
                # Perpendiculaire a la direction d'action
                laterals = [(0,1),(0,-1)] if dx != 0 else [(1,0),(-1,0)]

                for ldx, ldy in laterals:
                    dest_lat = (state[0]+ldx, state[1]+ldy)
                    if not grid.is_free(*dest_lat) or dest_lat not in valid:
                        dest_lat = state   # self-loop si bloque
                    self._add(i, dest_lat, epsilon / 2.0)

        # Normaliser pour garantir la propriete stochastique (CDC P3.2)
        self._normalize()

    def _add(self, i, dest, prob):
        """Incremente P[i, j] (j = indice de dest)."""
        j = self.state_to_idx.get(dest)
        if j is not None:
            self.P[i, j] += prob

    def _normalize(self):
        """Normalise chaque ligne de P a 1.0 (propriete stochastique)."""
        row_sums = self.P.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.P /= row_sums

    # -- Verification stochastique  (CDC P3.2) --------------------

    def is_stochastic(self, tol=1e-9):
        """Retourne True si toutes les lignes de P somment a 1."""
        return bool(np.allclose(self.P.sum(axis=1), 1.0, atol=tol))

    # -- P^n  (CDC ?3.2) -------------------------------------------

    def get_Pn(self, n):
        """Calcule P^n par puissance matricielle (numpy)."""
        return np.linalg.matrix_power(self.P, n)

    # -- Distribution pi(n) = pi(0).P^n  (CDC P3.3) -----------------

    def get_distribution(self, initial_state, n):
        """
        Calcule pi(n) = pi(0) . P^n.

        pi(0) = vecteur unitaire e_{s0} (toute la masse sur initial_state).
        Retourne  np.array de taille self.n.
        """
        if initial_state not in self.state_to_idx:
            raise ValueError(f"Etat inconnu : {initial_state}")
        i0  = self.state_to_idx[initial_state]
        pi0 = np.zeros(self.n)
        pi0[i0] = 1.0
        return pi0 @ self.get_Pn(n)

    def get_distribution_evolution(self, initial_state, max_steps):
        """
        Retourne [pi(1), pi(2), ..., pi(max_steps)].

        Utile pour tracer P(GOAL) au fil du temps (CDC E.2 / P3.3).
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

    # -- Classes de communication  (CDC P4.1 / P4.2) --------------

    def analyze_classes(self):
        """
        Identifie les classes de communication via les composantes
        fortement connexes (CFC) du graphe oriente des transitions.

        CDC P4.1 : arc i->j si pij > 0
        CDC P4.2 : classification transitoire / absorbant

        Un etat est ABSORBANT si sa CFC est fermee (aucune sortie)
        et contient un self-loop de probabilite 1.
        Un etat est TRANSITOIRE si sa CFC n'est pas fermee.

        Retourne  list de dicts :
            {
              'etats'    : [etat, ...],
              'indices'  : [idx, ...],
              'type'     : 'absorbant' | 'transitoire'
            }
        """
        G = nx.DiGraph()
        G.add_nodes_from(range(self.n))
        for i in range(self.n):
            for j in range(self.n):
                if self.P[i, j] > 1e-12:
                    G.add_edge(i, j)

        classes = []
        for comp in nx.strongly_connected_components(G):
            # Classe fermee = aucune sortie vers l'exterieur de comp
            is_closed = all(
                j in comp
                for i in comp
                for j in range(self.n)
                if self.P[i, j] > 1e-12
            )
            if is_closed and len(comp) == 1:
                idx   = next(iter(comp))
                ctype = 'absorbant' if np.isclose(self.P[idx, idx], 1.0) else 'transitoire'
            elif is_closed:
                ctype = 'absorbant'
            else:
                ctype = 'transitoire'

            classes.append({
                'etats':   [self.idx_to_state[i] for i in comp],
                'indices': list(comp),
                'type':    ctype,
            })

        return classes

    # -- Analyse d'absorption  (CDC P4.3) -------------------------

    def absorption_analysis(self):
        """
        Analyse d'absorption (CDC ?3.3 / P4.3).

        Decomposition de P :
            P = [ I  0 ]    I : absorbant -> absorbant  (identite)
                [ R  Q ]    Q : transitoire -> transitoire
                            R : transitoire -> absorbant

        Matrice fondamentale :  N = (I ? Q)-1
            N[i,j] = E[ nombre de visites en j avant absorption | depart i ]

        Probabilites d'absorption :  B = N . R
            B[i,k] = P( absorber en k | depart i )

        Temps moyen avant absorption :  t = N . 1
            t[i]   = E[ temps avant absorption | depart i ]

        Retourne  dict ou None si pas d'etats absorbants.
        """
        if not self.absorbing_indices:
            return None

        trans_idx = [i for i in range(self.n) if i not in self.absorbing_indices]
        abs_idx   = sorted(self.absorbing_indices)

        if not trans_idx:
            return None

        Q = self.P[np.ix_(trans_idx, trans_idx)]
        R = self.P[np.ix_(trans_idx, abs_idx)]

        try:
            N = np.linalg.inv(np.eye(len(trans_idx)) - Q)
            B = N @ R
            t = N @ np.ones(len(trans_idx))
            return {
                'B':             B,
                't':             t,
                'N':             N,
                'trans_indices': trans_idx,
                'abs_indices':   abs_idx,
            }
        except np.linalg.LinAlgError:
            return None

    # -- Periodicite  (CDC P4.4 -- option) -------------------------

    def analyze_periodicity(self, verbose=True):
        """
        Analyse la periode de chaque classe de communication.
        CDC P4.4 (option).

        Methode : chercher le plus petit k >= 1 tel que P^k[i,i] > 0.
        Si k = 1 -> aperiodique. Si k > 1 -> periode k.

        Retourne  list de dicts par classe.
        """
        if verbose:
            print("\n" + "="*60)
            print("P4.4 -- PERIODICITE  (option)")
            print("="*60)

        classes = self.analyze_classes()
        results = []

        for idx_c, cls in enumerate(classes):
            if cls['type'] == 'absorbant':
                periode = 1
                label   = "triviale (self-loop)"
            else:
                i    = cls['indices'][0]
                Pk   = self.P.copy()
                periode = None
                for k in range(1, min(2 * self.n + 1, 100)):
                    if Pk[i, i] > 1e-10:
                        periode = k
                        break
                    Pk = Pk @ self.P
                periode = periode if periode is not None else 0
                label   = ("Aperiodique" if periode == 1
                           else f"Periodique d={periode}" if periode > 1
                           else "Non determinee")

            if verbose:
                print(f"  Classe {idx_c+1} [{cls['type']}] : "
                      f"periode = {periode} -- {label}")
            results.append({
                'classe':         idx_c + 1,
                'etats':          cls['etats'],
                'type':           cls['type'],
                'periode':        periode,
                'est_periodique': bool(periode is not None and periode > 1),
            })

        if verbose:
            periodic = [r for r in results if r['est_periodique']]
            msg = (f"{len(periodic)} classe(s) periodique(s)"
                   if periodic else "Aucune classe periodique -- chaine aperiodique")
            print(f"\n  => {msg}")

        return results

    # -- Affichage complet -----------------------------------------

    def print_analysis(self, initial_state=None, with_periodicity=False):
        """
        Affiche l'analyse complete de la chaine (CDC ?4.3 & ?4.4).
        """
        sep = "=" * 60
        print(f"\n{sep}")
        print("ANALYSE DE LA CHAINE DE MARKOV")
        print(sep)
        print(f"  Dimension P          : {self.n} x {self.n}")
        print(f"  Matrice stochastique : {self.is_stochastic()}")

        # Classes
        classes = self.analyze_classes()
        print(f"\n  Classes de communication : {len(classes)}")
        for k, cls in enumerate(classes):
            etats_str = str(cls['etats'][:5])
            if len(cls['etats']) > 5:
                etats_str = etats_str[:-1] + ", ...]"
            print(f"    Classe {k+1} [{cls['type']:12s}] : {etats_str}")

        # Absorption
        ab = self.absorption_analysis()
        if ab and initial_state and initial_state in self.state_to_idx:
            i0 = self.state_to_idx[initial_state]
            if i0 in ab['trans_indices']:
                loc = ab['trans_indices'].index(i0)
                print(f"\n  Absorption depuis {initial_state} :")
                for j, ai in enumerate(ab['abs_indices']):
                    s    = self.idx_to_state[ai]
                    typ  = self.absorbing_states.get(s, '?')
                    print(f"    P(-> {s} [{typ}]) = {ab['B'][loc, j]:.6f}")
                print(f"    E[T] avant absorption = {ab['t'][loc]:.4f} etapes")
            else:
                print(f"\n  {initial_state} est deja absorbant.")
        else:
            print("\n  Pas d'analyse d'absorption disponible.")

=======
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
>>>>>>> origin/main
        if with_periodicity:
            self.analyze_periodicity()


<<<<<<< HEAD
# ----------------------------------------------------------------
if __name__ == "__main__":
    print("=== Test markov.py ===")
    from astar import Grid, AStar
    g = Grid(5, 5, obstacles=[(2, 2)])
    a = AStar(g)
    r = a.search((0,0), (4,4))
    pl = a.extract_policy(r['path'])

    absorbing  = {(4,4): 'GOAL', (-1,-1): 'FAIL'}
    all_states = set(r['path']) | {(-1,-1)}
    mc = MarkovChain(all_states, absorbing)
    mc.build_from_policy(pl, g, epsilon=0.1)
    mc.print_analysis((0,0), with_periodicity=True)
=======
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
>>>>>>> origin/main
