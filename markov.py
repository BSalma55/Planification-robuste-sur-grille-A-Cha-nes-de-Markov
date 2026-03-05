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

        if with_periodicity:
            self.analyze_periodicity()


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
