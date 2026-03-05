# -*- coding: utf-8 -*-
"""
========================================================
simulation.py  --  PHASE 5 : Simulation Monte-Carlo
========================================================
CDC ?4.5 :
  P5.1  Simuler N trajectoires Markov depuis s?
  P5.2  Estimer P^(GOAL), distribution du temps, taux d'echec
  P5.3  Comparer simulation vs calcul matriciel (P^n, absorption)
"""

import numpy as np
from collections import defaultdict


# ----------------------------------------------------------------
# SIMULATION MONTE-CARLO  (CDC ?4.5)
# ----------------------------------------------------------------
class MarkovSimulation:
    """
    Simulation Monte-Carlo de la chaine de Markov.

    CDC P5.1 : a chaque pas, l'etat suivant est tire selon
               la ligne courante de P -> np.random.choice(P[i]).

    CDC P5.2 : collecte P^(GOAL), P^(FAIL), T_moy, sigma(T).
    CDC P5.3 : comparaison empirique vs theorique (P^n, B, t).
    """

    def __init__(self, markov_chain):
        self.mc = markov_chain

    # -- Trajectoire unique  (CDC P5.1) ---------------------------

    def simulate_trajectory(self, start_state, max_steps=200):
        """
        Simule une trajectoire depuis start_state.

        Parametres
        ----------
        start_state : etat initial s?
        max_steps   : limite de pas (evite les boucles infinies)

        Retourne
        --------
        trajectory  : liste d'etats visites
        time_abs    : pas auquel l'absorption a eu lieu (ou max_steps)
        final_state : dernier etat
        """
        if start_state not in self.mc.state_to_idx:
            return None, None, None

        trajectory  = [start_state]
        current_idx = self.mc.state_to_idx[start_state]

        for step in range(max_steps):
            # Tirage selon la distribution de la ligne courante de P
            next_idx   = np.random.choice(self.mc.n, p=self.mc.P[current_idx])
            next_state = self.mc.idx_to_state[next_idx]
            trajectory.append(next_state)

            # Arret sur etat absorbant
            if next_state in self.mc.absorbing_states:
                return trajectory, step + 1, next_state

            current_idx = next_idx

        return trajectory, max_steps, trajectory[-1]

    # -- N simulations  (CDC P5.1 / P5.2) -------------------------

    def run_simulations(self, start_state, n_simulations=1000, max_steps=200):
        """
        Lance n_simulations trajectoires depuis start_state.

        Retourne  dict de statistiques :
            goal_count      : nombre de trajectoires atteignant GOAL
            fail_count      : nombre de trajectoires atteignant FAIL
            other_count     : non absorbees avant max_steps
            times_to_goal   : liste des temps d'atteinte de GOAL
            times_to_fail   : liste des temps d'atteinte de FAIL
            final_states    : compteur {etat_final: nb occurrences}
        """
        stats = {
            'goal_count':    0,
            'fail_count':    0,
            'other_count':   0,
            'times_to_goal': [],
            'times_to_fail': [],
            'final_states':  defaultdict(int),
            'start_state':   start_state,
            'n_simulations': n_simulations,
            'max_steps':     max_steps,
        }

        for _ in range(n_simulations):
            _, time_abs, final = self.simulate_trajectory(start_state, max_steps)
            if final is None:
                continue

            stats['final_states'][final] += 1

            if final in self.mc.absorbing_states:
                typ = self.mc.absorbing_states[final]
                if typ == 'GOAL':
                    stats['goal_count']   += 1
                    stats['times_to_goal'].append(time_abs)
                elif typ == 'FAIL':
                    stats['fail_count']   += 1
                    stats['times_to_fail'].append(time_abs)
            else:
                stats['other_count'] += 1

        return stats

    # -- Analyse des resultats  (CDC P5.2) ------------------------

    def analyze_simulations(self, stats, verbose=True):
        """
        Affiche et retourne les statistiques de simulation.
        CDC P5.2 : P^(GOAL), P^(FAIL), distribution du temps d'atteinte.
        """
        N = stats['n_simulations']

        p_goal  = stats['goal_count']  / N
        p_fail  = stats['fail_count']  / N
        p_other = stats['other_count'] / N

        t_goal_mean = (float(np.mean(stats['times_to_goal']))
                       if stats['times_to_goal'] else None)
        t_goal_std  = (float(np.std(stats['times_to_goal']))
                       if stats['times_to_goal'] else None)
        t_fail_mean = (float(np.mean(stats['times_to_fail']))
                       if stats['times_to_fail'] else None)

        if verbose:
            print("\n" + "="*60)
            print("P5.2 -- RESULTATS SIMULATION MONTE-CARLO")
            print("="*60)
            print(f"  N simulations     : {N}")
            print(f"  max_steps         : {stats['max_steps']}")
            print(f"\n  P^(GOAL)           : {p_goal:.4f}")
            print(f"  P^(FAIL)           : {p_fail:.4f}")
            print(f"  P^(non absorbe)    : {p_other:.4f}")
            if t_goal_mean is not None:
                print(f"\n  Temps -> GOAL      : "
                      f"moy={t_goal_mean:.2f}  sigma={t_goal_std:.2f}  "
                      f"min={min(stats['times_to_goal'])}  "
                      f"max={max(stats['times_to_goal'])}")
            if t_fail_mean is not None:
                print(f"  Temps -> FAIL      : moy={t_fail_mean:.2f}")

        return {
            'p_goal':       p_goal,
            'p_fail':       p_fail,
            'p_other':      p_other,
            't_goal_mean':  t_goal_mean,
            't_goal_std':   t_goal_std,
            't_fail_mean':  t_fail_mean,
        }

    # -- Comparaison simulation vs theorie  (CDC P5.3) ------------

    def compare_with_theory(self, stats, max_steps=30, verbose=True):
        """
        Compare les resultats empiriques avec le calcul matriciel.
        CDC P5.3 : simulation vs P^n et analyse d'absorption.
        """
        start_state = stats['start_state']
        N           = stats['n_simulations']

        if verbose:
            print("\n" + "="*60)
            print("P5.3 -- COMPARAISON SIMULATION vs THEORIE")
            print("="*60)

        # -- Calcul theorique par P^n -----------------------------
        pi_n = self.mc.get_distribution(start_state, max_steps)

        if verbose:
            print(f"\n  Via P^n apres {max_steps} etapes :")
            print(f"  {'Etat':<20}  {'Type':<8}  "
                  f"{'Theorique (P^n)':>16}  {'Empirique':>12}")
            print("  " + "-"*60)
            for state in self.mc.states:
                if state in self.mc.absorbing_states:
                    idx   = self.mc.state_to_idx[state]
                    p_th  = pi_n[idx]
                    p_emp = stats['final_states'].get(state, 0) / N
                    typ   = self.mc.absorbing_states[state]
                    print(f"  {str(state):<20}  {typ:<8}  "
                          f"{p_th:>16.6f}  {p_emp:>12.6f}")

        # -- Calcul theorique par absorption ---------------------
        ab = self.mc.absorption_analysis()
        if ab and start_state in self.mc.state_to_idx:
            i0 = self.mc.state_to_idx[start_state]
            if i0 in ab['trans_indices']:
                loc = ab['trans_indices'].index(i0)
                if verbose:
                    print(f"\n  Via absorption (B, t) :")
                    for j, ai in enumerate(ab['abs_indices']):
                        s    = self.mc.idx_to_state[ai]
                        typ  = self.mc.absorbing_states.get(s, '?')
                        p_th = ab['B'][loc, j]
                        p_emp = stats['final_states'].get(s, 0) / N
                        print(f"    P(-> {s} [{typ}])  "
                              f"theo={p_th:.6f}  emp={p_emp:.6f}  "
                              f"ecart={abs(p_th - p_emp):.6f}")
                    t_th  = ab['t'][loc]
                    t_emp = (float(np.mean(stats['times_to_goal']))
                             if stats['times_to_goal'] else None)
                    if verbose:
                        print(f"\n    E[T] theorique (absorption) : {t_th:.4f}")
                        if t_emp is not None:
                            print(f"    E[T] empirique (simulation) : {t_emp:.4f}")
                            print(f"    Ecart relatif               : "
                                  f"{abs(t_th - t_emp)/max(t_th,1e-9)*100:.2f} %")


# ----------------------------------------------------------------
if __name__ == "__main__":
    print("=== Test simulation.py ===")
    from astar import Grid, AStar
    from markov import MarkovChain

    g  = Grid(5, 5, obstacles=[(2, 2)])
    a  = AStar(g)
    r  = a.search((0,0), (4,4))
    pl = a.extract_policy(r['path'])

    absorbing  = {(4,4): 'GOAL', (-1,-1): 'FAIL'}
    all_states = set(r['path']) | {(-1,-1)}
    mc = MarkovChain(all_states, absorbing)
    mc.build_from_policy(pl, g, epsilon=0.1)

    sim   = MarkovSimulation(mc)
    stats = sim.run_simulations((0,0), n_simulations=1000, max_steps=50)
    sim.analyze_simulations(stats)
    sim.compare_with_theory(stats, max_steps=30)
