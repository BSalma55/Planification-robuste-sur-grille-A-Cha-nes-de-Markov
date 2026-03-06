# -*- coding: utf-8 -*-


import numpy as np
from collections import defaultdict


class MarkovSimulation:
    """
    Simule des trajectoires stochastiques selon une chaîne de Markov.
    Permet de valider les résultats théoriques par empirisme.
    """
    
    def __init__(self, markov_chain):
        """
        Paramètres
        ----------
        markov_chain : MarkovChain
            La chaîne de Markov à simuler
        """
        self.mc = markov_chain
    
    def simulate_trajectory(self, start_state, max_steps=200):
        """
        Simule une unique trajectoire stochastique.
        
        À chaque pas, l'état suivant est tiré selon la ligne
        courante de P avec np.random.choice().
        
        Paramètres
        ----------
        start_state : tuple
            État initial
        max_steps : int
            Limite de pas (pour éviter boucles infinies)
        
        Retourne
        --------
        tuple
            (trajectory, time_to_absorption, final_state)
        """
        if start_state not in self.mc.state_to_idx:
            return None, None, None
        
        trajectory = [start_state]
        current_idx = self.mc.state_to_idx[start_state]
        
        for step in range(max_steps):
            # Tirer l'état suivant selon P[current_idx]
            probs = self.mc.P[current_idx]
            next_idx = np.random.choice(self.mc.n, p=probs)
            next_state = self.mc.idx_to_state[next_idx]
            
            trajectory.append(next_state)
            
            # Arrêt si état absorbant
            if next_state in self.mc.absorbing_states:
                return trajectory, step + 1, next_state
            
            current_idx = next_idx
        
        # Max_steps atteint sans absorption
        return trajectory, max_steps, trajectory[-1]
    
    def run_simulations(self, start_state, n_simulations=1000, max_steps=200):
        """
        Lance n simulations indépendantes.
        
        Paramètres
        ----------
        start_state : tuple
            État initial
        n_simulations : int
            Nombre de trajectoires à simuler
        max_steps : int
            Limite de pas par trajectoire
        
        Retourne
        --------
        dict
            Statistiques collectées
        """
        stats = {
            'goal_count': 0,
            'fail_count': 0,
            'other_count': 0,
            'times_to_goal': [],
            'times_to_fail': [],
            'final_states': defaultdict(int),
            'start_state': start_state,
            'n_simulations': n_simulations,
            'max_steps': max_steps
        }
        
        for _ in range(n_simulations):
            _, time_abs, final = self.simulate_trajectory(start_state, max_steps)
            
            if final is None:
                continue
            
            stats['final_states'][final] += 1
            
            if final in self.mc.absorbing_states:
                abs_type = self.mc.absorbing_states[final]
                if abs_type == 'GOAL':
                    stats['goal_count'] += 1
                    stats['times_to_goal'].append(time_abs)
                elif abs_type == 'FAIL':
                    stats['fail_count'] += 1
                    stats['times_to_fail'].append(time_abs)
            else:
                stats['other_count'] += 1
        
        return stats
    
    def analyze_simulations(self, stats, verbose=True):
        """
        Analyse et affiche les résultats de simulation.
        
        Paramètres
        ----------
        stats : dict
            Résultats de run_simulations()
        verbose : bool
            Afficher les résultats
        
        Retourne
        --------
        dict
            Résumé statistique
        """
        N = stats['n_simulations']
        
        p_goal = stats['goal_count'] / N if N > 0 else 0.0
        p_fail = stats['fail_count'] / N if N > 0 else 0.0
        p_other = stats['other_count'] / N if N > 0 else 0.0
        
        t_goal_mean = np.mean(stats['times_to_goal']) if stats['times_to_goal'] else None
        t_goal_std = np.std(stats['times_to_goal']) if stats['times_to_goal'] else None
        t_fail_mean = np.mean(stats['times_to_fail']) if stats['times_to_fail'] else None
        
        if verbose:
            print("\n" + "="*70)
            print("RÉSULTATS SIMULATION MONTE-CARLO")
            print("="*70)
            print(f"Nombre de simulations : {N}")
            print(f"\nProbabilités empiriques :")
            print(f"  P(atteindre GOAL) = {p_goal:.4f}")
            print(f"  P(atteindre FAIL) = {p_fail:.4f}")
            print(f"  P(non absorbé) = {p_other:.4f}")
            
            if t_goal_mean is not None:
                print(f"\nTemps pour atteindre GOAL :")
                print(f"  Moyenne = {t_goal_mean:.2f} étapes")
                print(f"  Écart-type = {t_goal_std:.2f}")
                print(f"  Min = {min(stats['times_to_goal'])}")
                print(f"  Max = {max(stats['times_to_goal'])}")
            
            if t_fail_mean is not None:
                print(f"\nTemps pour atteindre FAIL :")
                print(f"  Moyenne = {t_fail_mean:.2f} étapes")
        
        return {
            'p_goal': p_goal,
            'p_fail': p_fail,
            'p_other': p_other,
            't_goal_mean': t_goal_mean,
            't_goal_std': t_goal_std,
            't_fail_mean': t_fail_mean
        }
    
    def compare_with_theory(self, stats, max_steps=30, verbose=True):
        """
        Compare résultats empiriques vs théoriques.
        
        Calcule π^(n) = π^(0) · P^n et compare avec
        les fréquences observées en simulation.
        
        Paramètres
        ----------
        stats : dict
            Résultats de run_simulations()
        max_steps : int
            Nombre d'étapes pour π^(n)
        verbose : bool
            Afficher les résultats
        """
        start_state = stats['start_state']
        N = stats['n_simulations']
        
        if verbose:
            print("\n" + "="*70)
            print("COMPARAISON THÉORIE vs EMPIRIQUE")
            print("="*70)
        
        # Distribution théorique via P^n
        try:
            pi_n = self.mc.get_distribution(start_state, max_steps)
            
            if verbose:
                print(f"\nAprès {max_steps} étapes :")
                print(f"{'État':<20} {'Type':<10} {'Théorique':>12} {'Empirique':>12}")
                print("-"*54)
                
                for state in self.mc.states:
                    if state in self.mc.absorbing_states:
                        idx = self.mc.state_to_idx[state]
                        p_th = pi_n[idx]
                        p_emp = stats['final_states'].get(state, 0) / N
                        typ = self.mc.absorbing_states[state]
                        
                        print(f"{str(state):<20} {typ:<10} {p_th:>12.6f} {p_emp:>12.6f}")
        except Exception as e:
            if verbose:
                print(f"Erreur lors du calcul de π^(n) : {e}")
        
        # Absorption
        ab = self.mc.absorption_analysis()
        if ab and start_state in self.mc.state_to_idx:
            i0 = self.mc.state_to_idx[start_state]
            if i0 in ab['trans_indices']:
                loc = ab['trans_indices'].index(i0)
                
                if verbose:
                    print(f"\nVia analyse d'absorption :")
                    print(f"{'État absorbant':<20} {'Théorique':>12} {'Empirique':>12} {'Écart':>10}")
                    print("-"*54)
                    
                    for j, ai in enumerate(ab['abs_indices']):
                        s = self.mc.idx_to_state[ai]
                        p_th = ab['B'][loc, j]
                        p_emp = stats['final_states'].get(s, 0) / N
                        ecart = abs(p_th - p_emp)
                        
                        print(f"{str(s):<20} {p_th:>12.6f} {p_emp:>12.6f} {ecart:>10.6f}")
                    
                    t_th = ab['t'][loc]
                    t_emp = np.mean(stats['times_to_goal']) if stats['times_to_goal'] else None
                    
                    print(f"\nTemps moyen avant absorption :")
                    print(f"  Théorique : {t_th:.4f} étapes")
                    if t_emp is not None:
                        print(f"  Empirique : {t_emp:.4f} étapes")
                        print(f"  Écart relatif : {abs(t_th - t_emp) / max(t_th, 1e-9) * 100:.2f}%")


# Test basique
if __name__ == "__main__":
    from Grid import Grid
    from astar import AStar
    from markov import MarkovChain
    
    # Créer et planifier
    grid = Grid(5, 5, obstacles=[(2, 2)])
    astar = AStar(grid)
    result = astar.search((0, 0), (4, 4))
    
    if result['success']:
        policy = astar.extract_policy(result['path'])
        
        absorbing = {(4, 4): 'GOAL', (-1, -1): 'FAIL'}
        all_states = set(result['path']) | {(-1, -1)}
        
        mc = MarkovChain(all_states, absorbing)
        mc.build_from_policy(policy, grid, epsilon=0.2)
        
        # Simuler
        sim = MarkovSimulation(mc)
        stats = sim.run_simulations((0, 0), n_simulations=1000, max_steps=50)
        
        # Analyser
        sim.analyze_simulations(stats)
        sim.compare_with_theory(stats, max_steps=20)