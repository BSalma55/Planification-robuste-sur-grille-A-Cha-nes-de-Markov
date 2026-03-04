# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict

class MarkovSimulation:
    """Simulation Monte-Carlo de la chaîne de Markov"""
    
    def __init__(self, markov_chain):
        self.mc = markov_chain
        
    def simulate_trajectory(self, start_state, max_steps=100):
        """
        Simule une trajectoire unique
        Retourne: (trajectoire, temps_arrivee, etat_final)
        """
        if start_state not in self.mc.state_to_idx:
            return None, None, None
        
        trajectory = [start_state]
        current_idx = self.mc.state_to_idx[start_state]
        
        for step in range(max_steps):
            # Distribution de probabilité pour l'état suivant
            probs = self.mc.P[current_idx]
            
            # Tirer l'état suivant
            next_idx = np.random.choice(self.mc.n, p=probs)
            next_state = self.mc.idx_to_state[next_idx]
            
            trajectory.append(next_state)
            
            # Vérifier si état absorbant
            if next_state in self.mc.absorbing_states:
                return trajectory, step + 1, next_state
            
            current_idx = next_idx
        
        # Max steps atteint sans absorption
        return trajectory, max_steps, trajectory[-1]
    
    def run_simulations(self, start_state, n_simulations=1000, max_steps=100):
        """
        Lance N simulations et collecte les statistiques
        """
        stats = {
            'goal_count': 0,
            'fail_count': 0,
            'other_count': 0,
            'times_to_goal': [],
            'times_to_fail': [],
            'trajectories': [],
            'final_states': defaultdict(int)
        }
        
        for _ in range(n_simulations):
            traj, time, final = self.simulate_trajectory(start_state, max_steps)
            
            if final is None:
                continue
                
            stats['trajectories'].append(traj)
            stats['final_states'][final] += 1
            
            if final in self.mc.absorbing_states:
                abs_type = self.mc.absorbing_states[final]
                if abs_type == 'GOAL':
                    stats['goal_count'] += 1
                    stats['times_to_goal'].append(time)
                elif abs_type == 'FAIL':
                    stats['fail_count'] += 1
                    stats['times_to_fail'].append(time)
            else:
                stats['other_count'] += 1
        
        return stats
    
    def analyze_simulations(self, stats, n_simulations):
        """Analyse les résultats des simulations"""
        print("\n" + "="*60)
        print("RÉSULTATS DE SIMULATION MONTE-CARLO")
        print("="*60)
        
        # Probabilités empiriques
        p_goal = stats['goal_count'] / n_simulations
        p_fail = stats['fail_count'] / n_simulations
        p_other = stats['other_count'] / n_simulations
        
        print(f"\nProbabilités empiriques (sur {n_simulations} simulations):")
        print(f"  P(atteindre GOAL): {p_goal:.4f}")
        print(f"  P(atteindre FAIL): {p_fail:.4f}")
        print(f"  P(autre): {p_other:.4f}")
        
        # Temps moyens
        if stats['times_to_goal']:
            mean_time_goal = np.mean(stats['times_to_goal'])
            std_time_goal = np.std(stats['times_to_goal'])
            print(f"\nTemps pour atteindre GOAL:")
            print(f"  Moyenne: {mean_time_goal:.2f} étapes")
            print(f"  Écart-type: {std_time_goal:.2f}")
            print(f"  Min: {min(stats['times_to_goal'])}")
            print(f"  Max: {max(stats['times_to_goal'])}")
        
        if stats['times_to_fail']:
            mean_time_fail = np.mean(stats['times_to_fail'])
            print(f"\nTemps pour atteindre FAIL:")
            print(f"  Moyenne: {mean_time_fail:.2f} étapes")
        
        # Distribution des états finaux
        print("\nDistribution des états finaux:")
        for state, count in sorted(stats['final_states'].items(), 
                                   key=lambda x: x[1], reverse=True)[:5]:
            abs_type = self.mc.absorbing_states.get(state, 'transitoire')
            print(f"  {state} ({abs_type}): {count} ({count/n_simulations:.4f})")
        
        return {
            'p_goal': p_goal,
            'p_fail': p_fail,
            'p_other': p_other,
            'mean_time_goal': np.mean(stats['times_to_goal']) if stats['times_to_goal'] else None,
            'mean_time_fail': np.mean(stats['times_to_fail']) if stats['times_to_fail'] else None
        }
    
    def compare_with_theory(self, stats, n_simulations, max_steps=100):
        """Compare les résultats empiriques avec la théorie"""
        print("\n" + "="*60)
        print("COMPARAISON SIMULATION VS THÉORIE")
        print("="*60)
        
        # Calcul théorique avec P^n
        start_state = stats['trajectories'][0][0] if stats['trajectories'] else None
        if start_state:
            i0 = self.mc.state_to_idx[start_state]
            
            # Distribution théorique après max_steps
            pi_n = self.mc.get_distribution(start_state, max_steps)
            
            print(f"\nAprès {max_steps} étapes:")
            print("  État\t\tThéorique\tEmpirique")
            
            for state in self.mc.states:
                if state in self.mc.absorbing_states:
                    idx = self.mc.state_to_idx[state]
                    p_theo = pi_n[idx]
                    p_emp = stats['final_states'].get(state, 0) / n_simulations
                    
                    abs_type = self.mc.absorbing_states[state]
                    print(f"  {state} ({abs_type}):\t{p_theo:.4f}\t\t{p_emp:.4f}")


# Test
if __name__ == "__main__":
    from markov import MarkovChain
    
    # Créer une chaîne simple
    states = [(0,0), (0,1), (1,0), (1,1)]
    absorbing = {(1,1): 'GOAL'}
    mc = MarkovChain(states, absorbing)
    
    mc.add_transition((0,0), (0,1), 0.8)
    mc.add_transition((0,0), (1,0), 0.2)
    mc.add_transition((0,1), (1,1), 0.9)
    mc.add_transition((0,1), (0,0), 0.1)
    mc.add_transition((1,0), (1,1), 0.7)
    mc.add_transition((1,0), (0,0), 0.3)
    mc.add_transition((1,1), (1,1), 1.0)
    mc.normalize_rows()
    
    # Simulations
    sim = MarkovSimulation(mc)
    stats = sim.run_simulations((0,0), n_simulations=1000, max_steps=20)
    
    # Analyse
    sim.analyze_simulations(stats, 1000)
    sim.compare_with_theory(stats, 1000, max_steps=20)